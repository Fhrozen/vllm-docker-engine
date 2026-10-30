#!/usr/bin/env python3
"""Evaluate what a rule denies, and what the entry path actually needs.

Three questions per name, each answered by measurement rather than by reading
the rule:

  size   - wheel bytes (compressed download) and tree bytes (what ships)
  reach  - does a shipped file import it at module scope
  run    - is it executed at import time on the entry path

The runtime half is a real import of the entry module against the materialised
selection, in a fresh interpreter per round. When the import stops, the module
it stopped on is recorded, put back into the view, and the import is retried.
The loop ends when the entry import succeeds, so the recorded names are the
ones the entry path executes and therefore cannot be pruned. A name with static
reach that the entry path never reaches is a candidate for pruning, not a
proven prune: a feature path can still reach it later.

Nothing here runs on the host. Run it inside the container, e.g.

  bash docker/run.sh script tools/evaluate_deny.py --wheelhouse out/wheelhouse \
      --venv out/eval-venv --rule vllm

The gate half can be run alone, and a branch can be tested without editing the
rule, which is how the cupy/ray decision was evaluated on both sides:

  bash docker/run.sh script tools/evaluate_deny.py --wheelhouse out/wheelhouse \
      --venv out/eval-venv --rule vllm --gate --extra-prune \
      'ray/util/collective/collective_group/cuda_stream.py'
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packaging"))

import arch as arch_mod  # noqa: E402
import importscan  # noqa: E402
import package as pkg_mod  # noqa: E402
import rules as rules_mod  # noqa: E402

ENTRY_DEFAULT = "vllm.entrypoints.cli.main"
MAX_ROUNDS = 40

RUNNER = """
import builtins, json, sys
view, entry = sys.argv[1], sys.argv[2]
sys.path.insert(0, view)
real = builtins.__import__
ok, failed = {}, {}
def hook(name, globals=None, *a, **k):
    top = name.split(".")[0]
    caller = (globals or {}).get("__name__", "<entry>")
    try:
        module = real(name, globals, *a, **k)
    except ImportError:
        failed.setdefault(top, caller)
        raise
    ok.setdefault(top, caller)
    return module
builtins.__import__ = hook
missing, error = None, None
try:
    __import__(entry)
except ModuleNotFoundError as exc:
    missing = exc.name
except Exception as exc:
    error = f"{type(exc).__name__}: {exc}"
print(json.dumps({"missing": missing, "error": error, "ok": ok, "failed": failed}))
"""


def wheel_sizes(wheelhouse: Path, dist: str) -> tuple[int, int]:
    """Compressed bytes and wheel count for one distribution in the wheelhouse."""
    total = 0
    count = 0
    for path in sorted(wheelhouse.glob("*.whl")):
        if rules_mod.norm(path.name.split("-")[0]) == dist:
            total += path.stat().st_size
            count += 1
    return total, count


def tree_sizes(tree: Path, top_levels: list[str]) -> tuple[int, int]:
    """Bytes and file count of a distribution's own files in the installed tree."""
    total = 0
    count = 0
    for top in top_levels:
        folder = tree / top
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
                count += 1
    return total, count


def sum_sizes(tree: Path, selected: set[str]) -> int:
    """Bytes of the selection only, which is what the image actually carries."""
    return sum((tree / rel).stat().st_size for rel in selected)


def gate_report(
    tree: Path,
    rule: dict,
    owner: dict[str, str],
    selected: set[str],
) -> tuple[list[str], list[str]]:
    """The two halves of the deny gate, answered for one selection.

    First half: a shipped file imports a denied distribution at module scope.
    Second half: a shipped file imports a module the selection drops, either as
    a whole top-level folder or as a single file inside a kept distribution.
    Both are reported, because a selection that passes only the first half can
    still be broken at runtime.
    """
    kept = {rel.split("/")[0] for rel in selected}
    dangling = []
    for rel in sorted(selected):
        if not rel.endswith(".py"):
            continue
        for module in importscan.module_level_modules(tree / rel):
            top = module.split(".")[0]
            if (tree / top).is_dir() and top not in kept:
                dangling.append(
                    f"{rel}: imports {module}, but the selection keeps no file of {top}"
                )
    return (
        importscan.runtime_conflicts(tree, rule, owner, selected),
        sorted(
            set(dangling) | set(importscan.imports_pointing_at_dropped(tree, selected))
        ),
    )


def parse_findings(findings: list[str]) -> list[tuple[str, str, str]]:
    """(file, module, denied dist) from the gate's finding lines."""
    rows = []
    for line in findings:
        rel, rest = line.split(": imports ", 1)
        module, tail = rest.split(" at module level, but ", 1)
        rows.append((rel, module, tail.split(" is denied")[0]))
    return rows


def is_top_level(name: str, tree: Path, owner: dict[str, str]) -> bool:
    """Whether a failing name is a package folder in the tree.

    A shipped package can also fail on a declared requirement checked through
    `importlib.metadata`; that is not an import, so the loop must not treat the
    message as a folder name.
    """
    return name in owner or (tree / name).is_dir()


def view_is_clean(view: Path) -> bool:
    """A view must start empty.

    Earlier rounds put folders back into the view, so a reused view answers for
    files the selection does not carry, and the entry import can look complete
    when the shipped image would fail.
    """
    return not view.exists() or not any(view.rglob("*"))


def materialise(tree: Path, selected: set[str], dest: Path) -> Path:
    """Write the selection to a view directory, so the import sees the shipped files."""
    for rel in sorted(selected):
        src = tree / rel
        if not src.is_file() or src.is_symlink():
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    return dest


def add_top(tree: Path, dest: Path, top: str) -> None:
    """Put one top-level folder back into the view for the next round."""
    folder = tree / top
    if not folder.is_dir():
        return
    for path in folder.rglob("*"):
        if path.is_file() and not path.is_symlink():
            target = dest / path.relative_to(tree)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def importers_by_top(tree: Path, selected: set[str], owner: dict[str, str]) -> dict:
    """Top-level -> the distributions that import it at module level, from the selection.

    Only Python files in the selection are read, because that is what the image
    carries. A tree with no importer here is not automatically prunable: a .so can
    be pulled in by DT_NEEDED or ctypes rather than by an import statement.
    """
    importers: dict[str, set[str]] = defaultdict(set)
    for rel in sorted(selected):
        path = tree / rel
        if path.suffix != ".py":
            continue
        source = owner.get(rel.split("/")[0], "?")
        for module in importscan.module_level_modules(path):
            importers[module.split(".")[0]].add(source)
    return importers


def run_import(view: Path, entry: str) -> dict:
    """Import the entry module in a fresh interpreter against one view.

    The entry import writes its own log lines to stdout, so the runner's answer
    is read from the last line rather than the whole stream.
    """
    proc = subprocess.run(
        [sys.executable, "-c", RUNNER, str(view), entry],
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"runner failed: {proc.stderr.strip() or 'no output'}")
    return json.loads(lines[-1])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--venv", required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--entry", default=ENTRY_DEFAULT)
    parser.add_argument("--arch", default="auto")
    parser.add_argument("--view", default=str(REPO / "out" / "eval-view"))
    parser.add_argument(
        "--gate",
        action="store_true",
        help="answer the deny gate for the selection instead of running the import loop",
    )
    parser.add_argument(
        "--extra-prune",
        default="",
        help="comma-separated prune patterns added to the rule for this run only",
    )
    parser.add_argument(
        "--extra-deny",
        default="",
        help="comma-separated distribution names added to the deny list for this run only",
    )
    args = parser.parse_args(argv)

    wheelhouse = Path(args.wheelhouse).resolve()
    venv = Path(args.venv).resolve()
    if not wheelhouse.is_dir():
        raise SystemExit(f"wheelhouse not found: {wheelhouse}")
    tree = venv / "lib" / "python3.12" / "site-packages"
    if not tree.is_dir():
        raise SystemExit(f"site-packages not found: {tree}")

    rule = rules_mod.load(args.rule)
    extra_prune = [p.strip() for p in args.extra_prune.split(",") if p.strip()]
    extra_deny = [p.strip() for p in args.extra_deny.split(",") if p.strip()]
    if extra_prune:
        rule["prune"] = list(rule.get("prune") or []) + extra_prune
    if extra_deny:
        rule["deny"] = list(rule.get("deny") or []) + extra_deny
    denied = {rules_mod.norm(name) for name in rule["deny"]}
    archs = arch_mod.resolve(args.arch)
    owner = importscan.top_level_map(tree)
    selected = set(pkg_mod.select(tree, rule, archs, owner))

    if args.gate:
        print(
            f"gate mode: {len(selected)} files, {sum_sizes(tree, selected) / 1e6:.1f} MB"
        )
        if extra_prune:
            print(f"extra prune: {', '.join(extra_prune)}")
        if extra_deny:
            print(f"extra deny: {', '.join(extra_deny)}")
        conflicts, dangling = gate_report(tree, rule, owner, selected)
        print(f"deny gate: {len(conflicts)} finding(s)")
        for line in conflicts:
            print(f"  {line}")
        print(f"imports pointing at a pruned folder: {len(dangling)} finding(s)")
        for line in dangling:
            print(f"  {line}")
        return 1 if conflicts or dangling else 0

    view = Path(args.view).resolve()
    if not view_is_clean(view):
        raise SystemExit(
            f"view is not empty: {view} — pass a fresh --view, a reused view "
            "answers for files the selection does not carry"
        )
    view.mkdir(parents=True, exist_ok=True)
    materialise(tree, selected, view)

    print(f"selection: {len(selected)} files, {sum_sizes(tree, selected) / 1e6:.1f} MB")
    print(f"entry: {args.entry}")
    print()

    hit_denied: dict[str, str] = {}
    hit_pruned: dict[str, str] = {}
    result = {}
    for round_no in range(1, MAX_ROUNDS + 1):
        result = run_import(view, args.entry)
        name = result["missing"]
        if name is None:
            print(f"round {round_no}: entry import completed")
            break
        top = name.split(".")[0]
        dist = owner.get(top, "?")
        trigger = result["failed"].get(top, "?")
        if not is_top_level(top, tree, owner):
            print(
                f"round {round_no}: stopped on a declared requirement, not an "
                f"import: {name}"
            )
            break
        if dist in denied:
            hit_denied[dist] = trigger
            print(
                f"round {round_no}: stopped on {name} ({dist}, denied), via {trigger}"
            )
        else:
            hit_pruned[name] = trigger
            print(
                f"round {round_no}: stopped on {name} ({dist}, not denied), via {trigger}"
            )
        add_top(tree, view, top)
    else:
        raise SystemExit(f"entry import did not complete in {MAX_ROUNDS} rounds")

    if result["error"]:
        print(f"entry import raised after the missing names: {result['error']}")

    executed = {rules_mod.norm(top): top for top in result["ok"]}
    attempted = {rules_mod.norm(top): top for top in result["failed"]}
    findings = parse_findings(importscan.runtime_conflicts(tree, rule, owner, selected))
    by_dist: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rel, module, dist in findings:
        by_dist[dist].append((rel, module))

    print()
    print("denied names, measured:")
    for dist in sorted(denied):
        comp, wheels = wheel_sizes(wheelhouse, dist)
        tops = [top for top, own in owner.items() if own == dist]
        disk, files = tree_sizes(tree, tops)
        rows = by_dist.get(dist, [])
        if dist in hit_denied:
            verdict = (
                f"executed at import time, cannot be pruned (via {hit_denied[dist]})"
            )
        elif dist in executed:
            verdict = (
                f"executed at import time (via {result['ok'].get(executed[dist], '?')})"
            )
        elif dist in attempted:
            verdict = (
                "attempted at import time, caught by the importing file "
                f"(via {result['failed'].get(attempted[dist], '?')})"
            )
        else:
            verdict = "not executed by the entry import"
        print(f"  {dist}: {verdict}")
        print(
            f"    size: wheel {comp / 1e6:.1f} MB ({wheels} wheel(s)), "
            f"tree {disk / 1e6:.1f} MB ({files} files)"
        )
        owners = defaultdict(int)
        for rel, _ in rows:
            owners[rel.split("/")[0]] += 1
        print(
            f"    module-level importers: {len(rows)} files, {dict(sorted(owners.items()))}"
        )

    print()
    print("pruned paths the entry import reached:")
    for name, trigger in sorted(hit_pruned.items()):
        dist = owner.get(name.split(".")[0], "?")
        print(f"  {name} ({dist}), via {trigger}")

    print()
    print("largest trees in the selection:")
    importers = importers_by_top(tree, selected, owner)
    sizes = defaultdict(int)
    for rel in selected:
        sizes[rel.split("/")[0]] += (tree / rel).stat().st_size
    for top, size in sorted(sizes.items(), key=lambda x: -x[1])[:20]:
        dist = owner.get(top, "?")
        who = importers.get(top, set())
        if rules_mod.norm(dist) in executed:
            flag = "executed at import time"
        elif not who:
            flag = "no module-level importer in the selection"
        elif all(rules_mod.norm(w) in denied for w in who):
            flag = f"imported only by denied: {sorted(who)}"
        else:
            flag = f"imported by {sorted(who)}"
        print(f"  {top}: {size / 1e6:.1f} MB ({dist}, {flag})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
