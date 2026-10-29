#!/usr/bin/env python3
"""Evaluate each denied name before a rule is changed.

Three questions per name, each answered by measurement rather than by reading
the rule:

  size   - compressed bytes in the wheelhouse, extracted bytes in the tree
  reach  - is the name imported at module level from the serve path
  prune  - can the importing files be pruned instead of shipping the package

A name is required when a file on the serve path imports it at module scope:
that file cannot be pruned without breaking the entry point. A name is prunable
when importers exist but none of them is reached from the entry module.

Nothing here runs on the host: run it inside the container, e.g.

  bash docker/run.sh script tools/evaluate_deny.py --wheelhouse out/wheelhouse \
      --venv out/eval-venv --rule vllm
"""

from __future__ import annotations

import argparse
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


def wheel_sizes(wheelhouse: Path, dist: str) -> tuple[int, int]:
    """Compressed bytes and wheel count for one distribution in the wheelhouse."""
    total = 0
    count = 0
    for path in sorted(wheelhouse.glob("*.whl")):
        if rules_mod.norm(path.name.split("-")[0]) == dist:
            total += path.stat().st_size
            count += 1
    return total, count


def extracted_sizes(tree: Path, top_levels: list[str]) -> tuple[int, int]:
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


def module_file(tree: Path, module: str, selected: set[str]) -> str | None:
    """Resolve a dotted module to a shipped file, honouring the selection."""
    parts = module.split(".")
    for rel in ("/".join(parts) + ".py", "/".join(parts) + "/__init__.py"):
        if rel in selected:
            return rel
    return None


def serve_closure(tree: Path, selected: set[str], entry: str) -> dict[str, str]:
    """Module-level import closure from the entry module, as child -> parent."""
    start = module_file(tree, entry, selected)
    if start is None:
        raise SystemExit(f"entry module is not in the selection: {entry}")
    parent = {start: ""}
    queue = [start]
    while queue:
        rel = queue.pop(0)
        for module in importscan.module_level_modules(tree / rel):
            child = module_file(tree, module, selected)
            if child is None or child in parent:
                continue
            parent[child] = rel
            queue.append(child)
    return parent


def chain(parent: dict[str, str], target: str) -> str:
    path = [target]
    while parent.get(path[0]):
        path.append(parent[path[0]])
    return " -> ".join(reversed(path))


def parse_findings(findings: list[str]) -> list[tuple[str, str, str]]:
    """(file, module, denied dist) from the gate's finding lines."""
    rows = []
    for line in findings:
        rel, rest = line.split(": ", 1)
        module = rest.split(" at module level")[0].split(" imports ")[1]
        dist = rest.split(" but ")[1].split(" is denied")[0]
        rows.append((rel, module, dist))
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--venv", required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--entry", default=ENTRY_DEFAULT)
    parser.add_argument("--arch", default="auto")
    args = parser.parse_args(argv)

    wheelhouse = Path(args.wheelhouse).resolve()
    venv = Path(args.venv).resolve()
    if not wheelhouse.is_dir():
        raise SystemExit(f"wheelhouse not found: {wheelhouse}")
    tree = venv / "lib" / "python3.12" / "site-packages"
    if not tree.is_dir():
        raise SystemExit(f"site-packages not found: {tree}")

    rule = rules_mod.load(args.rule)
    archs = arch_mod.resolve(args.arch)
    selected = set(pkg_mod.select(tree, rule, archs))
    owner = importscan.top_level_map(tree)
    parent = serve_closure(tree, selected, args.entry)

    rows = parse_findings(importscan.runtime_conflicts(tree, rule, owner, selected))
    by_dist: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rel, module, dist in rows:
        by_dist[dist].append((rel, module))

    print(f"selection: {len(selected)} files, serve closure: {len(parent)} files")
    print()
    for dist in sorted(rule["deny"]):
        comp, wheels = wheel_sizes(wheelhouse, dist)
        tops = [top for top, own in owner.items() if own == dist]
        disk, files = extracted_sizes(tree, tops)
        found = by_dist.get(dist, [])
        owners = defaultdict(int)
        for rel, _ in found:
            owners[rel.split("/")[0]] += 1
        reached = [rel for rel, _ in found if rel in parent]
        if not found:
            verdict = "no module-level importer"
        elif reached:
            verdict = "required (imported on the serve path)"
        else:
            verdict = "prunable (importers exist, none reached)"
        print(f"{dist}: {verdict}")
        print(
            f"  size: wheel {comp / 1e6:.1f} MB ({wheels} wheel(s)), "
            f"tree {disk / 1e6:.1f} MB ({files} files)"
        )
        print(
            f"  importers: {len(found)} files, by owner {dict(sorted(owners.items()))}"
        )
        if reached:
            print(f"  serve path: {chain(parent, reached[0])}")
        elif found:
            print(f"  first importer (not reached): {found[0][0]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
