"""Thin variant of the packaging layer: install a hash-pinned lock, then hand off.

The source for the thin variant is not an existing tree -- it is a lockfile plus
a wheelhouse. Installation is isolated and fully pinned:

    pip install --isolated --no-cache-dir --no-warn-script-location
              --require-hashes --no-deps --only-binary=:all: -r <lock>

Nothing is resolved, nothing is downloaded at install time, and every wheel is
hash-checked. The wheelhouse is the only install source (`--no-index`), so a
missing wheel fails loudly instead of quietly reaching the network. After install
the tree is pruned with the same rule as the engine variant, so the two variants
are directly comparable (`tools/compare_packaging.py`).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arch as arch_mod  # noqa: E402
import libdeps  # noqa: E402
import package as pkg_mod  # noqa: E402
import rules as rules_mod  # noqa: E402

INSTALL_FLAGS = [
    "--isolated",
    "--no-cache-dir",
    "--no-warn-script-location",
    "--require-hashes",
    "--no-deps",
    "--only-binary=:all:",
    "--no-index",
]

LOCK_LINE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[^\s\\]+)")
HASH_LINE = re.compile(r"^\s*--hash=sha256:([0-9a-f]{64})")


def lock_names(lock: Path) -> list[tuple[str, str]]:
    """(normalised name, version) for every entry in a lockfile."""
    entries = []
    for line in lock.read_text().splitlines():
        m = LOCK_LINE.match(line.strip())
        if m:
            entries.append((rules_mod.norm(m.group("name")), m.group("version")))
    return entries


def site_packages(venv: Path) -> Path:
    for path in (venv / "lib").rglob("site-packages"):
        return path
    raise SystemExit(f"no site-packages under {venv}")


def install_locked(lock: Path, wheelhouse: Path, venv: Path, rule: dict) -> None:
    if not lock.exists():
        raise SystemExit(
            f"lockfile not found: {lock} (generate it with packaging/lock.py)"
        )
    if not wheelhouse.is_dir():
        raise SystemExit(f"wheelhouse not found: {wheelhouse}")
    python = venv / "bin" / "python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = venv / "bin" / "python"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            *INSTALL_FLAGS,
            "--find-links",
            str(wheelhouse),
            "-r",
            str(lock),
        ],
        check=True,
    )
    problems = pip_check(venv)
    if problems:
        raise SystemExit("pip check:\n  " + "\n  ".join(problems))
    gate = lock_diff(venv, lock, rule)
    if gate:
        raise SystemExit("lock diff:\n  " + "\n  ".join(gate))


def pip_check(venv: Path) -> list[str]:
    python = venv / "bin" / "python"
    result = subprocess.run(
        [str(python), "-m", "pip", "check"], capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        return []
    text = result.stdout + result.stderr
    return [line for line in text.splitlines() if line.strip()]


def installed_names(venv: Path) -> set[str]:
    dists = site_packages(venv).glob("*.dist-info")
    return {rules_mod.norm(path.name.split("-")[0]) for path in dists}


def lock_diff(venv: Path, lock: Path, rule: dict) -> list[str]:
    """Every lock entry must be present, and nothing may be installed outside
    the lock.

    The environment's own tooling (pip) is denied by the rule and never reaches
    the payload, so it is reported as ignored rather than as a violation.
    """
    want = {name for name, _ in lock_names(lock)}
    have = installed_names(venv)
    problems = []
    ignored = []
    for name in sorted(want - have):
        problems.append(f"missing: {name}")
    for name in sorted(have - want):
        if rules_mod.denied(name, rule):
            ignored.append(name)
            continue
        problems.append(f"installed outside the lock: {name}")
    if ignored:
        print("lock diff: ignored (denied by the rule): " + ", ".join(ignored))
    return problems


def drop_build_path_launchers(venv: Path, build_root: Path) -> int:
    """Console scripts that hardcode the build machine are useless at runtime."""
    removed = 0
    bin_dir = venv / "bin"
    if not bin_dir.is_dir():
        return 0
    for path in bin_dir.iterdir():
        if not path.is_file() or path.is_symlink():
            continue
        try:
            head = path.read_bytes()[:256]
        except OSError:
            continue
        if b"#!" not in head[:2]:
            continue
        text = head.decode("utf-8", "replace")
        if str(build_root) in text or "pip install" in text:
            path.unlink()
            removed += 1
    return removed


def build(
    lock: Path,
    wheelhouse: Path,
    out: Path,
    rule: dict,
    archs: list[str],
) -> Path:
    venv = out / (rule["name"] + ".venv")
    if venv.exists():
        raise SystemExit(f"refusing to reuse an existing environment: {venv}")
    install_locked(lock, wheelhouse, venv, rule)

    tree = site_packages(venv)
    dropped = drop_build_path_launchers(venv, wheelhouse)
    print(f"launchers dropped: {dropped}")

    pkg = pkg_mod.build(tree, rule, out, archs, archive=True)

    uncovered = libdeps.uncovered(pkg)
    if uncovered:
        print("uncovered imports after prune:")
        for path, name in uncovered:
            print(f"  {path}: {name}")
        return pkg
    print("gate: no uncovered imports")
    return pkg


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a thin vLLM package from a lockfile + wheelhouse."
    )
    ap.add_argument("--lock", required=True, help="hash-pinned lockfile")
    ap.add_argument("--wheelhouse", required=True)
    ap.add_argument("--rule", default="vllm")
    ap.add_argument("--out", default="out/thin")
    ap.add_argument("--arch", help="comma-separated archs; default is detection")
    args = ap.parse_args()

    rule = rules_mod.load(args.rule)
    archs = arch_mod.resolve(args.arch)
    build(Path(args.lock), Path(args.wheelhouse), Path(args.out), rule, archs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
