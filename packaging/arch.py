#!/usr/bin/env python3
"""CUDA architecture selection, decided by command at build time.

    python packaging/arch.py        # the arch this machine's card answers to
    python packaging/arch.py --list # what the fallback list contains
    VDE_CUDA_ARCH=sm_90a python packaging/arch.py

Detection order:
  1. VDE_CUDA_ARCH (repeatable with `;` or `,`) - an explicit override wins.
  2. nvidia-smi compute_cap - the card that is actually here.
  3. The full list, when no GPU is reachable. A package built on the fallback
     list is labelled `archs: full` so nobody reads it as single-arch.

"No faked data for hardware we cannot run": a per-arch tree that is empty, or
that carries code for an arch the package does not declare, is a defect and is
reported, not silently kept.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

FULL_LIST = (
    "sm_80",
    "sm_86",
    "sm_89",
    "sm_90",
    "sm_90a",
    "sm_100a",
    "sm_120a",
    "sm_121a",
)
ARCH = re.compile(
    r"(?:sm_|compute_)?(?P<num>\d+)(?P<suf>[af])?(?:-real|-virtual)?", re.I
)
# In a file name an architecture must be spelled with its prefix. Without this
# rule `libz.so.1` or `vllm-2.0.0.dist-info` reads as an arch, and shared data
# is dropped from the package.
ARCH_NAME = re.compile(
    r"(?:sm_|compute_)(?P<num>\d+)(?P<suf>[af])?(?:-real|-virtual)?", re.I
)
GFX = re.compile(r"gfx[0-9a-f]+", re.I)


def normalise(value: str) -> str:
    """Accept both spellings: `sm_90a` and the `8.6` that nvidia-smi prints as
    compute_cap. Keep the arch suffix - sm_90a and sm_90 are different code
    paths, and a tree named for one is not valid for the other."""
    raw = value.strip().lower().replace(".", "")
    match = ARCH.fullmatch(raw)
    if not match:
        raise SystemExit(f"arch.py: cannot read a CUDA architecture from '{value}'")
    return f"sm_{match.group('num')}{match.group('suf') or ''}"


def parse(value: str) -> list[str]:
    """Split a comma or semicolon list of architectures, normalising each."""
    return [normalise(a) for a in re.split(r"[;,\s]+", value) if a.strip()]


def detect() -> tuple[list[str], str]:
    """(archs, how they were decided)."""
    override = os.environ.get("VDE_CUDA_ARCH", "").strip()
    if override:
        picked = parse(override)
        return picked, "override"

    if shutil.which("nvidia-smi"):
        try:
            done = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            lines = done.stdout.splitlines()
            caps = [normalise(line) for line in lines if line.strip()]
            if done.returncode == 0 and caps:
                return sorted(set(caps)), "nvidia-smi"
        except (OSError, subprocess.SubprocessError):
            pass

    return list(FULL_LIST), "fallback"


def resolve(value: str | None) -> list[str]:
    """Architectures for a command-line value.

    `auto` is what the rules declare (`arch_filter: auto`) and what the build
    passes by default, so it has to mean detection here rather than be fed to
    `normalise()`, which would reject it as an unreadable architecture.
    """
    if not value or value.strip().lower() == "auto":
        return detect()[0]
    return parse(value)


def arch_of(name: str) -> str | None:
    """The architecture a file is named for, in the same spelling the rule
    uses. Returns None for a file that names no architecture."""
    match = GFX.search(name)
    if match:
        return match.group(0).lower()
    match = ARCH_NAME.search(name)
    if not match:
        return None
    return f"sm_{match.group('num')}{match.group('suf') or ''}"


def arch_wanted(name: str, archs: list[str]) -> bool:
    """Data files named for another architecture are left out.

    Shared files, meaning those with no arch in the name, stay.

    Source is not filtered. A package folder named for an architecture is a
    module its parent imports at module level, so dropping it is a runtime
    ImportError rather than a saving: `triton/experimental/gluon/language/amd/
    __init__.py` has `from . import gfx1250`, and the harness build that dropped
    it failed at that line. Measured on the real tree for sm_86, the source the
    filter would drop is 7 files / 14.2 KB, three of them reached by a
    module-level import - below the ship-it threshold, so the filter is data-only.
    """
    if not archs:
        return True
    if name.endswith(".py"):
        return True
    named = arch_of(name)
    if named is None:
        return True
    return named in {a.lower() for a in archs}


def tree_defect(folder, archs: list[str]) -> list[str]:
    """Report a per-arch tree that is empty, or that carries code for an arch
    the package does not declare."""
    problems: list[str] = []
    root = str(folder)
    entries = [e for e in os.listdir(root) if os.path.isfile(os.path.join(root, e))]
    if not entries:
        problems.append(f"{root}: no files for any architecture")
        return problems
    wanted = {a.lower() for a in archs}
    for name in entries:
        found = arch_of(name)
        if found and found not in wanted:
            problems.append(f"{root}/{name}: code for {found}, not declared")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Print the CUDA architectures this build should target."
    )
    ap.add_argument(
        "--list", action="store_true", help="print the fallback list instead"
    )
    args = ap.parse_args()

    if args.list:
        for arch in FULL_LIST:
            print(arch)
        return 0

    archs, how = detect()
    print(" ".join(archs))
    print(f"# decided by {how}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
