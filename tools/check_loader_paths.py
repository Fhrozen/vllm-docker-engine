"""Can the dynamic linker find every library a shipped ELF asks for?

`DT_NEEDED` says what a binary wants. Whether the loader can find it is a
separate question, and it is the one that decides whether a CUDA base image is
required.

A `nvidia/cuda:*-devel` or `-runtime` image ships an ld.so configuration and a
populated cache for its own library folders. A runtime image built from a plain
base plus pip wheels has neither: the `nvidia-*` libraries arrive as files under
site-packages, and nothing registers them with the loader. Such a tree is only
self-contained when each binary carries a search path that covers the folder the
dependency lives in (`RUNPATH`/`RPATH` with `$ORIGIN`), or when the dependency
sits on the default search path.

This tool reports which of the two it is, so the choice of base image is a
measurement rather than a habit.

Run inside the container:

  bash docker/run.sh script tools/check_loader_paths.py \
    --tree out/run2/vllm.venv/lib/python3.12/site-packages
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packaging"))

import libdeps  # noqa: E402  (needs the sys.path insert above)

ORIGIN_TOKENS = ("$ORIGIN", "${ORIGIN}")
DEFAULT_DIRS = ("/lib", "/usr/lib", "/lib64", "/usr/lib64")


def expand_rpath(rpath: str, origin: Path) -> list[str]:
    """Search path of a binary, with $ORIGIN resolved to its own folder.

    RUNPATH entries are colon-separated; some toolchains write them space
    separated. Both forms are read here, because a missed separator turns a
    covered dependency into a reported gap.
    """
    dirs = []
    for token in rpath.replace(" ", ":").split(":"):
        if not token:
            continue
        for marker in ORIGIN_TOKENS:
            token = token.replace(marker, str(origin))
        dirs.append(token)
    return dirs


def classify(
    needed: str,
    sonames: dict[str, list[str]],
    origin: str,
    rpath_dirs: list[str],
) -> tuple[str, str]:
    """How the loader reaches one DT_NEEDED name.

    Returns (how, detail). `how` is one of: system, driver, runpath, default,
    cache-or-env, missing. Only the last two mean the tree is not
    self-contained on its own.
    """
    if libdeps.is_system(needed):
        return "system", needed
    if libdeps.is_driver(needed):
        return "driver", needed
    targets = sonames.get(needed) or []
    if not targets:
        # A name with no DT_SONAME recorded is matched by file name, which is
        # what a dlopen-by-path call would use.
        targets = [p for p in sonames.get("__byname__", []) if Path(p).name == needed]
    if not targets:
        return "missing", needed
    for target in targets:
        folder = str(Path(target).parent)
        if folder in DEFAULT_DIRS:
            return "default", target
        if folder in rpath_dirs:
            return "runpath", target
    return "cache-or-env", targets[0]


def build_soname_map(tree: Path) -> dict[str, list[str]]:
    """DT_SONAME of every shipped ELF, plus a name-only bucket for files that
    have no dynamic section (a dlopen by path finds those by file name)."""
    sonames: dict[str, list[str]] = {}
    byname: list[str] = []
    for path in sorted(tree.rglob("*")):
        if not path.is_file() or path.is_symlink() or not libdeps.is_elf(path):
            continue
        rel = path.relative_to(tree).as_posix()
        byname.append(rel)
        try:
            soname = libdeps.elf_soname(path)
        except IndexError:
            continue
        if soname:
            sonames.setdefault(soname, []).append(rel)
    sonames["__byname__"] = byname
    return sonames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report how the loader reaches each DT_NEEDED name."
    )
    parser.add_argument("--tree", required=True)
    parser.add_argument(
        "--show",
        default="cache-or-env,missing",
        help="comma list of categories to print in full",
    )
    args = parser.parse_args(argv)

    tree = Path(args.tree).resolve()
    if not tree.is_dir():
        print(f"no tree at {tree}")
        return 1

    show = {s.strip() for s in args.show.split(",") if s.strip()}
    sonames = build_soname_map(tree)

    counts: dict[str, int] = {}
    findings: list[tuple[str, str, str]] = []
    binaries = 0
    for path in sorted(tree.rglob("*")):
        if not path.is_file() or path.is_symlink() or not libdeps.is_elf(path):
            continue
        binaries += 1
        rel = path.relative_to(tree).as_posix()
        origin = str(Path(rel).parent)
        rpath_dirs = expand_rpath(libdeps.elf_rpath(path), tree / origin)
        for needed in libdeps.elf_needed(path):
            how, detail = classify(needed, sonames, origin, rpath_dirs)
            counts[how] = counts.get(how, 0) + 1
            if how in show:
                findings.append((rel, needed, detail))

    print(f"tree: {tree}")
    print(f"ELF files read: {binaries}")
    for how in sorted(counts):
        print(f"  {how}: {counts[how]}")

    if findings:
        print(f"edges that need an ld.so cache or LD_LIBRARY_PATH ({len(findings)}):")
        for rel, needed, detail in findings:
            print(f"  {rel}: {needed} -> {detail}")
        print(
            "verdict: the tree is not self-contained; a plain base image needs "
            "an ld.so cache entry, an LD_LIBRARY_PATH, or a RUNPATH rewrite"
        )
        return 1

    print("verdict: every DT_NEEDED edge is reachable from inside the tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
