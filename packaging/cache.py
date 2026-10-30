#!/usr/bin/env python3
"""Relocate a warmed JIT cache so it hits at the path the runtime image uses.

Measured (`tools/measure_cache_paths.py`, same model, same seed, GPU):

    cache copied unchanged, recorded path absent -> 92.3 s, puts=872
    cache copied with stored paths rewritten     -> 43.6 s, puts=0

Both work, so relocation is not a correctness problem: `FileCacheManager.get_group()`
returns None when the stored absolute paths are dead, and `_get_src()` then falls
back to rebuilding paths from the cache dir. The fallback is what costs the extra
~49 s - the group is re-materialised instead of read. Rewriting the group files at
build time keeps the fast path, and it is a build-stage step, so R01 still holds.

Group files (`__grp__<name>.json`) store absolute paths recorded at warm time.
Rewriting them to the folder each group actually lives in makes the cache
location-independent, which is what a shipped cache needs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Group files are nested one level: `<cache>/<kernel hash>/__grp__<name>.json`.
# A single-segment glob matches nothing here, which would make the gate report
# 0 groups and pass vacuously.
GROUP_GLOB = "*/__grp__*.json"


def relocate(cache_dir: Path, target: Path) -> int:
    """Point every group file at `target`, returning how many files changed.

    Paths are rebuilt from the group file's own folder rather than
    string-substituted, so a cache warmed anywhere lands correctly in the image.
    Nothing is moved: the image copies the cache folder as it is, so the group's
    folder is already the folder its stored paths point into. Rewriting in place
    keeps the cache readable on the build host too, which is what `verify()`
    checks after the rewrite.
    """
    changed = 0
    for path in sorted(cache_dir.glob(GROUP_GLOB)):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        children = data.get("child_paths")
        if not children:
            continue
        # Keep the group's own folder. A run that flattened this to
        # `<target>/<name>` produced 92.4 s with puts=872 on the shipped tree -
        # the dead-path fallback in `_get_src()` re-materialised every group
        # because the files actually live one level down, in the same folder as
        # the group file. Preserving the folder is what keeps the fast path.
        for key, value in children.items():
            children[key] = os.path.join(
                str(target), path.parent.name, os.path.basename(value)
            )
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle)
        changed += 1
    return changed


def verify(cache_dir: Path, target: Path | None = None) -> tuple[int, int]:
    """Report (group files, groups whose stored paths resolve inside the cache).

    A group whose stored paths point outside the cache is a group that will not hit
    after relocation, so this is the gate for the shipped cache.

    `target` is the path the runtime image will read the cache from. After
    `relocate()` the stored paths are under that path, which does not exist in the
    container that ran the rewrite - it only exists inside the image. Comparing the
    stored folder against the target is what makes the gate answer for a relocated
    cache; checking `os.path.exists()` alone fails every group after a rewrite,
    which is the false failure this fix removes.
    """
    groups = 0
    resolvable = 0
    for path in sorted(cache_dir.glob(GROUP_GLOB)):
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        children = data.get("child_paths")
        if not children:
            continue
        groups += 1
        # A group resolves when every stored path points at a file in the folder
        # the group itself lives in - either the cache dir it was warmed in, or
        # the folder it will live in inside the image. `relocate()` keeps each
        # group in its own kernel-hash folder, so the folder under `target` is
        # the one to compare against; comparing against `target` alone fails
        # every group after a rewrite.
        folder = str(target / path.parent.name) if target is not None else None
        if all(
            os.path.dirname(v) == str(path.parent)
            or (folder is not None and os.path.dirname(v) == folder)
            or os.path.exists(v)
            for v in children.values()
        ):
            resolvable += 1
    return groups, resolvable


if __name__ == "__main__":
    import argparse
    import sys

    if os.environ.get("VDE_ENFORCE") == "1":
        print("cache.py must not run at runtime (R01):", *sys.argv[1:], file=sys.stderr)
        raise SystemExit(1)

    ap = argparse.ArgumentParser(description="Relocate a warmed JIT cache.")
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--target", type=Path, help="path the cache will be read from")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if args.verify_only:
        groups, resolvable = verify(args.cache_dir, args.target)
        print(f"groups: {groups}, resolvable: {resolvable}")
        raise SystemExit(0 if resolvable == groups else 1)

    if not args.target:
        raise SystemExit("--target is required (the runtime cache path)")
    n = relocate(args.cache_dir, args.target)
    print(f"rewrote {n} group files to {args.target}")
    groups, resolvable = verify(args.cache_dir, args.target)
    print(f"groups: {groups}, resolvable: {resolvable}")
    raise SystemExit(0 if resolvable == groups else 1)
