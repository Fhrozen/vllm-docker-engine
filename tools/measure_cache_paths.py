#!/usr/bin/env python3
"""Measure whether a shipped JIT cache survives being moved to a new path.

The group files in the cache (`__grp__<name>.json`) store absolute paths recorded
at warm time, and `FileCacheManager.get_group()` checks `os.path.exists()` on
those stored paths rather than on paths rebuilt from the cache dir. So a cache
that is warmed at one path and read at another does not hit.

Two branches are run, as the repo rule requires:
  A. copy the cache to a new path unchanged;
  B. copy it and rewrite the stored paths to the new location.

Usage:
    bash docker/run.sh script tools/measure_cache_paths.py
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

SOURCE = "/workspace/out/jit-cache-warm"
MOVED = "/workspace/out/cache-moved"
REWRITTEN = "/workspace/out/cache-rewritten"
TREE = "/workspace/out/jit-verify-build/0.17.0-linux-x86_64-cuda-sm_86"


def copy_cache(src: str, dst: str, rewrite: bool) -> int:
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    changed = 0
    if rewrite:
        for path in glob.glob(os.path.join(dst, "*", "__grp__*.json")):
            with open(path) as handle:
                data = json.load(handle)
            children = data.get("child_paths")
            if not children:
                continue
            for key, value in children.items():
                name = os.path.basename(value)
                children[key] = os.path.join(os.path.dirname(path), name)
            with open(path, "w") as handle:
                json.dump(data, handle)
            changed += 1
    return changed


def run(cache_dir: str) -> None:
    print(f"\n=== cache at {cache_dir}")
    result = subprocess.run(
        [
            sys.executable,
            "/workspace/tools/evaluate_jit.py",
            "--tree",
            TREE,
            "--cache-dir",
            cache_dir,
            "--workload",
            "vllm",
            "--compiled",
        ],
        capture_output=True,
        text=True,
    )
    out = result.stdout + result.stderr
    for line in out.splitlines():
        if line.startswith(("run (", "  cache:", "text=")):
            print(line)
    if "Failed to find C compiler" in out:
        print("  note: fell back to compiling, so the cache did not hit")


def main() -> int:
    if not os.path.isdir(SOURCE):
        raise SystemExit(f"no warmed cache at {SOURCE}")

    print(f"source cache: {SOURCE}")
    groups = glob.glob(os.path.join(SOURCE, "*", "__grp__*.json"))
    print(f"group files: {len(groups)}")
    sample = json.load(open(groups[0]))
    stored = list(sample.get("child_paths", {}).values())
    print(f"stored paths are absolute: {all(os.path.isabs(p) for p in stored)}")

    # The stored paths in a group file are absolute, so the path they record
    # must not exist while a copy is read. Measured earlier: leaving the source
    # in place makes a moved cache "hit" by reading the original folder, which
    # is a false pass - the copy was never consulted.
    pristine = SOURCE + ".pristine"
    if os.path.exists(pristine):
        shutil.rmtree(pristine)
    shutil.copytree(SOURCE, pristine)

    copy_cache(SOURCE, MOVED, rewrite=False)
    changed = copy_cache(SOURCE, REWRITTEN, rewrite=True)
    shutil.rmtree(SOURCE)

    print("\n=== cache copied unchanged, recorded path no longer exists")
    run(MOVED)

    print(f"\nrewrote {changed} group files to the new location")
    print("\n=== cache copied with paths rewritten, recorded path no longer exists")
    run(REWRITTEN)

    shutil.copytree(pristine, SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
