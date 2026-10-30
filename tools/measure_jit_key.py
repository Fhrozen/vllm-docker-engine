#!/usr/bin/env python3
"""Report the JIT cache key a tree produces, and whether two trees agree.

The cache key is not a property of the model or the GPU: it is derived from the
bytes of the Python modules under triton/compiler, triton/backends and
triton/language, plus libtriton.so and cache.py itself. Pruning any of those
modules changes the key, so a cache warmed on the full tree does not hit on the
delivered tree. This is what made the delivered image recompile 1,696 kernels
instead of hitting the shipped cache.

Usage:
    bash docker/run.sh script tools/measure_jit_key.py TREE [TREE ...]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


CHILD = """
import sys
sys.path.insert(0, sys.argv[1])
from triton.runtime.cache import triton_key
print(triton_key())
"""


def cache_key(site_packages: str) -> str:
    """The key this tree produces, measured in a fresh process.

    Comparing trees in one process is not possible: triton's extension module is
    cached by path, so a second import returns the module loaded from the first
    tree. Dropping triton from sys.modules does not help either - re-importing an
    already-initialised extension does not run its init again, and the submodules
    libtriton registers at load time (triton._C.libtriton.linear_layout, used by
    triton/tools/__init__.py) are gone with it. That produced a false failure here.
    """
    path = os.path.join(site_packages, "triton/runtime/cache.py")
    if not os.path.isfile(path):
        raise SystemExit(f"no triton/runtime/cache.py in {site_packages}")
    result = subprocess.run(
        [sys.executable, "-c", CHILD, site_packages],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"{site_packages}: {result.stderr.strip().splitlines()[-1:]}")
    return result.stdout.strip()


def key_inputs(site_packages: str) -> tuple[int, int]:
    """Files and bytes that feed the key: cache.py, the walked packages, libtriton.

    Counting these is what shows a prune changing the key: the key is a hash over
    exactly this set, so a tree that drops one of them produces a different key.
    """
    triton = os.path.join(site_packages, "triton")
    total_files = 0
    total_bytes = 0
    for folder in ("compiler", "backends", "language"):
        for root, _, files in os.walk(os.path.join(triton, folder)):
            for name in files:
                if not name.endswith(".py"):
                    continue
                total_files += 1
                total_bytes += os.path.getsize(os.path.join(root, name))
    total_files += 1
    total_bytes += os.path.getsize(os.path.join(triton, "runtime/cache.py"))
    for ext in (".so",):
        lib = os.path.join(triton, "_C", f"libtriton{ext}")
        if os.path.exists(lib):
            total_files += 1
            total_bytes += os.path.getsize(lib)
    return total_files, total_bytes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trees", nargs="+")
    args = parser.parse_args()

    keys = {}
    for tree in args.trees:
        keys[tree] = cache_key(tree)

    for tree, key in keys.items():
        files, size = key_inputs(tree)
        print(f"{tree}\n  key: {key}\n  key inputs: {files} files, {size / 1e6:.1f} MB")

    values = list(keys.values())
    agree = len(set(values)) == 1
    print(f"\nkeys agree: {agree}")
    if not agree:
        print("a cache warmed on one tree will not hit on the other")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
