#!/usr/bin/env python3
"""Warm the JIT cache on the tree the image ships, then relocate it to the runtime path.

Two facts make this a build-stage step rather than a runtime one:

1. The runtime image has no C compiler (R01), so the cache has to be produced
   somewhere else. Measured: a cold cache on a toolchain-free tree fails at
   `triton/runtime/build.py` with "Failed to find C compiler".
2. Group files store absolute paths recorded at warm time. A cache that is simply
   copied to a new path still works through the fallback in `_get_src()`, but the
   fallback costs time: 92.3 s vs 43.6 s on the same model, seed and tree, with the
   recorded path absent. Relocating the group files keeps the fast path.

The cache key contains no GPU architecture, so the cache has to be warmed on the
arch the image will run on. Warming it for sm_80 and reading it on sm_86 is a miss.

Two more measured constraints:

- Warming has to use the compile path. An eager run does not go through the
  Triton cache at all, so an eager warm fills nothing.
- Warming runs on the tree that SHIPS, with the C headers put back for the run.
  The delivered tree prunes `triton/backends/nvidia/include`, and the C build that
  produces a cache entry needs `cuda.h`: gcc alone fails with "fatal error:
  cuda.h: No such file or directory". The headers are `.h`, not `.py`, so they are
  not key inputs - a view of the shipped tree plus the headers has the same key as
  the shipped tree alone (measured: `out/warm-view`, keys agree True).
- `strip` changes the key, so warming on the full install is wrong. The key hashes
  `triton/_C/libtriton.so`: 415,617,000 B before strip (`28daaa70a904e4f8`) and
  186,568,112 B after (`31952e683c13e95a`). A cache warmed on an unstripped tree is
  a cache the shipped image will never read. This is gated here by comparing the
  warm view against the shipped tree.

Run inside docker with the GPU attached and a compiler for the run:

    bash docker/run.sh warm
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/workspace")
sys.path.insert(0, str(REPO / "packaging"))

import cache as cache_mod  # noqa: E402

# The tree the image runs, i.e. the tree the cache is warmed on. The full install
# is only the source of the C headers the warm run needs.
DEFAULT_SHIPPED = REPO / "out/jit-verify-build/0.17.0-linux-x86_64-cuda-sm_86"
DEFAULT_HEADERS = REPO / "out/run2/vllm.venv/lib/python3.12/site-packages"
DEFAULT_CACHE = REPO / "out/jit-cache-ship"
RUNTIME_PATH = "/opt/vde/jit-cache"
HEADERS_REL = "triton/backends/nvidia/include"


def warm_view(shipped: Path, headers_from: Path, out: Path) -> Path:
    """Hardlink the shipped tree and add the C headers back for the warm run.

    Hardlinks keep the bytes identical, so the key the run computes is the key the
    shipped tree computes. Copying instead would be fine for the key and slower.
    """
    for src in sorted(shipped.rglob("*")):
        if not src.is_file():
            continue
        dest = out / src.relative_to(shipped)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Idempotent: a view left by a previous run is the same bytes, so a rerun
        # reuses it instead of failing on the first existing link.
        if dest.exists():
            continue
        os.link(src, dest)
    header_src = headers_from / HEADERS_REL
    if header_src.is_dir():
        dest_dir = out / HEADERS_REL
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(header_src.rglob("*")):
            if not src.is_file():
                continue
            dest = dest_dir / src.relative_to(header_src)
            # Same guard as the links above: a reused view already has the headers.
            if dest.exists():
                continue
            shutil.copy2(src, dest)
    return out


def warm(tree: Path, cache_dir: Path, workload: str) -> None:
    """Run the serving workload once so the cache is filled by the shipped tree.

    --compiled is not optional: an eager run never consults the Triton cache, so
    it would report a warm cache while writing nothing.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/evaluate_jit.py"),
            "--tree",
            str(tree),
            "--cache-dir",
            str(cache_dir),
            "--workload",
            workload,
            "--compiled",
        ],
        capture_output=True,
        text=True,
    )
    out = result.stdout + result.stderr
    for line in out.splitlines():
        if line.startswith(("run (", "  cache:", "text=")):
            print(line)
    if result.returncode != 0:
        print(out[-1200:])
        raise SystemExit(f"workload failed on {tree}")


def size_kb(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total // 1024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tree",
        type=Path,
        default=None,
        help="a view to warm on; defaults to the shipped tree plus the C headers",
    )
    ap.add_argument(
        "--shipped-tree",
        type=Path,
        default=DEFAULT_SHIPPED,
        help="the tree the image will run, whose cache key must match the warm view",
    )
    ap.add_argument(
        "--headers-from",
        type=Path,
        default=DEFAULT_HEADERS,
        help="install the C headers are taken from for the warm run",
    )
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--workload", default="vllm")
    ap.add_argument(
        "--arch",
        default="",
        help="the arch the cache is warmed for; recorded for the image label",
    )
    ap.add_argument(
        "--target",
        default=RUNTIME_PATH,
        help="the path the runtime image will read the cache from",
    )
    ap.add_argument(
        "--relocate-only",
        action="store_true",
        help="rewrite an existing cache without running the workload again",
    )
    args = ap.parse_args()

    if not args.relocate_only:
        # A warm run is a build-stage run: it compiles. On a toolchain-free tree
        # the same workload fails with "Failed to find C compiler", so running
        # this task without one produces an empty cache that looks like a pass.
        compilers = [c for c in ("cc", "gcc") if shutil.which(c)]
        if not compilers:
            print(
                "gate: FAIL - no C compiler on PATH; warm is a build-stage task "
                "(use: bash docker/run.sh warm, which installs one for the run)"
            )
            return 1
        if args.tree is None:
            view = args.cache_dir.parent / "warm-view"
            args.tree = warm_view(args.shipped_tree, args.headers_from, view)
        warm(args.tree, args.cache_dir, args.workload)

    if not args.cache_dir.is_dir():
        print(f"no cache at {args.cache_dir}")
        return 1

    # The cache is only worth shipping if the tree the image runs produces the
    # same key as the tree the cache was warmed on. Headers are not key inputs, so
    # the warm view and the shipped tree agree; strip would make them disagree,
    # which is what this gate catches.
    if args.tree is not None and args.shipped_tree.is_dir():
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "tools" / "measure_jit_key.py"),
                str(args.tree),
                str(args.shipped_tree),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        print(result.stdout.strip())
        if "keys agree: True" not in result.stdout:
            print(
                "gate: FAIL - a cache warmed on this tree will not hit the shipped one"
            )
            return 1

    # The cache key has no arch in it, so the arch has to be recorded separately:
    # a cache warmed for sm_80 is read on sm_86 and misses on the cubin. build.sh
    # reads this value for the image label.
    match = re.search(r"sm_\d+[a-z]?", str(args.tree))
    arch = args.arch or (match.group(0) if match else "unknown")
    (args.cache_dir.parent / "jit-arch.txt").write_text(arch + "\n")
    print(f"cache: {args.cache_dir} ({size_kb(args.cache_dir)} KB), arch {arch}")
    changed = cache_mod.relocate(args.cache_dir, Path(args.target))
    print(f"relocated {changed} group files to {args.target}")
    groups, resolvable = cache_mod.verify(args.cache_dir, Path(args.target))
    print(f"groups: {groups}, resolvable at the target: {resolvable}")
    if resolvable != groups:
        print("gate: FAIL - a group whose stored paths do not resolve will not hit")
        return 1
    print("gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
