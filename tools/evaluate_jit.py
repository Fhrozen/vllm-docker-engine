"""What JIT costs the image, and what breaks when it is removed.

JIT is not one thing in this tree. There is a compiler engine (the MLIR library
plus the backend toolchain), a per-kernel cache, and separate packages that
bring their own compiler. A folder listing cannot say which of those a serving
run needs, so this script measures three things:

1. the size of each part, split by role;
2. whether a run hits the cache or compiles, counted by wrapping the cache
   manager rather than by guessing;
3. what happens when the compile-side files are removed and the same workload
   is run again against the cache that was warmed by the full tree.

The cache key is the digest of the compiler source files, so removing them
changes the key. That is the point of the third measurement: it shows whether a
warm cache can survive the prune, or whether the prune destroys the cache it was
supposed to replace.

Each tree is measured in its own process. The cache key is computed by reading
files from disk, so a process that has already imported the full tree cannot
report the key of a pruned tree — it would report the key it already cached.

Run inside docker with a GPU attached:

  bash docker/run.sh gpu "python3 tools/evaluate_jit.py \
    --tree out/run2/vllm.venv/lib/python3.12/site-packages --inventory"

  bash docker/run.sh gpu "python3 tools/evaluate_jit.py \
    --tree out/run2/vllm.venv/lib/python3.12/site-packages \
    --cache-dir /workspace/out/jit-cache --workload triton"

  bash docker/run.sh gpu "python3 tools/evaluate_jit.py \
    --tree out/run2/vllm.venv/lib/python3.12/site-packages \
    --view /workspace/out/jit-view --build-view"

  bash docker/run.sh gpu "python3 tools/evaluate_jit.py \
    --tree /workspace/out/jit-view \
    --cache-dir /workspace/out/jit-cache --workload triton"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# Parts of the tree that exist to turn source into machine code. Whether a run
# needs them is what the --prune measurement answers; this list is the
# hypothesis being tested, not a conclusion.
COMPILE_SIDE = [
    "triton/_C",
    "triton/compiler",
    "triton/backends",
    "nvidia/cuda_nvrtc",
    "nvidia/nvjitlink",
]

# The files triton_key() actually reads. Removing one of these changes the cache
# key, so a cache warmed on the full tree no longer applies. This is the
# difference between "the file is a compiler" and "the file is part of the
# identity of the cache", and it is why the drop list has to be chosen from here.
KEY_FILES = ["triton/runtime/cache.py", "triton/_C/libtriton.so"]
KEY_DIRS = ["triton/compiler", "triton/backends"]

# Parts of the tree that a cached kernel still needs: the python runtime that
# loads a cubin and launches it through the driver.
EXECUTION_SIDE = [
    "triton/runtime",
    "triton/language",
    "triton/tools",
]

COMPILERS = ("cc", "gcc", "g++", "clang", "nvcc", "ptxas")


def tree(path: Path) -> tuple[int, int]:
    """Files and bytes under a path, following no symlinks."""
    count = total = 0
    for entry in path.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            count += 1
            total += entry.stat().st_size
    return count, total


def on_path() -> list[str]:
    return [name for name in COMPILERS if shutil.which(name)]


def triton_key_for(site_packages: Path) -> str:
    """The cache key triton computes from the files on disk for this tree."""
    sys.path.insert(0, str(site_packages))
    from triton.runtime.cache import triton_key

    return triton_key()


def count_cache_events(site_packages: Path) -> dict[str, int]:
    """Wrap the file cache manager so hits and misses are counted, not assumed."""
    sys.path.insert(0, str(site_packages))
    from triton.runtime import cache as cache_mod

    stats = {"hits": 0, "misses": 0, "puts": 0}
    original = cache_mod.FileCacheManager.get_file

    def get_file(self, filename):
        found = original(self, filename)
        if found is None:
            stats["misses"] += 1
        else:
            stats["hits"] += 1
        return found

    cache_mod.FileCacheManager.get_file = get_file

    original_put = cache_mod.FileCacheManager.put

    def put(self, data, filename, binary=True):
        stats["puts"] += 1
        return original_put(self, data, filename, binary)

    cache_mod.FileCacheManager.put = put
    return stats


def run_triton_workload(site_packages: Path, cache_dir: str) -> str:
    """A kernel small enough that the result is about the cache, not the model."""
    # The knob reads the environment, so the cache dir has to be set before the
    # import; setting it after would leave the default ~/.triton/cache in place.
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    sys.path.insert(0, str(site_packages))
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add_one(x_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(x_ptr + offs, tl.load(x_ptr + offs) + 1, mask=offs < n)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.arange(64, device=device, dtype=torch.float32)
    started = time.monotonic()
    add_one[(1,)](x, x.numel(), BLOCK=64)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    first = float(x[0].item())
    return (
        f"device={device} triton={triton.__version__} kernel={first:.1f} "
        f"launch={elapsed:.3f}s"
    )


def mapped_jit_files(site_packages: Path) -> list[tuple[str, int]]:
    """JIT files this process has mapped, i.e. the ones a serving run really needs.

    A folder listing says what is on disk; /proc/self/maps says what the process
    loaded. A compile-side library that is never mapped can be pruned, and one
    that is mapped cannot.
    """
    mapped = set()
    with open("/proc/self/maps", encoding="ascii") as handle:
        for line in handle:
            path = line.strip().split(" ")[-1]
            if not path.startswith("/") or ".so" not in path:
                continue
            if any(f"/{part}/" in path for part in COMPILE_SIDE + EXECUTION_SIDE):
                mapped.add(path)
    rows = []
    for path in sorted(mapped):
        try:
            size = Path(path).stat().st_size
        except OSError:
            size = 0
        rows.append((path, size))
    return rows


def run_vllm_workload(
    site_packages: Path, cache_dir: str, model: str, eager: bool = True
) -> str:
    """A real generation, so the answer is about the engine, not a toy kernel.

    eager=False leaves torch.compile on, which is the path a serving image uses,
    so the cache that has to ship with the image is the one this branch produces.
    """
    # vLLM inspects the model architecture in a subprocess, which does not see a
    # path inserted by this process. Without PYTHONPATH the subprocess cannot
    # import the tree, so the packaging layout has to survive that too.
    # The engine core normally runs in its own process, so a wrapper installed
    # here would miss every cache event the serving run produces. Disabling
    # multiprocessing keeps the run in this process and the count honest.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["PYTHONPATH"] = str(site_packages)
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    os.environ.setdefault("HF_HOME", "/workspace/out/hf")
    sys.path.insert(0, str(site_packages))
    import torch
    from vllm import LLM, SamplingParams

    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.monotonic()
    llm = LLM(model=model, enforce_eager=eager, max_model_len=512)
    out = llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0.0, max_tokens=8, seed=0),
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    text = out[0].outputs[0].text
    path = "eager=on" if eager else "eager=off"
    return f"device={device} {path} load+generate={elapsed:.1f}s text={text!r}"


def key_inputs(site_packages: Path) -> tuple[int, float]:
    """Bytes that triton_key() reads, i.e. the files the cache identity depends on.

    walk_packages() only opens module files, so the .py subset of compiler/ and
    backends/ is the key; the lib/ and bin/ folders inside backends/ are not.
    """
    n = 0
    size = 0
    for entry in site_packages.rglob("*"):
        if not entry.is_file() or entry.is_symlink():
            continue
        rel = entry.relative_to(site_packages).as_posix()
        if rel in KEY_FILES:
            n += 1
            size += entry.stat().st_size
        elif entry.suffix == ".py" and rel.startswith(tuple(KEY_DIRS)):
            if entry.suffix == ".py":
                n += 1
                size += entry.stat().st_size
    return n, size


def build_view(site_packages: Path, view: Path, drop: list[str]) -> list[str]:
    """Hardlink-copy the tree, then remove the listed parts.

    Hardlinks keep the byte count honest: the view is the same bytes, so the
    measured size is the size that would ship.
    """
    view.mkdir(parents=True, exist_ok=True)
    removed = []
    for entry in site_packages.rglob("*"):
        if entry.is_symlink():
            continue
        if not entry.is_file():
            continue
        rel = entry.relative_to(site_packages).as_posix()
        if any(rel == part or rel.startswith(part + "/") for part in drop):
            continue
        target = view / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        os.link(entry, target)
    for part in drop:
        removed.append(part)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, help="tree to import from")
    parser.add_argument("--cache-dir", default="/workspace/out/jit-cache")
    parser.add_argument("--view", default="/workspace/out/jit-view")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--build-view", action="store_true")
    parser.add_argument("--drop", default="", help="folders to remove, comma-separated")
    parser.add_argument(
        "--workload",
        default="",
        help="triton, vllm, or empty to only report the inventory",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--compiled",
        action="store_true",
        help="keep torch.compile on, i.e. the path a serving image uses",
    )
    args = parser.parse_args()

    sp = Path(args.tree)
    print(f"tree: {sp}")
    print(f"compilers on PATH: {on_path() or 'none'}")

    if args.inventory:
        print()
        print("compile-side (turns source into machine code):")
        for part in COMPILE_SIDE:
            path = sp / part
            if not path.exists():
                print(f"  {part}: missing")
                continue
            n, size = tree(path)
            print(f"  {part}: {n} files, {size / 1e6:.1f} MB")
        print("execution-side (loads and launches a cached kernel):")
        for part in EXECUTION_SIDE:
            path = sp / part
            if not path.exists():
                print(f"  {part}: missing")
                continue
            n, size = tree(path)
            print(f"  {part}: {n} files, {size / 1e6:.1f} MB")
        print()
        print("other packages that carry their own compiler:")
        for name in ("tokenspeed_triton", "tilelang", "flashinfer"):
            path = sp / name
            if path.exists():
                n, size = tree(path)
                print(f"  {name}: {n} files, {size / 1e6:.1f} MB")

    n_key, size_key = key_inputs(sp)
    print(f"key inputs: {n_key} files, {size_key / 1e6:.1f} MB")

    if args.build_view:
        view = Path(args.view)
        drop = [part for part in args.drop.split(",") if part]
        dropped = build_view(sp, view, drop)
        n, size = tree(view)
        print()
        print(
            f"view after removing {', '.join(dropped)}: {n} files, {size / 1e6:.1f} MB"
        )
        return

    if not args.workload:
        return

    print()
    print(f"cache key for this tree: {triton_key_for(sp)[:16]}")
    stats = count_cache_events(sp)
    run = run_triton_workload if args.workload == "triton" else run_vllm_workload
    try:
        if args.workload == "triton":
            print(f"run ({sp.name}): {run(sp, args.cache_dir)}")
        else:
            print(
                f"run ({sp.name}): {run(sp, args.cache_dir, args.model, not args.compiled)}"
            )
    except Exception as exc:  # noqa: BLE001 - the failure is the measurement
        print(f"run ({sp.name}): FAILED {type(exc).__name__}: {exc}")
        traceback.print_exc()
    # Printed even on failure: whether a run hit or compiled is the answer, and a
    # failure that happened after a hit means the cache was not the problem.
    print(
        f"  cache: hits={stats['hits']} misses={stats['misses']} puts={stats['puts']}"
    )
    print("  files this run mapped:")
    for path, size in mapped_jit_files(sp):
        print(f"    {size / 1e6:8.1f} MB  {path}")


if __name__ == "__main__":
    main()
