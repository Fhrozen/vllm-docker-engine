"""Which libraries a process actually maps at runtime.

A DT_NEEDED edge is not the whole story: a library can be reached through
dlopen from Python, and a library can be mapped only when the feature that
uses it runs. This script runs a workload and lists the libraries the process
has mapped, so a prune decision is based on what the process needs, not on
what a linker table claims.

Run inside docker with a GPU attached:

  bash docker/run.sh gpu "python3 tools/measure_runtime_libs.py \
    --site-packages out/run2/vllm.venv/lib/python3.12/site-packages"
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

CHUNK = 1024 * 1024


def mapped_libs(source: str = "/proc/self/maps") -> list[str]:
    """Libraries the process has mapped.

    A maps line is `start end perms offset inode device path`, so the file is
    the last field. Taking the remainder of the line instead of the last field
    returns the offset and inode and matches nothing.
    """
    paths = set()
    with open(source, encoding="ascii") as handle:
        for line in handle:
            path = line.strip().split(" ")[-1]
            if path.startswith("/") and ".so" in path:
                paths.add(path)
    return sorted(paths)


def run_workload(sp: Path, steps: list[str]) -> list[str]:
    notes = []
    sys.path.insert(0, str(sp))
    if not steps:
        return notes
    import torch

    if not torch.cuda.is_available():
        notes.append("no GPU visible, the workload ran on CPU only")
        device = "cpu"
    else:
        device = "cuda"
    for step in steps:
        if step == "blas":
            x = torch.randn(64, 64, device=device)
            notes.append(f"blas: {(x @ x).sum().item():.3f}")
        elif step == "conv":
            image = torch.randn(1, 3, 32, 32, device=device)
            weight = torch.randn(4, 3, 3, 3, device=device)
            out = torch.relu(torch.conv2d(image, weight))
            notes.append(f"conv: {out.sum().item():.3f}")
        elif step == "triton":
            try:
                import triton
                import triton.language as tl

                @triton.jit
                def add_one(ptr, n: tl.constexpr, block: tl.constexpr):
                    offset = tl.program_id(0) * block + tl.arange(0, block)
                    tl.store(ptr + offset, tl.load(ptr + offset) + 1.0)

                buf = torch.ones(16, device=device)
                add_one[(1,)](buf, n=16, block=16)
                notes.append(f"triton: {buf.sum().item():.1f}")
            except (
                Exception
            ) as exc:  # a runtime build failure is a measurement, not a crash
                notes.append(f"triton failed: {exc}")
        elif step == "triton-new":
            # A kernel the cache has never seen. Without a C compiler this is the
            # test of whether the runtime image can compile anything at all.
            try:
                import triton
                import triton.language as tl

                @triton.jit
                def double(ptr, block: tl.constexpr):
                    offset = tl.program_id(0) * block + tl.arange(0, block)
                    tl.store(ptr + offset, tl.load(ptr + offset) * 2.0)

                buf = torch.arange(16, device=device, dtype=torch.float32)
                double[(1,)](buf, block=16)
                notes.append(f"triton-new: {buf.sum().item():.1f}")
            except Exception as exc:
                notes.append(f"triton-new failed: {exc}")
        elif step == "sparse":
            m = torch.eye(8, device=device)
            notes.append(
                f"sparse: {torch.sparse.mm(m.to_sparse(), m).sum().item():.3f}"
            )
        elif step == "distributed":
            import torch.distributed as dist

            notes.append(
                f"distributed available: {dist.is_available()}, "
                f"nccl backend: {dist.is_backend_available('nccl')}, "
                f"gloo backend: {dist.is_backend_available('gloo')}"
            )
        else:
            notes.append(f"unknown step: {step}")
    if device == "cuda":
        torch.cuda.synchronize()
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the libraries a workload maps."
    )
    parser.add_argument("--site-packages", required=True)
    parser.add_argument(
        "--workload",
        default="blas,conv,triton",
        help="comma list of steps: blas, conv, triton, triton-new, sparse, distributed",
    )
    parser.add_argument(
        "--maps",
        default="/proc/self/maps",
        help="mapping source, defaults to the running process",
    )
    args = parser.parse_args(argv)
    sp = Path(args.site_packages).resolve()
    if not sp.is_dir():
        print(f"no site-packages at {sp}")
        return 1

    steps = [s.strip() for s in args.workload.split(",") if s.strip()]
    for note in run_workload(sp, steps):
        print(note)

    tree = sorted(p for p in sp.rglob("*.so*") if p.is_file())
    tree_bytes = {str(p): p.stat().st_size for p in tree}
    mapped = mapped_libs(args.maps)
    hit = [p for p in mapped if p in tree_bytes]
    miss = [p for p in sorted(tree_bytes) if p not in set(hit)]

    print(f"workload: {', '.join(steps)}")
    print(
        f"mapped from the tree: {len(hit)} files, {sum(tree_bytes[p] for p in hit) / 1e6:.1f} MB"
    )
    for path in hit:
        print(f"  {path.replace(str(sp) + '/', '')} {tree_bytes[path] / 1e6:.1f}")

    per = collections.Counter()
    for path in miss:
        per[path.replace(str(sp) + "/", "").split("/")[0]] += tree_bytes[path]
    rest_bytes = sum(tree_bytes.values()) - sum(tree_bytes[p] for p in hit)
    print(
        f"never mapped by this workload: {len(miss)} files, {rest_bytes / 1e6:.1f} MB"
    )
    for name, size in per.most_common():
        print(f"  {name} {size / 1e6:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
