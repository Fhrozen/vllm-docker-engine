#!/usr/bin/env python3
"""Distributed gate: can torch be built with USE_DISTRIBUTED=0 for this project?

Two parts, both read-only against the clone:

  1. static  — which names vLLM imports from torch.distributed at module load
  2. simulate — a stub `torch.distributed` that provides only what a
     USE_DISTRIBUTED=0 build exposes, then the real import is attempted against it

`torch.distributed.is_available()` is literally `hasattr(torch._C, "_c10d_init")`.
With USE_DISTRIBUTED=0 only GroupName and ProcessGroup survive as stubs; Backend,
Store, TCPStore, init_process_group, rendezvous and the functional-collective
module are inside the `if is_available():` block. So an unconditional import at
module load fails even at tp_size=1.

The gate reports what it finds. It does not decide for you.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packaging"))

import importscan  # noqa: E402

CLONE = REPO / "vllm" / "vllm"

# What a USE_DISTRIBUTED=0 build still exposes. Everything else is inside the
# `if is_available():` guard and is therefore absent.
STUB_PROVIDES = ("GroupName", "ProcessGroup")


def module_level_imports(path: Path) -> list[str]:
    """Import statements at module scope, i.e. executed at import time."""
    return importscan.module_level_imports(path, "torch.distributed")


def stub_distributed() -> types.ModuleType:
    """A torch.distributed as a USE_DISTRIBUTED=0 build would present."""
    importlib.import_module("torch")

    dist = types.ModuleType("torch.distributed")
    for name in STUB_PROVIDES:
        setattr(dist, name, type(name, (), {"__doc__": "stub"}))
    sys.modules["torch.distributed"] = dist
    return dist


def is_distributed(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name.startswith("torch.distributed") for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").startswith("torch.distributed")
    return False


def scan(folder: Path) -> tuple[int, int]:
    """(unguarded, guarded) import counts: module level vs inside an `if` block."""
    unguarded = 0
    guarded = 0
    for path in sorted(folder.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if is_distributed(node):
                unguarded += 1
            elif isinstance(node, ast.If):
                for inner in ast.walk(node):
                    if is_distributed(inner):
                        guarded += 1
    return unguarded, guarded


def simulate_off() -> list[str]:
    """Attempt the real import against the stub and report what breaks."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return ["UNVERIFIED: torch is not installed in this environment"]

    stub_distributed()
    sys.path.insert(0, str(REPO / "vllm"))

    try:
        importlib.import_module("vllm.distributed.parallel_state")
    except Exception as exc:  # noqa: BLE001 - the gate reports the failure, it does not fix it
        return [f"{type(exc).__name__}: {exc}"]
    return ["import succeeded against the stub"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gate the distributed dependency of the vLLM clone."
    )
    ap.add_argument("--file", default="vllm/vllm/distributed/parallel_state.py")
    ap.add_argument(
        "--simulate-off", action="store_true", help="attempt the import against a stub"
    )
    ap.add_argument(
        "--scan",
        nargs="?",
        const="vllm/vllm/distributed",
        help="scan every file under a clone folder (default vllm/vllm/distributed)",
    )
    args = ap.parse_args()

    target = REPO / args.file
    if not target.exists():
        raise SystemExit(f"clone file not found: {target} (the clone is read-only)")

    print(f"file: {args.file}")
    imports = module_level_imports(target)
    if imports:
        print("module-level imports (executed at import time):")
        for line in imports:
            print(f"  {line}")
    else:
        print("no module-level torch.distributed imports")

    if args.scan:
        folder = REPO / args.scan
        if not folder.is_dir():
            raise SystemExit(f"clone folder not found: {folder}")
        unguarded, guarded = scan(folder)
        print(f"scan {args.scan}: {unguarded} unguarded, {guarded} guarded")

    if args.simulate_off:
        print("simulation against a USE_DISTRIBUTED=0 stub:")
        for line in simulate_off():
            print(f"  {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
