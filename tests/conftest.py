"""Packaging suite. Runs inside the tools container:

    bash docker/run.sh test

Tests write only to tmp_path, never into the repo, and never reach the network.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for folder in (REPO / "packaging", REPO / "tools"):
    sys.path.insert(0, str(folder))


def tree(root: Path, files: dict[str, str]) -> Path:
    """A small source tree, written under tmp_path only."""
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def rule(**overrides) -> dict:
    base = {
        "schema_version": 2,
        "id": "test-package",
        "name": "foo",
        "version": "1.0",
        "platform": "linux-x86_64",
        "backend": "cuda",
        "entrypoint": "foo serve",
        "allow": ["**"],
        "deny": ["bar"],
        "prune": ["**/tests/**", "**/*.a"],
        "bytecode": "skip",
        "strip": False,
        "arch_filter": "auto",
        "system_libs": ["libc.so.6"],
        "driver_libs": ["libcuda.so.1"],
        "runtime_apt": ["tini"],
        "forbidden_tools": ["pip", "gcc"],
    }
    base.update(overrides)
    return base
