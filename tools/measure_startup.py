#!/usr/bin/env python3
"""Measure what a delivered package costs on a cold start.

Run inside the container:

    bash docker/run.sh py tools/measure_startup.py <package dir>

The report answers two questions for the package as it is delivered:

1. Are the bytecode caches it carries valid at import time? A deterministic
   archive rewrites every member's mtime, so a timestamp cache is invalid the
   moment it is extracted, and a hash cache is not.
2. How long does the entry import take, as the user the runtime image runs as
   (uid 1000, on a tree that cannot be written)?

Both are reported for the same tree, so the cost of an invalid cache is the
difference between this run and a run on a tree whose caches are valid.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import evaluate_bytecode as bc  # noqa: E402

RUNTIME_UID = 1000


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: measure_startup.py <package dir>", file=sys.stderr)
        return 2
    pkg = Path(sys.argv[1]).resolve()
    if not pkg.is_dir():
        print(f"not a directory: {pkg}", file=sys.stderr)
        return 2

    print(f"package: {pkg}")
    print(f"bytecode as delivered: {bc.scan(pkg, lambda p: p.stat().st_mtime)}")

    elapsed, detail = bc.timed_import(pkg, RUNTIME_UID)
    print(f"entry import as uid {RUNTIME_UID}: {elapsed:.1f} s, {detail}")
    print(f"modules executed: {bc.module_count(pkg, RUNTIME_UID)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
