#!/usr/bin/env python3
"""Compare packaging variants on the same input.

Rows are the variants, columns are what changed:

  bytes, file count, manifest, gate results, usability

The comparison is only meaningful when the input is identical, so every variant
runs against the same source tree or the same lock + wheelhouse. A variant that
cannot be built is reported as a failure, not skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packaging"))

import arch as arch_mod  # noqa: E402
import libdeps  # noqa: E402
import package as pkg_mod  # noqa: E402
import package_thin as thin_mod  # noqa: E402
import rules as rules_mod  # noqa: E402


def row(name: str, pkg: Path, rule: dict) -> dict:
    manifest_path = pkg / pkg_mod.MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    entries = (manifest or {}).get("files") or []
    return {
        "variant": name,
        "package": str(pkg),
        "bytes": sum(e["bytes"] for e in entries),
        "files": len(entries),
        "manifest": bool(manifest),
        "build_id": (manifest or {}).get("build_id", ""),
        "gate_verify": pkg_mod.verify(pkg, full=True),
        "gate_uncovered": libdeps.uncovered(pkg),
        "gate_static": [e["path"] for e in entries if libdeps.is_static(e["path"])],
        "bytecode": rule.get("bytecode", "skip"),
        "strip": bool(rule.get("strip")),
    }


def import_smoke(pkg: Path, module: str) -> str:
    env = dict(os.environ, PYTHONPATH=str(pkg))
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}; print('ok')"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode == 0:
        return "ok"
    text = (result.stdout + result.stderr).strip()
    return text.splitlines()[-1] if text else "no output"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare packaging variants on identical input."
    )
    ap.add_argument("--rule", default="vllm")
    ap.add_argument("--src", help="installed tree to select from (engine variant)")
    ap.add_argument("--lock", help="hash-pinned lockfile (thin variant)")
    ap.add_argument("--wheelhouse", help="wheelhouse for the thin variant")
    ap.add_argument("--out", default="out/compare")
    ap.add_argument("--arch", help="comma-separated archs; default is detection")
    ap.add_argument(
        "--modes", default="source,pyc-only", help="bytecode modes to compare"
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    base = rules_mod.load(args.rule)
    archs = arch_mod.normalise(args.arch) if args.arch else arch_mod.detect()[0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    failures: list[str] = []

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        rule = dict(base)
        rule["bytecode"] = mode

        if args.src:
            try:
                pkg = pkg_mod.build(
                    Path(args.src), rule, out / mode / "engine", archs, archive=False
                )
                rows.append(row(f"engine/{mode}", pkg, rule))
            except SystemExit as exc:
                failures.append(f"engine/{mode}: {exc}")

        if args.lock and args.wheelhouse:
            try:
                pkg = thin_mod.build(
                    Path(args.lock),
                    Path(args.wheelhouse),
                    out / mode / "thin",
                    rule,
                    archs,
                )
                rows.append(row(f"thin/{mode}", pkg, rule))
            except SystemExit as exc:
                failures.append(f"thin/{mode}: {exc}")

    for r in rows:
        module = base["name"].replace("-", "_")
        r["usability"] = import_smoke(Path(r["package"]), module)

    if args.json:
        print(
            json.dumps({"rows": rows, "failures": failures}, indent=2, sort_keys=True)
        )
    else:
        header = (
            f"{'variant':<16}{'files':>7}{'bytes':>14}"
            f"{'manifest':>10}{'gates':>8}{'usability':<24}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            gates = (
                "clean"
                if not (r["gate_verify"] or r["gate_uncovered"] or r["gate_static"])
                else "FAIL"
            )
            print(
                f"{r['variant']:<16}{r['files']:>7}{r['bytes']:>14}"
                f"{str(r['manifest']):>10}{gates:>8}{r['usability']:<24}"
            )
        for failure in failures:
            print(f"build failure: {failure}")

    return (
        1
        if failures
        or any(
            r["gate_verify"] or r["gate_uncovered"] or r["gate_static"] for r in rows
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
