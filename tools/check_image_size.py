#!/usr/bin/env python3
"""Size gate: budget and the top files inside an image.

Mirrors the shape of `.buildkite/check-wheel-size.py` in the vLLM clone, but
measures an image rather than a wheel, and compares against a baseline image so
the Phase 4 gate ("<= 4 GB AND >= 60% smaller than baseline") is a number, not
an opinion.

Nothing here estimates. A number is printed only when it was read from the
docker daemon; anything that cannot be read is reported as UNVERIFIED.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess

BUDGET_BYTES = 4 * 1024 * 1024 * 1024
TOP_N = 10


def docker(args: list[str]) -> str:
    if shutil.which("docker") is None:
        raise SystemExit("docker is not available; the size gate cannot run")
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        raise SystemExit(f"docker {' '.join(args)} failed:\n{detail}")
    return result.stdout


def image_size(image: str) -> int:
    out = docker(["image", "inspect", "--format", "{{.Size}}", image]).strip()
    if not out:
        raise SystemExit(f"image not found locally: {image}")
    return int(out)


def layer_sizes(image: str) -> tuple[int, str]:
    """Compressed download size, from the registry manifest when it is reachable.

    `docker image inspect` reports the *unpacked* size. The download is the sum
    of the compressed layer sizes, which only the manifest exposes. When the
    registry is unreachable, this returns (0, "UNVERIFIED") rather than a guess.
    """
    try:
        out = docker(["manifest", "inspect", "-v", image])
    except SystemExit:
        return 0, "UNVERIFIED"
    try:
        manifest = json.loads(out)
    except json.JSONDecodeError:
        return 0, "UNVERIFIED"

    entries = manifest if isinstance(manifest, list) else [manifest]
    total = 0
    for entry in entries:
        layers = entry.get("layers") or []
        if not layers:
            return 0, "UNVERIFIED"
        total += sum(int(layer.get("size", 0)) for layer in layers)
    return total, "measured"


def top_files(image: str, n: int) -> list[tuple[int, str]]:
    """Largest files, read from the image layers by running `find` in the image."""
    out = docker(
        [
            "run",
            "--rm",
            "--entrypoint",
            "/bin/sh",
            image,
            "-c",
            f"find / -xdev -type f -printf '%s\\t%p\\n' | sort -rn | head -{n}",
        ]
    )
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        size, _, path = line.partition("\t")
        rows.append((int(size), path))
    return rows


def human(value: int) -> str:
    return f"{value / (1024 * 1024 * 1024):.2f} GB"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Image size gate with a baseline comparison."
    )
    ap.add_argument("--image", required=True)
    ap.add_argument("--baseline", help="image to compare against (the Phase 3 result)")
    ap.add_argument("--budget", default="4GB", help="size budget, e.g. 4GB or 3500MB")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    budget = BUDGET_BYTES
    if args.budget.endswith("GB"):
        budget = int(float(args.budget[:-2]) * 1024**3)
    elif args.budget.endswith("MB"):
        budget = int(float(args.budget[:-2]) * 1024**2)

    size = image_size(args.image)
    compressed, how = layer_sizes(args.image)

    report: dict = {
        "image": args.image,
        "unpacked_bytes": size,
        "compressed_bytes": compressed,
        "compressed_how": how,
        "budget_bytes": budget,
        "top_files": [],
        "baseline": None,
        "reduction": None,
    }

    for size_bytes, path in top_files(args.image, args.top):
        report["top_files"].append({"bytes": size_bytes, "path": path})

    if args.baseline:
        base = image_size(args.baseline)
        report["baseline"] = {"image": args.baseline, "unpacked_bytes": base}
        report["reduction"] = round((1 - size / base) * 100, 1) if base else None

    problems = []
    if report["unpacked_bytes"] > budget:
        problems.append(
            f"over budget: {human(report['unpacked_bytes'])} > {human(budget)}"
        )
    if report["reduction"] is not None and report["reduction"] < 60.0:
        problems.append(
            f"not enough smaller than baseline: {report['reduction']}% < 60%"
        )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"image: {report['image']}")
        print(f"unpacked: {human(report['unpacked_bytes'])}")
        shown = human(compressed) if compressed else "UNVERIFIED"
        print(f"compressed download: {shown} ({how})")
        if report["baseline"]:
            print(f"baseline: {human(report['baseline']['unpacked_bytes'])} "
                  f"({report['baseline']['image']})")
            print(f"reduction: {report['reduction']}%")
        print(f"top {args.top} files:")
        for row in report["top_files"]:
            print(f"  {row['bytes'] / (1024 * 1024):8.1f} MB  {row['path']}")
        if problems:
            print("gate: FAIL")
            for problem in problems:
                print(f"  {problem}")
        else:
            print("gate: PASS")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
