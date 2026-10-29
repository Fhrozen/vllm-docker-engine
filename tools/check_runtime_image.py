#!/usr/bin/env python3
"""Runtime image gate (R01) plus the image-config gate.

The rule is simple: a runtime image may not contain a compiler, an installer, a
version control tool, or a downloader. Those are read from the image itself, not
from the Dockerfile, because a Dockerfile can be edited after the fact and an
image cannot.

Gates that cannot be read from the image are reported as UNVERIFIED rather than
assumed. Layout, boot, and registry reachability are outside what image metadata
exposes, so they are stated as UNVERIFIED here and checked separately.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packaging"))

import rules as rules_mod  # noqa: E402

UNVERIFIED = ("layout", "boot", "hub_reach")


def docker(args: list[str]) -> str:
    if shutil.which("docker") is None:
        raise SystemExit("docker is not available; the runtime gate cannot run")
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        detail = (result.stdout + result.stderr).strip()
        raise SystemExit(f"docker {' '.join(args)} failed:\n{detail}")
    return result.stdout


def inspect(image: str) -> dict:
    out = docker(["image", "inspect", "--format", "{{json .Config}}", image])
    return json.loads(out)


def present_tools(image: str, tools: list[str]) -> list[str]:
    """Which forbidden tools are actually present in the image."""
    checks = " ".join(
        f"command -v {tool} >/dev/null 2>&1 && printf '%s\\n' {tool};" for tool in tools
    )
    out = docker(["run", "--rm", "--entrypoint", "/bin/sh", image, "-c", checks])
    return sorted({line.strip() for line in out.splitlines() if line.strip()})


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gate a runtime image against its rule file."
    )
    ap.add_argument("--image", required=True)
    ap.add_argument("--rule", default="vllm", help="rule name under packaging/rules/")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rule = rules_mod.load(args.rule)
    config = inspect(args.image)
    tools = present_tools(args.image, rule["forbidden_tools"])

    problems: list[str] = []
    report: dict = {
        "image": args.image,
        "rule": rule["id"],
        "user": config.get("User") or "",
        "entrypoint": config.get("Entrypoint") or [],
        "healthcheck": (config.get("Healthcheck") or {}).get("Test") or [],
        "env": config.get("Env") or [],
        "labels": config.get("Labels") or {},
        "forbidden_tools_present": tools,
        "unverified": list(UNVERIFIED),
    }

    if tools:
        problems.append(f"forbidden tools present: {', '.join(tools)}")
    if not report["user"]:
        problems.append("image runs as root (expected USER 1000:1000)")
    if not report["healthcheck"]:
        problems.append("image has no HEALTHCHECK")
    if not report["entrypoint"]:
        problems.append("image has no ENTRYPOINT")

    base = report["labels"].get("org.opencontainers.image.base.name", "")
    if base and ":" in base and "@" not in base:
        # A base that is not digest-pinned can change under a rebuild. This is a
        # warning, not a failure, because the pin is a build-time choice.
        report["base_not_pinned"] = base

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"image: {report['image']} (rule {report['rule']})")
        print(f"user: {report['user'] or 'root'}")
        print(f"entrypoint: {report['entrypoint']}")
        print(f"healthcheck: {report['healthcheck']}")
        print(f"forbidden tools: {'none' if not tools else ', '.join(tools)}")
        print(f"unverified: {', '.join(report['unverified'])}")
        if report.get("base_not_pinned"):
            print(f"warning: base is not digest-pinned ({report['base_not_pinned']})")
        print("gate: " + ("FAIL" if problems else "PASS"))
        for problem in problems:
            print(f"  {problem}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
