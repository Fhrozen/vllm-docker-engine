#!/usr/bin/env python3
"""Rule files: what a package is allowed to contain. Committed, read by both
packaging variants.

A rule file is JSON. Keys:

  schema_version, id, name, version, platform, backend, entrypoint, python
  requirements  - requirement files the lock is generated from, relative to
                  the source clone
  allow         - globs a file must match to be shipped (default: everything)
  deny          - top-level package names or globs that are never shipped
  prune         - globs removed from the shipped tree (tests, static libs,
                  headers, pkgconfig)
  bytecode      — "source" or "pyc-only"
  strip         — run `strip --strip-debug` on ELF files as an explicit pipeline step
  arch_filter   — "auto" (detected by command) or a list of CUDA architectures
  system_libs   — sonames the distribution provides, recorded in SYSTEM-LIBS.txt
  driver_libs   — sonames the GPU driver provides, never bundled
  runtime_apt   — packages the runtime image installs
  forbidden_tools — must be absent from the runtime image (gate R01)

Nothing here imports anything from the reference clones.
"""

from __future__ import annotations

import fnmatch
import json
import re
from pathlib import Path

RULES_DIR = Path(__file__).resolve().parent / "rules"


def norm(name: str) -> str:
    """PEP 503 name normalisation, so `PyNvVideoCodec` and `pynvvideocodec`
    are the same package."""
    return re.sub(r"[-_.]+", "-", name).lower()


def load(path: Path | str) -> dict:
    rule_path = Path(path)
    if not rule_path.is_absolute():
        # a bare rule name means packaging/rules/<name>.json
        if rule_path.suffix != ".json":
            rule_path = RULES_DIR / f"{rule_path}.json"
        else:
            rule_path = RULES_DIR / rule_path
    with rule_path.open("r", encoding="utf-8") as stream:
        rule = json.load(stream)
    core = (
        "schema_version",
        "id",
        "name",
        "version",
        "platform",
        "backend",
        "entrypoint",
    )
    for key in core:
        if key not in rule:
            raise SystemExit(f"{rule_path}: rule file is missing the key '{key}'")
    if rule["schema_version"] != 2:
        raise SystemExit(
            f"{rule_path}: schema_version must be 2, got {rule['schema_version']}"
        )
    return rule


def matches(rel: str, patterns: list[str]) -> bool:
    """Glob match against a package-relative path. `**` spans folder separators."""
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def top_level(rel: str) -> str:
    """The package a file belongs to: the first path segment, or the wheel name
    for a loose file."""
    first = rel.split("/", 1)[0]
    return first


def denied(rel: str, rule: dict) -> bool:
    pkg = norm(top_level(rel))
    deny = {norm(pattern) for pattern in rule.get("deny", [])}
    return pkg in deny or matches(pkg, list(rule.get("deny", [])))


def pruned(rel: str, rule: dict) -> bool:
    return matches(rel, rule.get("prune", []))
