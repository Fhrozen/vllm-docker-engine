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
  cuda_variant  — folder that ships one copy per CUDA series (e.g.
                  nvidia_cutlass_dsl); copies of other series are pruned. The
                  series is decided by command, never by a default in a Dockerfile
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
DIST_INFO_SUFFIX = ".dist-info"


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


def denied(rel: str, rule: dict, owner: dict[str, str] | None = None) -> bool:
    """Whether a file is excluded by the deny list.

    Deny entries are distribution names. The first path segment is an import
    name, and the two are not always the same (`flashinfer-python` installs
    `flashinfer`, `opencv-python-headless` installs `cv2`). When the RECORD
    owner map is given, a folder is denied when the distribution that owns it
    is denied, which is what the deny list means.
    """
    top = top_level(rel)
    deny = {norm(pattern) for pattern in rule.get("deny", [])}
    if top.endswith(DIST_INFO_SUFFIX):
        # a denied distribution ships nothing, its metadata included
        stem = top[: -len(DIST_INFO_SUFFIX)]
        return norm(stem.rsplit("-", 1)[0]) in deny
    pkg = norm(top)
    if pkg in deny or matches(pkg, list(rule.get("deny", []))):
        return True
    if owner:
        dist = owner.get(top)
        if dist and norm(dist) in deny:
            return True
    return False


def pruned(rel: str, rule: dict) -> bool:
    return matches(rel, rule.get("prune", []))


def cuda_variant(rel: str, rule: dict, series: int | None) -> bool:
    """Whether a file belongs to a CUDA series other than the declared one.

    `nvidia_cutlass_dsl` ships a copy of the same runtime library for cu12 and
    for cu13 (41.6 MB each), and the environment resolves one of them. The rule
    names the folder; the series comes from detection, so the dead copy is
    pruned by measurement rather than by assumption.
    """
    folder = rule.get("cuda_variant")
    if not folder or series is None:
        return False
    parts = rel.split("/")
    if parts[0] != folder or len(parts) < 2:
        return False
    match = re.fullmatch(r"cu(\d+)", parts[1])
    return bool(match) and int(match.group(1)) != int(series)
