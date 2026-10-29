"""Which shipped files execute an import at import time, and who owns what.

A name the rule denies is not shipped. If a file that *is* shipped imports it at
module scope, the image breaks the moment that file is imported -- no amount of
`tp_size = 1` or unused-feature reasoning saves it. Imports inside a function, a
`try`, or an `if` guard are not executed at import time, so a denied name reached
only through those is optional in the sense the rule means.

Ownership comes from each distribution's `RECORD`, not from guessing a package
name into an import name: `flashinfer-python` is imported as `flashinfer`,
`opencv-python-headless` as `cv2`, and a name-based mapping would be a guess.
`RECORD` lists the files a wheel installs, so the first path segment is the top
level that import statements actually see.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rules as rules_mod  # noqa: E402

DIST_INFO_SUFFIX = ".dist-info"


def head(module: str) -> str:
    """The top-level package an import statement resolves to."""
    return module.split(".")[0]


def module_level_modules(path: Path) -> list[str]:
    """Dotted module names imported at module scope, in source order."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.append(node.module)
    return found


def module_level_imports(path: Path, prefix: str | None = None) -> list[str]:
    """Import statements at module scope, as readable lines.

    `prefix` restricts the result to one package (the distributed gate asks for
    `torch.distributed`); `None` returns every module-scope import.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if prefix is None or alias.name.startswith(prefix):
                    found.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if prefix is None or module.startswith(prefix):
                names = ", ".join(alias.name for alias in node.names)
                found.append(f"from {module} import {names}")
    return found


def top_level_map(site_packages: Path) -> dict[str, str]:
    """top-level directory -> normalised distribution name, read from RECORD."""
    owner: dict[str, str] = {}
    for dist in sorted(site_packages.glob(f"*{DIST_INFO_SUFFIX}")):
        record = dist / "RECORD"
        if not record.exists():
            continue
        name = rules_mod.norm(dist.name.split("-")[0])
        for line in record.read_text(encoding="utf-8", errors="replace").splitlines():
            first = line.split(",")[0].strip()
            if "/" not in first:
                continue
            top = first.split("/")[0]
            if top.endswith(DIST_INFO_SUFFIX):
                continue
            owner.setdefault(top, name)
    return owner


def runtime_conflicts(
    root: Path,
    rule: dict,
    owner: dict[str, str],
    selected: set[str] | None = None,
) -> list[str]:
    """Shipped files that import a denied package at module scope.

    Each conflict names the file, the import, and the denied distribution, so the
    decision is recorded as evidence rather than as an opinion.

    `selected` is the set of relative paths the rule's selection keeps. Pass it
    when the shipped tree is scanned after bytecode conversion: a `pyc-only` rule
    removes the `.py` files, so a scan of the shipped tree alone finds nothing and
    the gate passes on an empty set. The files that were chosen are the files that
    ship, whichever form they are shipped in.
    """
    denied = {rules_mod.norm(name) for name in (rule.get("deny") or [])}
    problems = set()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if selected is not None and rel.as_posix() not in selected:
            continue
        source = owner.get(rel.parts[0]) if rel.parts else None
        for module in module_level_modules(path):
            top = head(module)
            dist = owner.get(top)
            if dist in denied and dist != source:
                problems.add(
                    f"{rel}: imports {module} at module level, but {dist} is denied"
                )
    return sorted(problems)
