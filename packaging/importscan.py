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

DIST_INFO_SUFFIX = rules_mod.DIST_INFO_SUFFIX


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


def module_file(module: str, tree: Path) -> str | None:
    """The tree-relative file an absolute dotted import resolves to.

    `ray.util.collective.collective_group.cuda_stream` is the file
    `ray/util/collective/collective_group/cuda_stream.py`, or the package folder
    `.../cuda_stream/__init__.py`. Relative imports are skipped: they resolve
    inside the importing package, not through the tree root.

    Stubs are not targets. `triton/_C/libtriton/linear_layout.pyi` and the
    `torch/_C/*.pyi` files are type stubs for compiled extensions, and pruning
    them does not break the import: measured in the materialised selection view,
    `triton._C.libtriton.linear_layout`, `torch._C._profiler` and `tvm_ffi.core`
    all resolve to the `.so` and import fine with the stub folders gone. Counting
    `.pyi` here reported 324 findings that cannot happen at runtime, so the gate
    keeps `.py` targets only.
    """
    if module.startswith("."):
        return None
    base = tree.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate.relative_to(tree).as_posix()
    return None


def relative_targets(path: Path, tree: Path) -> list[str]:
    """Tree-relative files a module-level relative import points at.

    `module_file` resolves through the tree root and skips a leading dot, so a
    target inside the importing package is invisible to it. That is the gap the
    harness build fell into: `triton/experimental/gluon/language/amd/__init__.py`
    has `from . import gfx1250`, the arch filter dropped that package, and the
    gate reported nothing while the image failed at import.
    """
    try:
        parsed = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return []
    pkg = path.parent
    found: list[str] = []
    for node in parsed.body:
        if not isinstance(node, ast.ImportFrom) or not node.level:
            continue
        base = pkg
        for _ in range(node.level - 1):
            base = base.parent
        for part in (node.module or "").split("."):
            if part:
                base = base / part
        candidates = [base.with_suffix(".py"), base / "__init__.py"]
        for name in (alias.name for alias in node.names):
            candidates.extend((base / f"{name}.py", base / name / "__init__.py"))
        for candidate in candidates:
            rel = candidate.relative_to(tree).as_posix()
            if rel != path.relative_to(tree).as_posix() and candidate.is_file():
                found.append(rel)
    return found


def imports_pointing_at_dropped(
    tree: Path,
    selected: set[str],
) -> list[str]:
    """Shipped files that import a module the selection has dropped.

    A denied distribution leaves no file at all, so the deny gate covers that
    case. A pruned file inside a kept distribution does not: the importer ships
    and the imported module does not, which is a runtime ImportError on a path
    the gate would otherwise pass. Both are reported together so a selection is
    judged by what it can actually import, not only by what it keeps.
    """
    problems = set()
    for rel in sorted(selected):
        if not rel.endswith(".py"):
            continue
        for module in module_level_modules(tree / rel):
            target = module_file(module, tree)
            if target is not None and target not in selected:
                problems.add(
                    f"{rel}: imports {module} at module level, but {target} "
                    "is not in the selection"
                )
        for target in relative_targets(tree / rel, tree):
            if target not in selected:
                problems.add(
                    f"{rel}: imports a module inside its own package at module "
                    f"level, but {target} is not in the selection"
                )
    return sorted(problems)


def top_level_map(site_packages: Path) -> dict[str, str]:
    """top-level directory -> normalised distribution name, read from RECORD."""
    owner: dict[str, str] = {}
    for dist in sorted(site_packages.glob(f"*{DIST_INFO_SUFFIX}")):
        record = dist / "RECORD"
        if not record.exists():
            continue
        # the dist-info folder is `<name>-<version>.dist-info`; drop the suffix
        # then split at the last hyphen, so a distribution name containing a
        # hyphen is not cut short
        stem = dist.name[: -len(DIST_INFO_SUFFIX)]
        name = rules_mod.norm(stem.rsplit("-", 1)[0])
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
