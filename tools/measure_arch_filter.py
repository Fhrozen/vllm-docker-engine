#!/usr/bin/env python3
"""What the architecture filter drops, and which of it is still imported.

The filter is a name test on the whole relative path: an entry named for an
architecture other than the declared one is left out. That is right for cubins,
fatbins and per-arch shared libraries, and wrong for Python modules - a package
folder named `gfx1250` is a module its parent imports at module level, so
dropping it breaks the import rather than saving bytes. This measures both sides:
how much the filter cuts, split by source or data, and which dropped source is
reached by a module-level import from a kept file.

Run inside docker:
    bash docker/run.sh script tools/measure_arch_filter.py --arch sm_86
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packaging"))

import arch as arch_mod  # noqa: E402
import package as package_mod  # noqa: E402
import rules as rules_mod  # noqa: E402


def module_level_relative(path: Path) -> list[tuple[str, list[str]]]:
    """Relative imports at module level as (module, imported names), dots kept."""
    try:
        parsed = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return []
    found: list[tuple[str, list[str]]] = []
    for node in parsed.body:
        if isinstance(node, ast.ImportFrom) and node.level:
            found.append(
                ("." * node.level + (node.module or ""), [a.name for a in node.names])
            )
    return found


def resolve_relative(tree: Path, rel: str, module: str, names: list[str]) -> str | None:
    """Tree-relative file a module-level relative import points at.

    `from . import gfx1250` inside a package resolves to `gfx1250.py` or
    `gfx1250/__init__.py` next to the importing file; `from .gfx1250 import x`
    resolves through the dotted module part.
    """
    pkg = (tree / rel).parent
    level = len(module) - len(module.lstrip("."))
    base = pkg
    for _ in range(level - 1):
        base = base.parent
    for part in [p for p in module.split(".") if p]:
        base = base / part
    candidates = [base.with_suffix(".py"), base / "__init__.py"]
    for name in names:
        candidates.extend(
            (base / name, base / f"{name}.py", base / name / "__init__.py")
        )
    for candidate in candidates:
        if candidate.is_file() and candidate.relative_to(tree).as_posix() != rel:
            return candidate.relative_to(tree).as_posix()
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out/run2/vllm.venv/lib/python3.12/site-packages")
    ap.add_argument("--rule", default="vllm")
    ap.add_argument("--arch", default="sm_86")
    ap.add_argument("--out", default="out/arch-filter")
    args = ap.parse_args()

    tree = Path(args.src)
    rule = rules_mod.load(args.rule)
    archs = arch_mod.parse(args.arch)

    # Same selection, run twice: the only difference is the arch test, so the
    # difference between the two sets is exactly what the filter drops.
    no_arch = set(package_mod.select(tree, rule, []))
    with_arch = set(package_mod.select(tree, rule, archs))
    dropped = no_arch - with_arch

    def size(paths: set[str]) -> int:
        return sum((tree / rel).stat().st_size for rel in paths)

    py = {r for r in dropped if r.endswith(".py")}
    data = dropped - py

    print(
        f"arch filter {', '.join(archs)}: {len(dropped)} entries "
        f"{size(dropped) / 1e6:.1f} MB dropped, {len(with_arch)} entries "
        f"{size(with_arch) / 1e6:.1f} MB kept"
    )
    print(f"  source dropped: {len(py)} {size(py) / 1e6:.1f} MB")
    print(f"  data dropped:   {len(data)} {size(data) / 1e6:.1f} MB")
    print("  source entries:")
    for rel in sorted(py):
        print(f"    {rel} {(tree / rel).stat().st_size} B")

    findings: list[str] = []
    for rel in sorted(with_arch):
        if not rel.endswith(".py"):
            continue
        for module, names in module_level_relative(tree / rel):
            target = resolve_relative(tree, rel, module, names)
            if target is not None and target not in with_arch:
                findings.append(f"{rel}: imports {module} -> {target} dropped")

    print(
        f"\nmodule-level relative imports pointing at source the filter drops: "
        f"{len(findings)}"
    )
    for line in findings:
        print("  " + line)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(findings) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
