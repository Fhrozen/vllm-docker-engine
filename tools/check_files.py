#!/usr/bin/env python3
"""The pre-commit hook checks that have no standalone tool in the container:
check-yaml, end-of-file-fixer, mixed-line-ending, trailing-whitespace,
debug-statements.

Run on files passed explicitly. A gate with no files is not a pass.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - the tools image has pyyaml
    yaml = None

MD_LINEBREAK_EXT = "md"
DEBUG_CALLS = {"breakpoint"}


def trailing_ws(line: str, is_markdown: bool) -> bool:
    """Same rule as the pre-commit hook: [ \t]+$ , with the markdown exception
    for a hard line break (two spaces or a backslash)."""
    stripped = line.rstrip("\n")
    if not stripped or stripped[-1] not in " \t":
        return False
    if is_markdown:
        return not (stripped.endswith("  ") or stripped.endswith("\\"))
    return True


def check(path: Path, fix: bool) -> list[str]:
    problems: list[str] = []
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")

    # mixed-line-ending --fix=lf
    if b"\r\n" in raw or b"\r" in raw:
        problems.append("mixed line endings")
        if fix:
            raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            text = raw.decode("utf-8", errors="replace")

    # end-of-file-fixer: exactly one final newline, not zero and not more.
    if not raw.endswith(b"\n"):
        problems.append("no final newline")
        if fix:
            raw = raw + b"\n"
            text = raw.decode("utf-8", errors="replace")
    elif raw.endswith(b"\n\n"):
        problems.append("more than one final newline")
        if fix:
            raw = raw.rstrip(b"\n") + b"\n"
            text = raw.decode("utf-8", errors="replace")

    # trailing-whitespace
    bad = [
        i
        for i, line in enumerate(text.splitlines(True), 1)
        if trailing_ws(line, path.suffix == ".md")
    ]
    if bad:
        problems.append(f"trailing whitespace on line {bad[0]}")
        if fix:
            lines = []
            for line in text.splitlines(True):
                if trailing_ws(line, path.suffix == ".md"):
                    # Keep the newline: rstrip(" \t") alone leaves it in place,
                    # so adding another one would duplicate it.
                    lines.append(line.rstrip("\n").rstrip(" \t") + "\n")
                else:
                    lines.append(line)
            raw = "".join(lines).encode()

    # check-yaml / json
    if path.suffix in (".yaml", ".yml"):
        if yaml is None:
            problems.append("UNVERIFIED: pyyaml is not installed")
        else:
            try:
                list(yaml.safe_load_all(text))
            except yaml.YAMLError as exc:
                problems.append(f"invalid YAML ({exc}")
    elif path.suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            problems.append(f"invalid JSON ({exc}")

    # debug-statements
    if path.suffix == ".py":
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import) and any(
                a.name == "pdb" for a in node.names
            ):
                problems.append("pdb import")
            elif isinstance(node, ast.ImportFrom) and node.module == "pdb":
                problems.append("pdb import")
            elif isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if name in DEBUG_CALLS or name == "set_trace":
                    problems.append("debug statement")

    if fix and problems:
        path.write_bytes(raw)
    return problems


def main(argv: list[str]) -> int:
    files = [a for a in argv if a != "--fix"]
    fix = "--fix" in argv
    if not files:
        print("check_files: no files given. Pass the changed files explicitly.")
        return 1
    failed = 0
    for arg in files:
        path = Path(arg)
        if not path.is_file():
            print(f"{path}: not a file")
            failed += 1
            continue
        problems = check(path, fix)
        if problems:
            print(f"{path}: " + "; ".join(problems))
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
