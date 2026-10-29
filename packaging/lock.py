"""Generate hash-pinned lockfiles from the vLLM requirement files.

The clone is read-only: requirement files are read, never edited. Wheels are
downloaded for one target platform, hashed, and written in the format
`package_thin.py` installs from:

    name==version \
        --hash=sha256:<hex>

Locks are build artifacts and live under `out/locks/`; they are not committed.
`--check` re-hashes an existing wheelhouse against an existing lock, which is
the gate the runtime image relies on (nothing is resolved at install time).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rules as rules_mod  # noqa: E402

CHUNK = 1024 * 1024
WHEEL = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)(?:-\d+)?-[^-]+-[^-]+-[^-]+\.whl$"
)
LOCK_LINE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[^\s\\]+)")
HASH_LINE = re.compile(r"^\s*--hash=sha256:([0-9a-f]{64})")

# The image target. Kept explicit so a lock is never silently rebuilt for another
# platform; a mismatch here is why hash-pinned installs fail in CI.
TARGETS = {
    "linux-x86_64": {
        "python_version": "3.12",
        "implementation": "cp",
        "abi": "cp312",
        "platform": "manylinux_2_28_x86_64",
    },
    "linux-aarch64": {
        "python_version": "3.12",
        "implementation": "cp",
        "abi": "cp312",
        "platform": "manylinux_2_28_aarch64",
    },
}


def read_requirements(path: Path) -> list[str]:
    """Requirement names only: comments, markers, and option lines are dropped."""
    names: list[str] = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0]
        if name:
            names.append(name)
    return names


def requirements_for(rule: dict) -> list[str]:
    """Requirement names from the read-only clone named by the rule."""
    spec = rule.get("requirements")
    if isinstance(spec, str):
        clone, files = Path(spec), None
    else:
        clone, files = Path(spec["clone"]), list(spec.get("files") or [])
    if not clone.is_dir():
        raise SystemExit(
            f"clone not found: {clone} (rule 'requirements' is a read-only path)"
        )
    names: list[str] = []
    for rel in files or ["requirements/common.txt", "requirements/cuda.txt"]:
        file = clone / rel
        if file.exists():
            names.extend(read_requirements(file))
    if not names:
        raise SystemExit(f"no requirement files found under {clone}")
    return sorted(set(names))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                return h.hexdigest()
            h.update(block)


def download(names: list[str], wheelhouse: Path, target: str) -> None:
    args = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--dest",
        str(wheelhouse),
    ]
    spec = TARGETS[target]
    for key in ("python_version", "implementation", "abi", "platform"):
        args.extend([f"--{key}", spec[key]])
    args.extend(names)
    subprocess.run(args, check=True)


def write_lock(wheelhouse: Path, dest: Path, target: str) -> int:
    entries = []
    for path in sorted(wheelhouse.glob("*.whl")):
        m = WHEEL.match(path.name)
        if not m:
            continue
        key = (rules_mod.norm(m.group("name")), m.group("version"))
        entries.append((key, sha256(path)))
    if not entries:
        raise SystemExit(f"wheelhouse has no wheels: {wheelhouse}")
    lines = []
    for (name, version), digest in sorted(entries):
        lines.append(f"{name}=={version} \\\n    --hash=sha256:{digest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    print(f"lock: {dest} ({len(entries)} entries, target {target})")
    return len(entries)


def read_lock(lock: Path) -> dict[tuple[str, str], set[str]]:
    """(name, version) -> set of sha256 digests, as written by write_lock."""
    entries: dict[tuple[str, str], set[str]] = {}
    current: tuple[str, str] | None = None
    for line in lock.read_text().splitlines():
        m = LOCK_LINE.match(line.strip())
        if m:
            current = (rules_mod.norm(m.group("name")), m.group("version"))
            entries.setdefault(current, set())
            continue
        h = HASH_LINE.match(line)
        if h and current is not None:
            entries[current].add(h.group(1))
    return entries


def check(lock: Path, wheelhouse: Path) -> list[str]:
    """Every wheel in the wheelhouse must match the lock, and vice versa."""
    want = read_lock(lock)
    have = {}
    for path in wheelhouse.glob("*.whl"):
        m = WHEEL.match(path.name)
        if m:
            have[(rules_mod.norm(m.group("name")), m.group("version"))] = sha256(path)

    problems: list[str] = []
    for key, digest in sorted(have.items()):
        if key not in want:
            problems.append(f"{key[0]}=={key[1]}: wheel not in lock")
        elif digest not in want[key]:
            problems.append(f"{key[0]}=={key[1]}: hash mismatch")
    for key in sorted(want):
        if key not in have:
            problems.append(f"{key[0]}=={key[1]}: no wheel in wheelhouse")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate or check a hash-pinned vLLM lockfile."
    )
    ap.add_argument("--rule", default="vllm", help="rule name under packaging/rules/")
    ap.add_argument("--out", default="out/locks", help="where the lock is written")
    ap.add_argument("--wheelhouse", default="out/wheelhouse")
    ap.add_argument("--target", default="linux-x86_64", choices=sorted(TARGETS))
    ap.add_argument(
        "--download", action="store_true", help="download wheels before writing"
    )
    ap.add_argument(
        "--check", action="store_true", help="re-hash a wheelhouse against a lock"
    )
    args = ap.parse_args()

    rule = rules_mod.load(args.rule)
    names = requirements_for(rule)

    if args.check:
        lock = Path(args.out) / f"{rule['name']}-{args.target}.txt"
        problems = check(lock, Path(args.wheelhouse))
        print("\n".join(problems) if problems else f"lock check: clean ({lock})")
        return 1 if problems else 0

    if args.download:
        download(names, Path(args.wheelhouse), args.target)

    lock = Path(args.out) / f"{rule['name']}-{args.target}.txt"
    write_lock(Path(args.wheelhouse), lock, args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
