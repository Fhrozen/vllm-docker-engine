"""Generate hash-pinned lockfiles from the vLLM requirement files.

The clone is read-only: requirement files are read, never edited. Wheels are
downloaded for one target platform, hashed, and written in the format
`package_thin.py` installs from:

    name==version \
        --hash=sha256:<hex>

Locks are build artifacts and live under `out/locks/`; they are not committed.
`--check` re-hashes an existing wheelhouse against an existing lock, which is
the gate the runtime image relies on (nothing is resolved at install time).

Two facts decided this by measurement, not by assumption:

* pip does not widen a `--platform` request across manylinux revisions. With
  `--platform manylinux_2_28_x86_64` it rejects `blake3`, whose only x86_64
  wheels are tagged `manylinux_2_17`. A single pinned platform tag therefore
  cannot resolve a real requirement set.
* pip does not accept `abi3` when `--abi cp312` is pinned, so `triton` (whose
  wheels are `cp312-abi3`) is invisible to the resolver.

So the lock is generated **inside the target environment** — the builder
container for that target — where pip resolves against its own tags, and the
target is enforced afterwards by `check()`, which reads the tags back out of
every wheel filename. A wheel that came from the wrong platform, ABI, or a
glibc newer than the runtime base fails the gate instead of silently shipping.
`--cross` keeps the explicit-flag path for a target the current machine cannot
run; that path is best-effort because pip can hit dependency conflicts when it
cannot see the target's wheels.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rules as rules_mod  # noqa: E402

CHUNK = 1024 * 1024
WHEEL = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)(?:-\d+)?"
    r"-(?P<python>[^-]+)-(?P<abi>[^-]+)-(?P<platform>[^-]+)\.whl$"
)
LOCK_LINE = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)==(?P<version>[^\s\\]+)")
HASH_LINE = re.compile(r"^\s*--hash=sha256:([0-9a-f]{64})")
MANYLINUX = re.compile(r"^manylinux_(\d+)_(\d+)_([A-Za-z0-9_]+)$")
# Legacy aliases still present in wheel filenames. They name the same glibc
# floor, so `manylinux2014_x86_64` is accepted where `manylinux_2_17_x86_64` is.
LEGACY_GLIBC = {
    "manylinux1": (2, 5),
    "manylinux2010": (2, 12),
    "manylinux2014": (2, 17),
}

# The image target. Kept explicit so a lock is never silently rebuilt for another
# platform; a mismatch here is why hash-pinned installs fail in CI.
#
# `glibc` is the runtime base floor, not the build host: a wheel tagged above it
# cannot load in the runtime image. ubuntu:24.04 is glibc 2.39, so the whole
# manylinux_2_x range below it is accepted, which is what a real requirement set
# needs (blake3 publishes 2_17, numpy publishes 2_27/2_28).
TARGETS = {
    "linux-x86_64": {
        "python_version": "3.12",
        "implementation": "cp",
        "abi": ("cp312", "abi3", "none"),
        "platform": (
            "manylinux_2_17_x86_64",
            "manylinux_2_27_x86_64",
            "manylinux_2_28_x86_64",
            "any",
        ),
        "arch": "x86_64",
        "glibc": (2, 39),
    },
    "linux-aarch64": {
        "python_version": "3.12",
        "implementation": "cp",
        "abi": ("cp312", "abi3", "none"),
        "platform": (
            "manylinux_2_17_aarch64",
            "manylinux_2_27_aarch64",
            "manylinux_2_28_aarch64",
            "any",
        ),
        "arch": "aarch64",
        "glibc": (2, 39),
    },
}


def current_environment() -> dict[str, str]:
    """What the running interpreter actually is, as wheel tags."""
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "implementation": "cp"
        if sys.implementation.name == "cpython"
        else sys.implementation.name,
        "abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "arch": platform.machine(),
    }


def python_tags(target: str) -> tuple[str, ...]:
    version = TARGETS[target]["python_version"].replace(".", "")
    return (f"cp{version}", f"py{version}", "py3")


def _floor(part: str) -> tuple[tuple[int, int], str] | None:
    """(glibc floor, arch) for one platform-tag part, or None if it is not one."""
    m = MANYLINUX.match(part)
    if m:
        return (int(m.group(1)), int(m.group(2))), m.group(3)
    head, _, arch = part.partition("_")
    if head in LEGACY_GLIBC:
        return LEGACY_GLIBC[head], arch
    return None


def platform_ok(tag: str, target: str) -> bool:
    """A platform field may be a chain of aliases; every part must fit the target.

    musllinux is not accepted: the runtime base is a glibc image, and a musl
    wheel cannot load there even when its floor is lower.
    """
    spec = TARGETS[target]
    for part in tag.split("."):
        if part == "any" or part == f"linux_{spec['arch']}":
            continue
        floor = _floor(part)
        if floor is None or floor[1] != spec["arch"] or floor[0] > spec["glibc"]:
            return False
    return True


def tag_problems(filename: str, target: str) -> list[str]:
    """Wheel filename read back against the declared target."""
    m = WHEEL.match(filename)
    if not m:
        return [f"{filename}: not a wheel filename"]
    spec = TARGETS[target]
    problems = []
    if m.group("python") not in python_tags(target):
        problems.append(f"{filename}: python tag {m.group('python')} not in {target}")
    if m.group("abi") not in spec["abi"]:
        problems.append(
            f"{filename}: abi tag {m.group('abi')} not accepted by {target}"
        )
    if not platform_ok(m.group("platform"), target):
        floor = spec["glibc"]
        problems.append(
            f"{filename}: platform tag {m.group('platform')} not accepted by "
            f"{target} (glibc floor {floor[0]}.{floor[1]})"
        )
    return problems


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
    """Requirement names from the read-only clone named by the rule.

    A rule may carry `requirements.names` as a snapshot of what the clone's
    requirement files resolve to. That snapshot is what a build uses, because
    the clone is never in the build context: a build must not depend on a
    read-only clone being present. The clone path is the source of the snapshot,
    so `names` is generated from it, never hand-written.

    Names the rule denies are dropped. Downloading a package the rule will not
    ship is wasted work, and a lock that contains it cannot be matched by the
    shipped tree, so the deny list is applied here rather than after the fact.

    The package named by the rule is part of the set as well: the thin variant
    installs it from a binary wheel, and without it the installed tree has no
    `*.dist-info` for `package.resolve_version` to read. The engine variant
    builds that package from the clone, so for it this entry is a superset.
    """
    spec = rule.get("requirements")
    if isinstance(spec, str):
        clone, files, snapshot = Path(spec), None, []
    else:
        clone = Path(spec["clone"])
        files = list(spec.get("files") or [])
        snapshot = list(spec.get("names") or [])

    if snapshot:
        names = [rules_mod.norm(n) for n in snapshot]
    else:
        if not clone.is_dir():
            raise SystemExit(
                f"clone not found: {clone} (rule 'requirements' is a read-only "
                "path, and no 'names' snapshot is recorded in the rule)"
            )
        names = []
        for rel in files or ["requirements/common.txt", "requirements/cuda.txt"]:
            file = clone / rel
            if file.exists():
                names.extend(read_requirements(file))
        if not names:
            raise SystemExit(f"no requirement files found under {clone}")

    kept = [n for n in names if not rules_mod.denied(n, rule)]
    dropped = len(names) - len(kept)
    if dropped:
        print(f"requirements: {dropped} denied name(s) not downloaded")

    self_name = rules_mod.norm(rule["name"])
    if rules_mod.denied(self_name, rule):
        raise SystemExit(
            f"rule denies its own package: {self_name} (name={rule['name']})"
        )
    kept.append(self_name)
    return sorted(set(kept))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                return h.hexdigest()
            h.update(block)


def download(
    names: list[str], wheelhouse: Path, target: str, cross: bool = False
) -> None:
    args = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--dest",
        str(wheelhouse),
    ]
    if cross:
        spec = TARGETS[target]
        for key in ("python_version", "implementation", "abi", "platform"):
            # pip spells these with hyphens; the dict keys use underscores so the
            # target table reads like a wheel tag. Repeating a flag is how pip is
            # told to accept several tags, which a single pinned tag cannot do.
            for value in spec[key]:
                args.extend([f"--{key.replace('_', '-')}", value])
        print(
            f"download: cross-target {target}; pip cannot see the target's wheels, "
            "so dependency conflicts are possible here"
        )
    else:
        env = current_environment()
        print(
            f"download: resolved for the running environment "
            f"(cp{env['python_version'].replace('.', '')}-{env['abi']}-{env['arch']}); "
            f"check() enforces target {target} on the wheel filenames"
        )
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

    # One distribution may not appear twice. A wheelhouse is a directory, so a
    # second download run into the same folder leaves the older version next to
    # the newer one, and the lock then asks pip to install both at once, which
    # cannot be resolved. The lock is only a reading of the wheelhouse, so the
    # conflict has to be reported here rather than left for the install.
    versions: dict[str, list[str]] = {}
    for name, version in sorted({k for k, _ in entries}):
        versions.setdefault(name, []).append(version)
    dupes = {n: v for n, v in versions.items() if len(v) > 1}
    if dupes:
        raise SystemExit(
            "wheelhouse has more than one version of the same distribution, "
            "and a lock that lists both cannot be installed:\n  "
            + "\n  ".join(
                f"{n}: {', '.join(sorted(v))}" for n, v in sorted(dupes.items())
            )
            + "\n  the wheelhouse is the source of the lock, so start from an "
            "empty one or remove the version that is not wanted"
        )

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


def check(lock: Path, wheelhouse: Path, target: str = "linux-x86_64") -> list[str]:
    """Every wheel in the wheelhouse must match the lock, and vice versa.

    Tags are part of the check: a wheel that satisfies the hash but carries a
    python, ABI, or platform tag outside the target is not usable in the runtime
    image, and reporting it here is what keeps a lock generated in the wrong
    environment from passing.
    """
    want = read_lock(lock)
    have = {}
    problems: list[str] = []
    for path in sorted(wheelhouse.glob("*.whl")):
        m = WHEEL.match(path.name)
        if not m:
            continue
        problems.extend(tag_problems(path.name, target))
        have[(rules_mod.norm(m.group("name")), m.group("version"))] = sha256(path)

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
    ap.add_argument(
        "--cross",
        action="store_true",
        help="pin pip's target tags instead of resolving for the running environment",
    )
    args = ap.parse_args()

    rule = rules_mod.load(args.rule)
    names = requirements_for(rule)

    if args.check:
        lock = Path(args.out) / f"{rule['name']}-{args.target}.txt"
        problems = check(lock, Path(args.wheelhouse), args.target)
        print("\n".join(problems) if problems else f"lock check: clean ({lock})")
        return 1 if problems else 0

    if args.download:
        download(names, Path(args.wheelhouse), args.target, cross=args.cross)

    lock = Path(args.out) / f"{rule['name']}-{args.target}.txt"
    write_lock(Path(args.wheelhouse), lock, args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
