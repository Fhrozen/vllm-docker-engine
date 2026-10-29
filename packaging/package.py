"""Engine variant of the packaging layer: select, verify, manifest, archive.

A rule file (see `rules.py`) turns a source tree -- an installed site-packages,
or an extracted wheelhouse -- into a package directory named

    out/<version>-<platform>-<backend>[-<arch>]/

Nothing is trusted before it is verified. Files are hashed after strip and
bytecode, so the manifest describes the package as delivered; the package is
staged as `<name>.partial`, checked against the manifest, and only then renamed
into place. Existing output is refused, never overwritten. Symlinks are never
followed, never copied, never archived.

The archive writer is deterministic: sorted members, fixed timestamps, no
symlinks or hardlinks, so the same input tree always produces the same bytes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arch as arch_mod  # noqa: E402
import libdeps  # noqa: E402
import rules as rules_mod  # noqa: E402

CHUNK = 1024 * 1024
SOURCE_DATE_EPOCH = int(os.environ.get("SOURCE_DATE_EPOCH", "1600000000"))
MANIFEST = "manifest.json"


def sha256(path: Path) -> str:
    """Streamed hash: large shared libraries must not be read into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                return h.hexdigest()
            h.update(block)


def select(
    src: Path,
    rule: dict,
    archs: list[str],
    owner: dict[str, str] | None = None,
) -> list[str]:
    """Relative paths that the rule wants, sorted, no symlinks, no statics."""
    picked: list[str] = []
    want = libdeps.machine_for_platform(rule["platform"])
    for path in sorted(src.rglob("*")):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(src).as_posix()
        if rules_mod.denied(rel, rule, owner) or rules_mod.pruned(rel, rule):
            continue
        if not rules_mod.matches(rel, rule["allow"]):
            continue
        if libdeps.is_static(rel):
            continue
        if rule.get("arch_filter") and not arch_mod.arch_wanted(rel, archs):
            continue
        # An ELF object for another architecture cannot be loaded by this image.
        # The build run hit this as a strip failure on cutlass-dsl stubs built for
        # aarch64, which is the same defect seen from the other side.
        if want is not None and libdeps.elf_machine(path) not in (None, want):
            continue
        picked.append(rel)
    return sorted(picked)


def resolve_version(src: Path, rule: dict) -> str:
    """Rules may declare version: auto; the real version comes from dist-info."""
    if rule["version"] != "auto":
        return rule["version"]
    want = rules_mod.norm(rule["name"])
    for path in src.glob("*.dist-info"):
        stem = path.name.removesuffix(".dist-info")
        parts = stem.split("-", 1)
        if len(parts) == 2 and rules_mod.norm(parts[0]) == want:
            return parts[1]
    raise SystemExit(f"version: auto but no dist-info for {rule['name']} under {src}")


def package_dir(out: Path, rule: dict, archs: list[str]) -> Path:
    parts = [rule["version"], rule["platform"], rule["backend"]]
    if rule.get("arch_filter") == "auto":
        parts.append("-".join(archs))
    return out / "-".join(parts)


def build_id(entries: list[dict]) -> str:
    """Digest of the selection itself, so a rebuild is comparable to a cache."""
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda e: e["path"]):
        h.update(f"{e['path']}\0{e['sha256']}\0{e['bytes']}\n".encode())
    return h.hexdigest()


def write_manifest(
    pkg: Path,
    rule: dict,
    archs: list[str],
    entries: list[dict],
) -> dict:
    manifest = {
        "schema_version": rule["schema_version"],
        "id": rule["id"],
        "name": rule["name"],
        "version": rule["version"],
        "platform": rule["platform"],
        "backend": rule["backend"],
        "archs": archs,
        "min_driver": rule.get("min_driver", ""),
        "build_id": build_id(entries),
        "entrypoint": rule["entrypoint"],
        "files": sorted(entries, key=lambda e: e["path"]),
        "signature": "",
    }
    (pkg / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def strip_tree(pkg: Path) -> int:
    """`strip` is an explicit pipeline step, not a side effect of a build flag."""
    if shutil.which("strip") is None:
        print("strip: tool unavailable, skipping")
        return 0
    done = 0
    for path in pkg.rglob("*"):
        if path.is_file() and libdeps.is_elf(path):
            subprocess.run(["strip", "--strip-debug", str(path)], check=False)
            done += 1
    return done


def bytecode_tree(pkg: Path, rule: dict) -> int:
    import compileall

    mode = rule.get("bytecode", "skip")
    if mode == "skip":
        return 0
    os.environ.setdefault("SOURCE_DATE_EPOCH", str(SOURCE_DATE_EPOCH))
    # stripdir + prependdir fix the filename recorded inside the code object.
    # Without them the pyc carries the build directory, so two builds of the
    # same source into two output folders are not the same package.
    compileall.compile_dir(
        str(pkg),
        quiet=1,
        workers=os.cpu_count() or 1,
        stripdir=str(pkg),
        prependdir="/",
    )
    if mode != "pyc-only":
        return 0
    tag = sys.implementation.cache_tag
    removed = 0
    for path in pkg.rglob("*.py"):
        mirror = path.parent / "__pycache__" / f"{path.stem}.{tag}.pyc"
        if mirror.exists():
            path.unlink()
            removed += 1
    return removed


def verify(pkg: Path, full: bool = True) -> list[str]:
    """Re-hash the package against the manifest. Returns problems, empty = clean."""
    manifest = json.loads((pkg / MANIFEST).read_text())
    problems: list[str] = []
    listed = set()
    for entry in manifest["files"]:
        path = pkg / entry["path"]
        listed.add(entry["path"])
        if path.is_symlink():
            problems.append(f"{entry['path']}: symlink in package")
            continue
        if not path.is_file():
            problems.append(f"{entry['path']}: missing")
            continue
        if path.stat().st_size != entry["bytes"]:
            size = path.stat().st_size
            problems.append(f"{entry['path']}: size {size} != {entry['bytes']}")
            continue
        if full and sha256(path) != entry["sha256"]:
            problems.append(f"{entry['path']}: hash mismatch")
    # The manifest can only vouch for the files it names, so anything else in
    # the tree is a problem: an unlisted file is unverified, and a symlink is
    # refused outright because it can point outside the package.
    for path in sorted(pkg.rglob("*")):
        rel = path.relative_to(pkg).as_posix()
        if rel in (MANIFEST, libdeps.SYSTEM_LIBS_FILE):
            continue
        if path.is_symlink():
            problems.append(f"{rel}: symlink in package")
        elif path.is_file() and rel not in listed:
            problems.append(f"{rel}: file in package is not in the manifest")
    return problems


def check_archive(dest: Path) -> list[str]:
    """P11: verify sha256 and size before anything is unpacked."""
    problems: list[str] = []
    with tarfile.open(dest, "r:gz") as tf:
        members: dict[str, tarfile.TarInfo] = {}
        manifest = None
        for member in tf.getmembers():
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                problems.append(f"{member.name}: escapes the target directory")
                continue
            if member.type != tarfile.REGTYPE:
                problems.append(f"{member.name}: not a regular file ({member.type})")
                continue
            members[member.name] = member
            if member.name == MANIFEST:
                fileobj = tf.extractfile(member)
                if fileobj is None:
                    problems.append(f"{member.name}: unreadable")
                    continue
                try:
                    manifest = json.loads(fileobj.read().decode())
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    problems.append(f"{member.name}: not readable as JSON ({exc})")
                    continue
        if manifest is None:
            return problems + ["archive has no manifest"]
        if not isinstance(manifest, dict):
            return problems + ["manifest is not a JSON object"]
        if not isinstance(manifest.get("files"), list):
            return problems + ["manifest has no file list"]
        listed = set()
        for entry in manifest["files"]:
            path = entry.get("path")
            if not isinstance(path, str):
                problems.append("manifest entry has no path")
                continue
            listed.add(path)
            if len(entry.get("sha256", "")) != 64:
                problems.append(f"{path}: manifest hash is not a sha256")
                continue
            member = members.get(path)
            if member is None:
                problems.append(f"{path}: manifest names a file the archive lacks")
                continue
            if member.size != entry.get("bytes"):
                problems.append(f"{path}: size {member.size} != {entry.get('bytes')}")
                continue
            fileobj = tf.extractfile(member)
            if fileobj is None:
                problems.append(f"{path}: unreadable")
                continue
            digest = hashlib.sha256()
            while True:
                block = fileobj.read(CHUNK)
                if not block:
                    break
                digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                problems.append(f"{path}: hash mismatch")
        for name in sorted(members):
            if name != MANIFEST and name not in listed:
                problems.append(f"{name}: member is not in the manifest")
    return problems


def write_archive(pkg: Path, dest: Path, manifest: dict) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    handle = open(dest, "wb")
    with gzip.GzipFile(
        filename="",
        mode="wb",
        fileobj=handle,
        compresslevel=9,
        mtime=SOURCE_DATE_EPOCH,
    ) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tf:
            info = tarfile.TarInfo(MANIFEST)
            info.size = len(payload)
            info.mtime = SOURCE_DATE_EPOCH
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            tf.addfile(info, BytesIO(payload))
            for entry in sorted(manifest["files"], key=lambda e: e["path"]):
                info = tarfile.TarInfo(entry["path"])
                info.size = entry["bytes"]
                info.mtime = SOURCE_DATE_EPOCH
                info.mode = 0o555
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                with open(pkg / entry["path"], "rb") as f:
                    tf.addfile(info, f)
    return dest


def build(src: Path, rule: dict, out: Path, archs: list[str], archive: bool) -> Path:
    if not src.is_dir():
        raise SystemExit(f"source tree not found: {src}")
    rule["version"] = resolve_version(src, rule)
    target = package_dir(out, rule, archs)
    if target.exists():
        raise SystemExit(f"refusing to overwrite existing package: {target}")
    out.mkdir(parents=True, exist_ok=True)

    archs = list(archs)
    picked = select(src, rule, archs)
    if not picked:
        raise SystemExit("rule selected nothing -- check allow/deny/prune patterns")

    staged = out / (target.name + ".partial")
    if staged.exists():
        shutil.rmtree(staged)
    entries: list[dict] = []
    for rel in picked:
        source = src / rel
        dest = staged / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)

    if rule.get("strip"):
        strip_tree(staged)
    bytecode_tree(staged, rule)

    # The manifest describes the package as delivered, so it is hashed after
    # strip and bytecode, not at copy time: both can change bytes or remove
    # files, and a manifest that names a file the package no longer holds is
    # not a manifest anyone can verify on another machine.
    for path in sorted(staged.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(staged).as_posix()
        entries.append(
            {"path": rel, "sha256": sha256(path), "bytes": path.stat().st_size}
        )

    manifest = write_manifest(staged, rule, archs, entries)
    problems = verify(staged, full=True)
    if problems:
        shutil.rmtree(staged)
        raise SystemExit("verify failed before rename:\n  " + "\n  ".join(problems))

    uncovered = libdeps.uncovered(staged)
    if uncovered:
        shutil.rmtree(staged)
        detail = "\n  ".join(f"{p}: {n}" for p, n in uncovered)
        raise SystemExit("uncovered imports:\n  " + detail)

    libdeps.write_system_libs(staged, [e["path"] for e in manifest["files"]])
    staged.rename(target)
    print(f"package: {target}")
    total = sum(e["bytes"] for e in manifest["files"])
    print(f"files: {len(manifest['files'])}  bytes: {total}")
    print(f"build_id: {manifest['build_id']}")

    if archive:
        dest = out / (target.name + ".tar.gz")
        write_archive(target, dest, manifest)
        gate = check_archive(dest)
        if gate:
            raise SystemExit("archive gate:\n  " + "\n  ".join(gate))
        print(f"archive: {dest} ({dest.stat().st_size} bytes)")
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a vLLM package from a rule file.")
    ap.add_argument(
        "src",
        nargs="?",
        help="source tree to select from (site-packages or extraction)",
    )
    ap.add_argument("--rule", help="rule name or path under packaging/rules/")
    ap.add_argument("--out", default="out/packages")
    ap.add_argument("--arch", help="comma-separated archs; default is detection")
    ap.add_argument(
        "--archive", action="store_true", help="also write a deterministic tar.gz"
    )
    ap.add_argument(
        "--verify", action="store_true", help="re-hash an existing package and exit"
    )
    ap.add_argument(
        "--check-archive", action="store_true", help="run the P11 archive gate"
    )
    ap.add_argument("--no-strip", action="store_true")
    args = ap.parse_args()

    if args.check_archive:
        if not args.rule and not args.src:
            raise SystemExit("give an archive path")
        problems = check_archive(Path(args.src))
        print("\n".join(problems) if problems else "archive gate: clean")
        return 1 if problems else 0

    rule = rules_mod.load(args.rule or "vllm")
    archs = arch_mod.resolve(args.arch)

    if args.verify:
        pkg = Path(args.src)
        problems = verify(pkg)
        print("\n".join(problems) if problems else f"verify: clean ({pkg})")
        return 1 if problems else 0

    if args.no_strip:
        rule["strip"] = False

    build(Path(args.src), rule, Path(args.out), archs, args.archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
