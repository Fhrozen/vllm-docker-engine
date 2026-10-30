#!/usr/bin/env python3
"""Bytecode as delivered: what the runtime image can actually use.

The rule has a `bytecode` key, and the pipeline writes bytecode into the
package. What that means at import time is not obvious, so each part is
answered by measurement:

  header   - what a `.pyc` records, and whether it still matches the source it
             is read against in three views: as built, as extracted from the
             deterministic archive, as copied by the runtime stage
  sourceless - whether a package with no `.py` files can be imported at all
  writable  - whether the runtime user can rebuild the cache at launch
  startup   - what a cache that cannot be validated costs on the first import

Two facts drive the whole answer. A timestamp `.pyc` is validated against the
source's `mtime` and size, so a package whose mtimes are rewritten (the
deterministic archive forces `SOURCE_DATE_EPOCH` on every member) invalidates
every cache it carries. And the runtime stage runs as `USER 1000:1000` on a
root-owned tree, so a cache that fails validation cannot be refreshed -- it is
recompiled in memory on every start.

Hash-based pycs are validated by hashing the source with
`importlib.util.source_hash` (blake2b, 8 bytes), so they survive the forced
mtime. `compileall` writes them when `SOURCE_DATE_EPOCH` is set, but only for
the files it actually compiles: a cache that is already valid is skipped, so a
build mixes both kinds unless `invalidation_mode` forces hash-based output.

Run it inside the container:

  bash docker/run.sh script tools/evaluate_bytecode.py --pkg out/compare-cupy2/source/engine
"""

from __future__ import annotations

import argparse
import compileall
import importlib.util
import py_compile
import json
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packaging"))

import package as pkg_mod  # noqa: E402

TAG = sys.implementation.cache_tag
RUNTIME_UID = 1000
ENTRY = "vllm"


def header(path: Path) -> tuple[int, int, int]:
    """(flags, field1, field2) from the 16-byte pyc header."""
    with open(path, "rb") as f:
        block = f.read(16)
    if len(block) < 16:
        return (-1, 0, 0)
    flags = struct.unpack("<I", block[4:8])[0]
    field1 = struct.unpack("<I", block[8:12])[0]
    field2 = struct.unpack("<I", block[12:16])[0]
    return flags, field1, field2


def invalidation(flags: int) -> str:
    if not flags & 1:
        return "timestamp"
    return "checked-hash" if flags & 2 else "unchecked-hash"


def mirror(pyc: Path) -> Path | None:
    """The source a `__pycache__` entry was compiled from, if it is still there."""
    name = pyc.name
    if not name.endswith(f".{TAG}.pyc"):
        return None
    source = pyc.parent.parent / f"{name[: -len(f'.{TAG}.pyc')]}.py"
    return source if source.is_file() else None


def scan(pkg: Path, mtime_of) -> dict:
    """Classify every shipped pyc against the source as the loader would see it."""
    counts: dict[str, int] = {}
    samples: list[str] = []
    for pyc in sorted(pkg.rglob("*.pyc")):
        flags, f1, f2 = header(pyc)
        mode = invalidation(flags)
        src = mirror(pyc)
        if src is None:
            key = f"{mode}/sourceless"
            counts[key] = counts.get(key, 0) + 1
            continue
        if mode == "timestamp":
            ok = f1 == int(mtime_of(src)) and f2 == src.stat().st_size
        else:
            # The loader hashes with blake2b(digest_size=8), not sha256.
            ok = struct.pack("<II", f1, f2) == importlib.util.source_hash(
                src.read_bytes()
            )
        key = f"{mode}/{'valid' if ok else 'invalid'}"
        counts[key] = counts.get(key, 0) + 1
        if not ok and len(samples) < 3:
            samples.append(pyc.relative_to(pkg).as_posix())
    return {"counts": counts, "samples": samples}


def sourceless_probe(root: Path, layout: str) -> str:
    """Import a module whose source is gone, in the two layouts a package can have."""
    pkg = root / "probe_pkg"
    shutil.rmtree(pkg, ignore_errors=True)
    (pkg / "__pycache__").mkdir(parents=True)
    source = pkg / "mod.py"
    source.write_text("VALUE = 42\n")
    compileall.compile_dir(str(pkg), quiet=1)
    pyc = pkg / "__pycache__" / f"mod.{TAG}.pyc"
    if not pyc.is_file():
        return "no pyc produced"
    source.unlink()
    if layout == "cache":
        target = pkg / "__pycache__" / f"mod.{TAG}.pyc"
    else:
        # legacy sourceless: the pyc sits where the source would have been
        target = pkg / "mod.pyc"
        shutil.copyfile(pyc, target)
    result = subprocess.run(
        [sys.executable, "-c", "import probe_pkg.mod as m; print(m.VALUE)"],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ, PYTHONPATH=str(root)),
    )
    detail = (result.stdout + result.stderr).strip().splitlines()
    return f"rc={result.returncode} {detail[-1] if detail else ''}"


def timed_import(view: Path, uid: int | None) -> tuple[float, str]:
    """Wall time of the entry import, as the runtime user would run it."""

    def drop_privileges() -> None:
        os.setgid(uid)
        os.setuid(uid)

    started = time.perf_counter()
    # A folder with no source resolves as a namespace package and imports without
    # running any code, so the check has to say whether a file was executed.
    code = (
        "import vllm; print('namespace package: no code ran' "
        "if vllm.__file__ is None else 'ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        # getpass.getuser() falls back to pwd when the env names are unset, and uid
        # 1000 has no passwd entry in the container.
        env=dict(os.environ, PYTHONPATH=str(view), LOGNAME="uid1000"),
        preexec_fn=drop_privileges if uid is not None else None,
    )
    elapsed = time.perf_counter() - started
    detail = (result.stdout + result.stderr).strip().splitlines()
    return elapsed, f"rc={result.returncode} {detail[-1] if detail else 'ok'}"


def module_count(view: Path, uid: int) -> int:
    """How many shipped modules the entry import actually executes."""
    code = (
        "import sys; sys.path.insert(0, %r); __import__('vllm'); "
        "print(sum(1 for m in sys.modules.values() "
        "if getattr(m, '__file__', '') and %r in str(m.__file__)))"
        % (str(view), str(view))
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ, PYTHONPATH=str(view), LOGNAME="uid1000"),
        preexec_fn=lambda: (os.setgid(uid), os.setuid(uid)),
    )
    out = (result.stdout + result.stderr).strip().splitlines()
    return int(out[-1]) if out and out[-1].isdigit() else -1


def force_hash_view(pkg: Path, out: Path) -> dict:
    """Recompile the whole cache as checked-hash, then read it as the archive delivers it."""
    view = out / "view-force-hash"
    shutil.rmtree(view, ignore_errors=True)
    started = time.perf_counter()
    shutil.copytree(pkg, view, copy_function=shutil.copy2)
    # Force the mtime first: compileall skips a cache it considers valid, so
    # without this the existing timestamp pycs are left as they are.
    for path in view.rglob("*.py"):
        os.utime(path, (pkg_mod.SOURCE_DATE_EPOCH, pkg_mod.SOURCE_DATE_EPOCH))
    # The enum is required: compileall ignores the string form of
    # invalidation_mode and still writes unchecked-hash pycs.
    compileall.compile_dir(
        str(view),
        quiet=1,
        workers=os.cpu_count() or 1,
        invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
        stripdir=str(view),
        prependdir="/",
    )
    compiled = time.perf_counter() - started
    for path in view.rglob("*"):
        if path.is_file():
            os.utime(path, (pkg_mod.SOURCE_DATE_EPOCH, pkg_mod.SOURCE_DATE_EPOCH))
    result = scan(view, lambda p: pkg_mod.SOURCE_DATE_EPOCH)
    result["compiled_s"] = round(compiled, 1)
    result["pyc_bytes"] = sum(p.stat().st_size for p in view.rglob("*.pyc"))
    return result


def import_check(view: Path, uid: int) -> dict:
    """Import the package as the runtime user and say whether code actually ran.

    A folder with no readable source imports as a namespace package, so a plain
    return code is not enough: ``__file__`` has to resolve for the package to be
    usable.
    """
    code = (
        "import sys; sys.path.insert(0, %r); import vllm; "
        "print(getattr(vllm, '__file__', None), getattr(vllm, '__version__', None))"
        % str(view)
    )
    env = dict(os.environ, LOGNAME=f"uid{uid}")
    run = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd="/tmp",
        check=False,
    )
    return {
        "returncode": run.returncode,
        "resolved": run.stdout.strip(),
        "error": (run.stderr.strip().splitlines() or [""])[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure what shipped bytecode is worth.")
    ap.add_argument("--pkg", required=True, help="delivered package directory")
    ap.add_argument("--out", default="out/eval-bytecode", help="scratch area")
    ap.add_argument(
        "--archive",
        action="store_true",
        help="write and extract the real archive instead of simulating its effect",
    )
    ap.add_argument("--entry-time", action="store_true", help="run the startup timings")
    ap.add_argument(
        "--force-hash",
        action="store_true",
        help="recompile the whole cache as checked-hash and re-read it as the archive delivers it",
    )
    ap.add_argument(
        "--legacy-fix",
        action="store_true",
        help="mirror each pyc to the module location (the layout that is importable without source)",
    )
    args = ap.parse_args()

    pkg = Path(args.pkg).resolve()
    if not (pkg / pkg_mod.MANIFEST).exists():
        # a variant root holds one package directory named for the build
        for child in pkg.iterdir():
            if child.is_dir() and (child / pkg_mod.MANIFEST).exists():
                pkg = child
                break
    if not (pkg / pkg_mod.MANIFEST).exists():
        raise SystemExit(f"no manifest under {pkg}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"package: {pkg}")
    pycs = list(pkg.rglob("*.pyc"))
    print(f"py files: {len(list(pkg.rglob('*.py')))}  pyc files: {len(pycs)}")
    print(
        f"cache bytes as delivered: {sum(p.stat().st_size for p in pycs) / 1e6:.1f} MB"
    )

    print("\n1. pyc header against the source as delivered in the package")
    first = scan(pkg, lambda p: p.stat().st_mtime)
    print(first["counts"])
    for sample in first["samples"]:
        print(f"  invalid sample: {sample}")

    print("\n2. pyc header against the source as the deterministic archive delivers it")
    extract = out / "extracted"
    if args.archive:
        archive = pkg_mod.write_archive(
            pkg, out / "sample.tar.gz", json.loads((pkg / pkg_mod.MANIFEST).read_text())
        )
        extract.mkdir(exist_ok=True)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(extract)
    else:
        # Same effect as extraction without compressing 10 GB: write_archive sets
        # every member's mtime to SOURCE_DATE_EPOCH and its mode to 0o555.
        shutil.rmtree(extract, ignore_errors=True)
        shutil.copytree(pkg, extract, copy_function=shutil.copy2)
        for path in extract.rglob("*"):
            if path.is_file():
                os.utime(path, (pkg_mod.SOURCE_DATE_EPOCH, pkg_mod.SOURCE_DATE_EPOCH))
                path.chmod(0o555)
    print(f"archive member mtime: {pkg_mod.SOURCE_DATE_EPOCH}")
    second = scan(extract, lambda p: pkg_mod.SOURCE_DATE_EPOCH)
    print(second["counts"])
    for sample in second["samples"]:
        print(f"  invalid sample: {sample}")

    print("\n3. sourceless layouts")
    print("  __pycache__ only:", sourceless_probe(out / "probe-cache", "cache"))
    print("  pyc at module location:", sourceless_probe(out / "probe-legacy", "legacy"))

    print("\n4. can the runtime user write the cache?")
    ro = out / "read-only"
    ro.mkdir(exist_ok=True)
    target = ro / "x.py"
    target.write_text("y = 1\n")
    (ro / "__pycache__").mkdir(exist_ok=True)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, %r); import x; print('imported')" % str(ro),
        ],
        capture_output=True,
        text=True,
        check=False,
        preexec_fn=lambda: (os.setgid(RUNTIME_UID), os.setuid(RUNTIME_UID)),
    )
    wrote = (ro / "__pycache__" / f"x.{TAG}.pyc").exists()
    print(
        f"  uid {RUNTIME_UID} on a root-owned tree: {(probe.stdout + probe.stderr).strip()}"
    )
    print(f"  cache written: {wrote}")

    if args.force_hash:
        print("\n5. cache recompiled as checked-hash, read as the archive delivers it")
        result = force_hash_view(pkg, out)
        print(result["counts"])
        print(
            f"  cache bytes: {result['pyc_bytes'] / 1e6:.1f} MB  compile: {result['compiled_s']} s"
        )

    if args.entry_time:
        print("\n6. startup cost, read-only tree, uid 1000")
        fresh = out / "view-fresh"
        shutil.rmtree(fresh, ignore_errors=True)
        shutil.copytree(pkg, fresh, copy_function=shutil.copy2)
        stale = out / "view-stale"
        shutil.rmtree(stale, ignore_errors=True)
        shutil.copytree(pkg, stale, copy_function=shutil.copy2)
        for path in stale.rglob("*.py"):
            os.utime(path, (pkg_mod.SOURCE_DATE_EPOCH, pkg_mod.SOURCE_DATE_EPOCH))
        for name, view in (("fresh cache", fresh), ("invalid cache", stale)):
            for path in view.rglob("*"):
                if path.is_file():
                    path.chmod(0o555)
            elapsed, detail = timed_import(view, RUNTIME_UID)
            print(f"  {name}: {elapsed:.1f} s  {detail}")
        print(
            f"  modules executed by the entry import: {module_count(fresh, RUNTIME_UID)}"
        )

    print("\n7. is the delivered package importable as it stands")
    print("  as delivered:", import_check(pkg, RUNTIME_UID))
    if args.legacy_fix:
        legacy = out / "view-legacy"
        shutil.rmtree(legacy, ignore_errors=True)
        shutil.copytree(pkg, legacy, copy_function=shutil.copy2)
        for pyc in legacy.rglob("*.pyc"):
            if pyc.parent.name != "__pycache__":
                continue
            # legacy layout is module.pyc, without the cache tag
            mirror = pyc.parent.parent / (pyc.name.split(".")[0] + ".pyc")
            if not mirror.exists():
                shutil.copy2(pyc, mirror)
        print(
            "  pyc mirrored to the module location:", import_check(legacy, RUNTIME_UID)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
