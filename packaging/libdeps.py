#!/usr/bin/env python3
"""What a shipped binary needs from the machine it runs on, read with no tools
and no third-party packages, plus the rule "everything imported is bundled, or
belongs to the OS, or belongs to the GPU driver".

    python packaging/libdeps.py <package folder>   # exit 0 when every import
                                                   # is covered, 1 with the list
    python packaging/libdeps.py --copy <wheelhouse> --out <stage> [--arch sm_90a]

Why the closure copy is the biggest size lever: a `nvidia-*` wheel is not only
its `.so` - it also carries static `.a` archives, headers, `pkgconfig`, `nvvm`,
`cupti`, `cusparseLt` and tool binaries. Reading `DT_NEEDED` of
`torch/lib/libtorch_cuda.so` and of vLLM's own extensions tells us which
libraries are actually loaded, and only those are copied.

The GPU driver boundary: `libcuda.so.1` and `libnvidia-*` are never bundled. The
host provides them through nvidia-container-toolkit. Only non-stub runtime
libraries (cudart, cublas, cublasLt, nvJitLink) travel in the package.

Libraries opened with `dlopen` never show up in `DT_NEEDED`; pass `--also GLOB`
for those.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arch as arch_mod  # noqa: E402

SYSTEM_DIRS = ("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")

#: Sonames every distribution has (glibc and the compiler runtime). A missing
#: one is named in SYSTEM-LIBS.txt so a failure on a target machine can be
#: traced to a package, not to a guess.
SYSTEM_LIBS = (
    "linux-vdso.so.1", "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1",
    "libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0", "librt.so.1",
    "libutil.so.1", "libresolv.so.2", "libnss_*.so.*", "libgcc_s.so.1",
    "libstdc++.so.6", "libgomp.so.1", "libatomic.so.1", "libz.so.1",
    "libnuma.so.1", "libdrm.so.2", "libelf.so.1", "libtinfo.so.*",
    "libcrypt.so.1", "libffi.so.*", "libpci.so.3", "libudev.so.1", "libdl",
)

#: From the GPU driver package: never redistributable, never bundled.
DRIVER_LIBS = (
    "libcuda.so.1", "libcuda.so", "libnvidia-*.so*", "libnvcuvid.so*", "libvulkan.so*",
    "libGLX*.so*", "libEGL*.so*", "libOpenCL.so*", "libdrm_amdgpu.so*",
)

LDD_LINE = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(0x")

#: Name of the report written beside a package. Not part of the manifest: it is
#: produced after the manifest is hashed, so `verify` must know to skip it.
SYSTEM_LIBS_FILE = "SYSTEM-LIBS.txt"
ELF_MAGIC = b"\x7fELF"

#: SHT_DYNAMIC, DT_NULL, DT_NEEDED, DT_SONAME, DT_RPATH, DT_RUNPATH.
SHT_DYNAMIC = 6
DT_NULL, DT_NEEDED, DT_SONAME, DT_RPATH, DT_RUNPATH = 0, 1, 14, 15, 24
FORBIDDEN_SUFFIXES = (".a", ".o", ".cmake", ".prl")


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(4) == ELF_MAGIC
    except OSError:
        return False


def _dyn_strings(path: Path) -> tuple[list[tuple[int, int]], int, bytes]:
    """(tag, value) pairs of the SHT_DYNAMIC section, the string-table offset,
    and the file bytes."""
    data = path.read_bytes()
    is64 = data[4] == 2
    le = "<" if data[5] == 1 else ">"
    if is64:
        shoff, = struct.unpack_from(le + "Q", data, 0x28)
        shentsize, shnum = struct.unpack_from(le + "HH", data, 0x3A)
    else:
        shoff, = struct.unpack_from(le + "I", data, 0x20)
        shentsize, shnum = struct.unpack_from(le + "HH", data, 0x2E)
    for i in range(shnum):
        base = shoff + i * shentsize
        stype, = struct.unpack_from(le + "I", data, base + 4)
        if stype != SHT_DYNAMIC:
            continue
        if is64:
            offset, size, link = struct.unpack_from(le + "QQI", data, base + 0x18)
        else:
            offset, size, link = struct.unpack_from(le + "III", data, base + 0x10)
        sbase = shoff + link * shentsize
        wsize = 0x18 if is64 else 0x10
        wcode = "Q" if is64 else "I"
        stroff = struct.unpack_from(le + wcode, data, sbase + wsize)[0]
        step = 16 if is64 else 8
        entries = []
        for o in range(offset, offset + size, step):
            tag, val = struct.unpack_from(le + ("qQ" if is64 else "iI"), data, o)
            if tag == DT_NULL:
                break
            entries.append((tag, val))
        return entries, stroff, data
    return [], 0, data


def _cstr(data: bytes, offset: int) -> str:
    end = data.index(b"\0", offset)
    return data[offset:end].decode("utf-8", "replace")


def elf_needed(path: Path) -> list[str]:
    """DT_NEEDED sonames of an ELF file (what ldd lists). Libraries opened with
    dlopen are not visible here."""
    entries, stroff, data = _dyn_strings(path)
    return [_cstr(data, stroff + val) for tag, val in entries if tag == DT_NEEDED]


def elf_soname(path: Path) -> str | None:
    """The name its users ask the loader for. A static archive or a non-ELF file
    has none."""
    entries, stroff, data = _dyn_strings(path)
    for tag, val in entries:
        if tag == DT_SONAME:
            return _cstr(data, stroff + val)
    return None


def elf_rpath(path: Path) -> str:
    entries, stroff, data = _dyn_strings(path)
    found = [
        _cstr(data, stroff + val)
        for tag, val in entries
        if tag in (DT_RPATH, DT_RUNPATH)
    ]
    return " ".join(found)


def is_system(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in SYSTEM_LIBS)


def is_driver(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in DRIVER_LIBS)


def is_static(path: Path | str) -> bool:
    """Static artifacts are never part of a runtime package, and the gate
    refuses them; `select` is called with a relative string, so accept both."""
    return Path(path).suffix in FORBIDDEN_SUFFIXES


def uncovered(folder: Path, files: list[str] | None = None) -> list[tuple[str, str]]:
    """(file, imported library) pairs that are neither in the package, nor the
    OS's, nor the driver's. Empty means a clean machine runs it."""
    folder = Path(folder)
    if files is None:
        rels = [
            p.relative_to(folder).as_posix()
            for p in sorted(folder.rglob("*"))
            if p.is_file() and not p.is_symlink()
        ]
    else:
        rels = files
    present = {Path(rel).name for rel in rels}
    problems: list[tuple[str, str]] = []
    for rel in sorted(rels):
        path = folder / rel
        if not is_elf(path):
            continue
        for name in elf_needed(path):
            if name in present:
                continue
            if is_system(name) or is_driver(name):
                continue
            problems.append((rel, name))
    return problems


def ldd_missing(folder: Path, archs: list[str]) -> list[str]:
    """What the dynamic linker cannot find inside this tree, run on the machine
    that builds it."""
    if not shutil.which("ldd"):
        return []
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(folder)
    problems: list[str] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or not is_elf(path):
            continue
        try:
            done = subprocess.run(
                ["ldd", str(path)],
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        for line in done.stdout.splitlines():
            match = LDD_LINE.match(line)
            if "->" in line and "not found" in line:
                name = line.split()[0]
                if is_system(name) or is_driver(name):
                    continue
                problems.append(f"{path.name}: {name} not found")
            elif match and not match.group(2).startswith(SYSTEM_DIRS):
                if not Path(match.group(2)).exists():
                    problems.append(
                        f"{path.name}: {match.group(1)} resolves outside this tree"
                    )
    return sorted(set(problems))


def copy_closure(
    src: Path,
    dest: Path,
    archs: list[str],
    also: list[str] | None = None,
) -> list[str]:
    """Copy only the libraries the shipped binaries actually load, under their
    soname. Returns notes."""
    dest.mkdir(parents=True, exist_ok=True)
    roots = [
        p
        for p in sorted(src.rglob("*"))
        if p.is_file() and is_elf(p) and not is_static(p)
    ]
    wanted: dict[str, Path] = {}

    def add(path: Path) -> None:
        name = elf_soname(path) or path.name
        if name in wanted:
            return
        if not arch_mod.arch_wanted(name, archs):
            return
        wanted[name] = path

    for path in roots:
        add(path)
        for name in elf_needed(path):
            hit = next((p for p in roots if (elf_soname(p) or p.name) == name), None)
            if hit is not None:
                add(hit)

    for pattern in also or []:
        for path in sorted(src.rglob(pattern)):
            if path.is_file() and is_elf(path):
                add(path)

    notes = []
    for name, path in sorted(wanted.items()):
        target = dest / name
        if target.exists() and os.path.samefile(path, target):
            continue
        shutil.copy2(path, target)
        notes.append(f"{name} <- {path}")
    return notes


def write_system_libs(folder: Path, files: list[str]) -> list[str]:
    """SYSTEM-LIBS.txt beside the package: the libraries the distribution must
    provide."""
    needed: set[str] = set()
    for rel in files:
        path = folder / rel
        if is_elf(path):
            needed.update(n for n in elf_needed(path) if is_system(n))
    lines = [
        "Libraries this package does not carry: the distribution must provide them.",
        "The GPU driver's libraries (libcuda.so.1, libnvidia-*) are never bundled.",
        "",
    ]
    lines += sorted(needed)
    (folder / SYSTEM_LIBS_FILE).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return sorted(needed)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check or build the shared-library closure of a package."
    )
    ap.add_argument("folder", nargs="?", help="package folder to check")
    ap.add_argument(
        "--copy",
        metavar="SRC",
        help="copy only the DT_NEEDED closure from SRC into --out",
    )
    ap.add_argument("--out", metavar="DEST", help="where the closure copy goes")
    ap.add_argument(
        "--arch", default="", help="comma-separated architectures to keep, e.g. sm_90a"
    )
    ap.add_argument(
        "--also",
        action="append",
        default=[],
        help="extra glob for dlopen'd libraries (repeatable)",
    )
    args = ap.parse_args()

    archs = arch_mod.parse(args.arch) if args.arch else arch_mod.detect()[0]

    if args.copy:
        if not args.out:
            raise SystemExit("libdeps.py: --copy needs --out")
        notes = copy_closure(Path(args.copy), Path(args.out), archs, args.also)
        for note in notes:
            print(note)
        print(f"{len(notes)} libraries copied")
        return 0

    if not args.folder:
        raise SystemExit("libdeps.py: give a package folder, or --copy SRC --out DEST")

    folder = Path(args.folder)
    problems = uncovered(folder)
    static = [p for p in sorted(folder.rglob("*")) if p.is_file() and is_static(p)]
    if static:
        print("static artifacts in the shipped tree (must not ship):")
        for path in static:
            print(f"  {path.relative_to(folder).as_posix()}")

    if problems:
        print("UNCOVERED IMPORTS:")
        for rel, name in problems:
            print(f"  {rel} -> {name}")
        return 1

    print("every import is bundled, or belongs to the OS, or belongs to the GPU driver")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
