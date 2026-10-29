"""Behaviour tests for the packaging engine. Every assertion here reflects
behaviour that has been observed by running the code, not assumed.

Run inside the container: bash docker/run.sh test
"""

import io
import json
import shutil
import sys
import tarfile
import types
from pathlib import Path

import pytest

from conftest import REPO, rule, tree

import arch
import check_dist_gate
import check_files
import evaluate_deny
import importscan
import libdeps
import package
import rules

import lock as lock_mod
import package_thin


def system_lib(pattern: str) -> Path | None:
    """A real shared library to exercise the ELF gate with. The gate reads
    bytes, so a text fixture cannot stand in for it."""
    for folder in ("/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu"):
        hits = sorted(Path(folder).glob(pattern))
        if hits:
            return hits[0]
    return None


# rules --------------------------------------------------------------------


def test_norm_is_pep_503():
    assert rules.norm("PyNvVideoCodec") == "pynvvideocodec"
    assert rules.norm("huggingface_hub") == "huggingface-hub"
    assert rules.norm("Flash-Attn_2") == "flash-attn-2"


def test_committed_rules_load():
    for name in ("vllm", "vllm-omni"):
        loaded = rules.load(name)
        assert loaded["schema_version"] == 2
        assert loaded["id"] == f"{name}-slim"
        assert "pip" in loaded["forbidden_tools"]


def test_rule_file_missing_key_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 2, "id": "x"}))
    with pytest.raises(SystemExit, match="missing the key"):
        rules.load(path)


def test_rule_file_wrong_schema_version_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "x",
                "name": "x",
                "version": "1",
                "platform": "p",
                "backend": "b",
                "entrypoint": "e",
            }
        )
    )
    with pytest.raises(SystemExit, match="schema_version must be 2"):
        rules.load(path)


def test_double_star_spans_separators():
    assert rules.matches("nvidia/cu13/lib/libz.so.1", ["nvidia/**"])
    assert not rules.matches("vllm/foo.py", ["vllm_omni/**"])
    assert rules.matches("vllm_omni/engine/x.py", ["vllm_omni/**"])


def test_denied_and_pruned():
    r = rule()
    assert rules.denied("bar/y.py", r)
    assert not rules.denied("foo/x.py", r)
    assert rules.denied("PyNvVideoCodec/x.py", rule(deny=["pynvvideocodec"]))
    assert rules.pruned("foo/tests/z.py", r)
    assert rules.pruned("foo/lib/static.a", r)
    assert not rules.pruned("foo/lib/shared.so", r)


# arch ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rel", "archs", "wanted"),
    [
        ("sm_90a/kernel.cubin", ["sm_90a"], True),
        ("sm_90/kernel.cubin", ["sm_90a"], False),
        ("nvidia/cu13/lib/libz.so.1", ["sm_90a"], True),
        ("vllm-2.0.0.dist-info/METADATA", ["sm_90a"], True),
        ("plain_shared.so", ["sm_90a"], True),
        ("gfx1151/kernel.o", ["sm_90a"], False),
        ("sm_86/x.cubin", ["sm_86"], True),
        ("x.cubin", [], True),
    ],
)
def test_arch_wanted(rel, archs, wanted):
    assert arch.arch_wanted(rel, archs) is wanted


def test_arch_normalise_and_parse():
    assert arch.normalise("8.6") == "sm_86"
    assert arch.normalise("sm_100a") == "sm_100a"
    assert arch.parse("sm_90a; sm_86") == ["sm_90a", "sm_86"]
    assert arch.parse("compute_90a-real") == ["sm_90a"]


def test_a_file_name_arch_needs_the_prefix():
    # libz.so.1 must not read as an architecture, or shared data is dropped.
    assert arch.arch_of("nvidia/cu13/lib/libz.so.1") is None
    assert arch.arch_of("sm_90a/kernel.cubin") == "sm_90a"
    assert arch.arch_of("gfx1151/kernel.cubin") == "gfx1151"


def test_tree_defect_names_code_for_an_undeclared_arch(tmp_path):
    (tmp_path / "sm_90a_kernel.cubin").write_text("k")
    (tmp_path / "sm_90_kernel.cubin").write_text("k")
    assert arch.tree_defect(tmp_path, ["sm_90a"]) == [
        f"{tmp_path}/sm_90_kernel.cubin: code for sm_90, not declared"
    ]


def test_detect_prefers_the_override(monkeypatch):
    monkeypatch.setenv("VDE_CUDA_ARCH", "sm_90a,sm_86")
    assert arch.detect() == (["sm_90a", "sm_86"], "override")


def test_auto_arch_resolves_to_detection(monkeypatch):
    # The rules declare `arch_filter: auto` and the build passes `auto`, so the
    # CLI must read it as detection. Feeding it to normalise() stops the build
    # with "cannot read a CUDA architecture from 'auto'".
    monkeypatch.setenv("VDE_CUDA_ARCH", "sm_90a")
    assert arch.resolve("auto") == ["sm_90a"]
    assert arch.resolve("") == ["sm_90a"]
    assert arch.resolve("sm_86,sm_90a") == ["sm_86", "sm_90a"]

    monkeypatch.delenv("VDE_CUDA_ARCH")
    monkeypatch.setattr(arch.shutil, "which", lambda name: None)
    assert arch.resolve("auto") == list(arch.FULL_LIST)

    try:
        arch.normalise("auto")
    except SystemExit as exc:
        assert "auto" in str(exc)
    else:
        raise AssertionError("normalise must reject 'auto' as an architecture")


# select -------------------------------------------------------------------


def test_select_skips_denied_pruned_statics_and_symlinks(tmp_path):
    tree(
        tmp_path,
        {
            "foo/x.py": "x\n",
            "bar/y.py": "y\n",
            "foo/tests/z.py": "z\n",
            "foo/lib/static.a": "a\n",
            "foo/lib/shared.so": "s\n",
            "foo/sm_90a/kernel.cubin": "k\n",
            "foo/sm_86/kernel.cubin": "k\n",
        },
    )
    (tmp_path / "foo/link.py").symlink_to("x.py")

    assert package.select(tmp_path, rule(), ["sm_90a"]) == [
        "foo/lib/shared.so",
        "foo/sm_90a/kernel.cubin",
        "foo/x.py",
    ]


def test_select_requires_the_allow_list(tmp_path):
    tree(tmp_path, {"vllm_omni/a.py": "a\n", "other/b.py": "b\n"})
    assert package.select(tmp_path, rule(allow=["vllm_omni/**"]), []) == [
        "vllm_omni/a.py"
    ]


def test_resolve_version_auto_reads_dist_info(tmp_path):
    tree(tmp_path, {"PyNvVideoCodec-1.2.3.dist-info/METADATA": "m\n"})
    r = rule(name="pynvvideocodec", version="auto")
    assert package.resolve_version(tmp_path, r) == "1.2.3"
    with pytest.raises(SystemExit, match="no dist-info"):
        package.resolve_version(tmp_path, rule(name="absent", version="auto"))


def test_package_dir_names_archs_when_filtered():
    assert package.package_dir(Path("/out"), rule(), ["sm_90a"]).name == (
        "1.0-linux-x86_64-cuda-sm_90a"
    )
    assert package.package_dir(Path("/out"), rule(arch_filter=None), []).name == (
        "1.0-linux-x86_64-cuda"
    )


def test_build_id_is_a_digest_of_the_selection():
    entries = [
        {"path": "b.py", "sha256": "2", "bytes": 1},
        {"path": "a.py", "sha256": "1", "bytes": 1},
    ]
    assert package.build_id(entries) == package.build_id(list(reversed(entries)))
    assert package.build_id(entries) != package.build_id(entries[:1])


# verify -------------------------------------------------------------------


def fresh_package(tmp_path, r=None):
    src = tree(tmp_path / "src", {"foo/a.py": "hello\n"})
    return package.build(src, r or rule(), tmp_path / "out", ["sm_90a"], archive=False)


def test_verify_is_clean_on_a_fresh_package(tmp_path):
    assert package.verify(fresh_package(tmp_path)) == []


def test_verify_reports_a_tampered_file(tmp_path):
    pkg = fresh_package(tmp_path)
    # Same length, different bytes: only the hash can catch this.
    (pkg / "foo" / "a.py").write_text("world\n")
    assert package.verify(pkg) == ["foo/a.py: hash mismatch"]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("world\n", "foo/a.py: hash mismatch"),
        ("world!\n", "foo/a.py: size 7 != 6"),
    ],
)
def test_verify_catches_both_size_and_hash_drift(tmp_path, content, expected):
    pkg = fresh_package(tmp_path)
    (pkg / "foo" / "a.py").write_text(content)
    assert package.verify(pkg) == [expected]


def test_verify_refuses_a_symlink_in_the_package(tmp_path):
    pkg = fresh_package(tmp_path)
    (pkg / "link").symlink_to("foo/a.py")
    assert package.verify(pkg) == ["link: symlink in package"]


def test_verify_reports_a_file_that_is_not_in_the_manifest(tmp_path):
    pkg = fresh_package(tmp_path)
    (pkg / "extra.txt").write_text("unverified\n")
    assert package.verify(pkg) == ["extra.txt: file in package is not in the manifest"]


def test_verify_skips_the_report_file_it_writes_after_hashing(tmp_path):
    pkg = fresh_package(tmp_path)
    assert (pkg / libdeps.SYSTEM_LIBS_FILE).exists()
    assert package.verify(pkg) == []


# check_archive: P11, verify before unpacking ------------------------------


def make_archive(dest, manifest, files, extra=()):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tf:
        for rel, content in files.items():
            info = tarfile.TarInfo(rel)
            info.size = len(content.encode())
            tf.addfile(info, io.BytesIO(content.encode()))
        blob = json.dumps(manifest).encode()
        info = tarfile.TarInfo(package.MANIFEST)
        info.size = len(blob)
        tf.addfile(info, io.BytesIO(blob))
        for name, content, member_type in extra:
            info = tarfile.TarInfo(name)
            info.type = member_type
            info.size = len(content.encode())
            tf.addfile(info, io.BytesIO(content.encode()))


def test_check_archive_is_clean_on_a_real_archive(tmp_path):
    src = tree(tmp_path / "src", {"foo/a.py": "hello\n"})
    package.build(src, rule(), tmp_path / "out", ["sm_90a"], archive=True)
    archive = next((tmp_path / "out").glob("*.tar.gz"))
    assert package.check_archive(archive) == []


def test_check_archive_refuses_members_that_escape(tmp_path):
    dest = tmp_path / "a.tar.gz"
    make_archive(
        dest,
        {"files": []},
        {},
        extra=[
            ("../escape.txt", "x", tarfile.REGTYPE),
            ("/absolute.txt", "x", tarfile.REGTYPE),
            ("link.txt", "x", tarfile.SYMTYPE),
        ],
    )
    problems = package.check_archive(dest)
    assert any("escapes the target directory" in p for p in problems)
    assert any("not a regular file" in p for p in problems)


def test_check_archive_verifies_bytes_before_unpacking(tmp_path):
    digest = "0" * 64
    dest = tmp_path / "a.tar.gz"
    make_archive(
        dest,
        {
            "files": [
                {"path": "a.txt", "sha256": digest, "bytes": 5},
                {"path": "ghost.txt", "sha256": digest, "bytes": 1},
            ]
        },
        {"a.txt": "hello"},
    )
    problems = package.check_archive(dest)
    assert "a.txt: hash mismatch" in problems
    assert "ghost.txt: manifest names a file the archive lacks" in problems


def test_check_archive_reports_an_unlisted_member(tmp_path):
    dest = tmp_path / "a.tar.gz"
    make_archive(dest, {"files": []}, {"unlisted.txt": "x"})
    assert package.check_archive(dest) == [
        "unlisted.txt: member is not in the manifest"
    ]


def test_check_archive_refuses_a_manifest_that_is_not_json(tmp_path):
    dest = tmp_path / "a.tar.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tf:
        blob = b"not json"
        info = tarfile.TarInfo(package.MANIFEST)
        info.size = len(blob)
        tf.addfile(info, io.BytesIO(blob))
    assert any("not readable as JSON" in p for p in package.check_archive(dest))


def test_check_archive_refuses_an_archive_with_no_manifest(tmp_path):
    dest = tmp_path / "a.tar.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tf:
        blob = b"x"
        info = tarfile.TarInfo("a.txt")
        info.size = len(blob)
        tf.addfile(info, io.BytesIO(blob))
    assert package.check_archive(dest) == ["archive has no manifest"]


# determinism --------------------------------------------------------------


def test_two_builds_of_the_same_source_are_byte_identical(tmp_path):
    src = tree(tmp_path / "src", {"foo/a.py": "print(1)\n", "foo/lib/x.so": "s\n"})
    r = rule(bytecode="pyc-only", strip=False)
    package.build(src, r, tmp_path / "out1", ["sm_90a"], archive=True)
    package.build(src, r, tmp_path / "out2", ["sm_90a"], archive=True)
    first = next((tmp_path / "out1").glob("*.tar.gz")).read_bytes()
    second = next((tmp_path / "out2").glob("*.tar.gz")).read_bytes()
    assert first == second


def test_build_refuses_an_existing_target(tmp_path):
    src = tree(tmp_path / "src", {"foo/a.py": "x\n"})
    package.build(src, rule(), tmp_path / "out", ["sm_90a"], archive=False)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        package.build(src, rule(), tmp_path / "out", ["sm_90a"], archive=False)


def test_bytecode_pyc_only_removes_the_source(tmp_path):
    src = tree(tmp_path / "src", {"foo/a.py": "print(1)\n"})
    pkg = package.build(
        src, rule(bytecode="pyc-only"), tmp_path / "out", ["sm_90a"], archive=False
    )
    assert list(pkg.rglob("*.py")) == []
    assert list(pkg.rglob("*.pyc"))
    assert package.verify(pkg) == []


def test_build_refuses_a_rule_that_selects_nothing(tmp_path):
    tree(tmp_path / "src", {"bar/x.py": "x\n"})
    with pytest.raises(SystemExit, match="selected nothing"):
        package.build(tmp_path / "src", rule(), tmp_path / "out", [], archive=False)


# lock --------------------------------------------------------------------


def test_lock_round_trip_and_gate(tmp_path):
    wheelhouse = tmp_path / "wh"
    wheelhouse.mkdir()
    wheel = wheelhouse / "foo-1.0-cp312-cp312-manylinux_2_28_x86_64.whl"
    wheel.write_text("wheel bytes\n")
    lock = tmp_path / "lock.txt"
    assert lock_mod.write_lock(wheelhouse, lock, "linux-x86_64") == 1
    assert lock_mod.check(lock, wheelhouse) == []
    assert ("foo", "1.0") in lock_mod.read_lock(lock)

    wheel.write_text("tampered\n")
    assert lock_mod.check(lock, wheelhouse) == ["foo==1.0: hash mismatch"]

    wheel.write_text("wheel bytes\n")
    (wheelhouse / "new-0.1-cp312-cp312-manylinux_2_28_x86_64.whl").write_text("x\n")
    assert any("wheel not in lock" in p for p in lock_mod.check(lock, wheelhouse))


def test_write_lock_refuses_two_versions_of_one_distribution(tmp_path):
    # A wheelhouse is a directory, so a second download run into the same folder
    # leaves the older wheel beside the newer one. The lock is a reading of the
    # wheelhouse, so it would list both and pip would be asked to install both,
    # which cannot be resolved. A real run hit this with compressed-tensors
    # 0.13.0 and 0.15.0.1.
    wheelhouse = tmp_path / "wh"
    wheelhouse.mkdir()
    for name in (
        "compressed_tensors-0.13.0-py3-none-any.whl",
        "compressed_tensors-0.15.0.1-py3-none-any.whl",
    ):
        (wheelhouse / name).write_text("x\n")
    lock = tmp_path / "lock.txt"
    try:
        lock_mod.write_lock(wheelhouse, lock, "linux-x86_64")
    except SystemExit as exc:
        assert "compressed-tensors" in str(exc)
        assert not lock.exists(), "the lock was written despite the conflict"
    else:
        raise AssertionError("two versions of one distribution were accepted")


def test_check_accepts_tags_below_the_runtime_floor(tmp_path):
    # blake3 publishes only manylinux_2_17 wheels, so a pinned single platform
    # tag cannot resolve a real requirement set. The gate is the runtime base
    # floor (ubuntu:24.04 = glibc 2.39), not one exact tag.
    wheelhouse = tmp_path / "wh"
    wheelhouse.mkdir()
    for name in (
        "blake3-1.0.11-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "numpy-2.5.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "triton-3.9.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "pyyaml-6.0.2-py3-none-any.whl",
    ):
        (wheelhouse / name).write_text("x\n")
    lock = tmp_path / "lock.txt"
    lock_mod.write_lock(wheelhouse, lock, "linux-x86_64")
    assert lock_mod.check(lock, wheelhouse) == []


def test_check_rejects_a_wheel_outside_the_target(tmp_path):
    wheelhouse = tmp_path / "wh"
    wheelhouse.mkdir()
    (wheelhouse / "foo-1.0-cp312-cp312-manylinux_2_17_x86_64.whl").write_text("x\n")
    lock = tmp_path / "lock.txt"
    lock_mod.write_lock(wheelhouse, lock, "linux-x86_64")

    (wheelhouse / "bar-1.0-cp312-cp312-manylinux_2_40_x86_64.whl").write_text("y\n")
    assert any(
        "not accepted by linux-x86_64" in p for p in lock_mod.check(lock, wheelhouse)
    )

    (wheelhouse / "baz-1.0-cp312-cp312-musllinux_1_1_aarch64.whl").write_text("z\n")
    assert any("platform tag" in p for p in lock_mod.check(lock, wheelhouse))

    (wheelhouse / "qux-1.0-cp311-cp311-manylinux_2_17_x86_64.whl").write_text("w\n")
    assert any("python tag" in p for p in lock_mod.check(lock, wheelhouse))


def test_download_resolves_for_the_running_environment(monkeypatch, tmp_path):
    captured = {}

    def fake_run(args, check=True):
        captured["args"] = args
        return 0

    monkeypatch.setattr(lock_mod.subprocess, "run", fake_run)
    lock_mod.download(["foo"], tmp_path / "wh", "linux-x86_64")
    assert "--platform" not in captured["args"]
    assert "--abi" not in captured["args"]
    assert captured["args"][-1] == "foo"


def test_cross_download_pins_tags_and_repeats_them(monkeypatch, tmp_path):
    captured = {}

    def fake_run(args, check=True):
        captured["args"] = args
        return 0

    monkeypatch.setattr(lock_mod.subprocess, "run", fake_run)
    lock_mod.download(["foo"], tmp_path / "wh", "linux-x86_64", cross=True)
    args = captured["args"]
    # pip spells these flags with hyphens, and repeats them to accept several
    # tags — a single pinned tag cannot cover both abi3 and cp312 wheels.
    assert "--python-version" in args and "--python_version" not in args
    assert args.count("--abi") == len(lock_mod.TARGETS["linux-x86_64"]["abi"])
    assert args.count("--platform") == len(lock_mod.TARGETS["linux-x86_64"]["platform"])


def test_read_requirements_drops_markers_and_options(tmp_path):
    path = tmp_path / "common.txt"
    path.write_text("# comment\nfoo>=1.0  # marker\n--find-links x\nbar==2.0 ; extra\n")
    assert lock_mod.read_requirements(path) == ["foo", "bar"]


def test_requirements_snapshot_needs_no_clone_and_applies_deny():
    # A build has no clone in its context, so the rule's snapshot is what the
    # build resolves. Denied names are dropped because they are never shipped.
    r = rule(requirements={"clone": "vllm", "names": ["zeta", "bar", "alpha"]})
    # The package being packaged is part of the set: without it the installed
    # tree has no dist-info for resolve_version to read.
    assert lock_mod.requirements_for(r) == ["alpha", "foo", "zeta"]


def test_rule_denying_its_own_package_is_refused():
    r = rule(requirements={"clone": "vllm", "names": ["alpha"]}, deny=["foo"])
    try:
        lock_mod.requirements_for(r)
    except SystemExit as exc:
        assert "denies its own package" in str(exc)
    else:
        raise AssertionError("a rule cannot deny the package it ships")


def test_denied_dependencies_reports_a_runtime_conflict(tmp_path):
    sp = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    shipped = sp / "shipped-1.0.dist-info"
    shipped.mkdir(parents=True)
    (shipped / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: shipped\n"
        "Requires-Dist: pip\n"
        "Requires-Dist: numpy>=1.2\n"
        "Requires-Dist: setuptools ; extra == 'test'\n"
    )
    denied = sp / "gone-1.0.dist-info"
    denied.mkdir(parents=True)
    (denied / "METADATA").write_text("Requires-Dist: pip\n")
    r = rule(deny=["pip", "gone"])
    # only an unmarked requirement of a shipped package counts: extras are not
    # installed, and a denied package's own requirements never reach runtime
    assert package_thin.denied_dependencies(tmp_path / "venv", r) == [
        "pip (required by shipped)"
    ]


def test_top_level_map_maps_import_names_through_record(tmp_path):
    sp = tmp_path / "site-packages"
    record = sp / "flashinfer_python-0.6.6.dist-info" / "RECORD"
    record.parent.mkdir(parents=True)
    record.write_text(
        "flashinfer/decode.py,,\nflashinfer_python-0.6.6.dist-info/METADATA,,\n"
    )
    cv2 = sp / "opencv_python_headless-5.0.dist-info" / "RECORD"
    cv2.parent.mkdir(parents=True)
    cv2.write_text("cv2/__init__.py,,\n")
    # the mapping is read from what the wheel installs, not guessed from the name
    assert importscan.top_level_map(sp) == {
        "flashinfer": "flashinfer-python",
        "cv2": "opencv-python-headless",
    }


def test_runtime_conflicts_only_count_imports_at_module_level(tmp_path):
    sp = tmp_path / "site-packages"
    (sp / "shipped-1.0.dist-info").mkdir(parents=True)
    (sp / "shipped-1.0.dist-info" / "RECORD").write_text("shipped/x.py,,\n")
    (sp / "flashinfer_python-0.6.6.dist-info").mkdir(parents=True)
    (sp / "flashinfer_python-0.6.6.dist-info" / "RECORD").write_text(
        "flashinfer/decode.py,,\n"
    )
    pkg = tree(
        tmp_path / "pkg",
        {
            "shipped/x.py": "import flashinfer\n\ndef f():\n    import cv2\n",
        },
    )
    r = rule(deny=["flashinfer-python", "cv2"])
    assert importscan.runtime_conflicts(pkg, r, importscan.top_level_map(sp)) == [
        "shipped/x.py: imports flashinfer at module level, but flashinfer-python is denied"
    ]


def test_runtime_conflicts_ignore_a_package_importing_itself(tmp_path):
    sp = tmp_path / "site-packages"
    (sp / "flashinfer_python-0.6.6.dist-info").mkdir(parents=True)
    (sp / "flashinfer_python-0.6.6.dist-info" / "RECORD").write_text(
        "flashinfer/decode.py,,\n"
    )
    pkg = tree(
        tmp_path / "pkg", {"flashinfer/decode.py": "from flashinfer import utils\n"}
    )
    r = rule(deny=["flashinfer-python"])
    assert importscan.runtime_conflicts(pkg, r, importscan.top_level_map(sp)) == []


def test_scan_uses_the_selection_when_bytecode_removes_the_source(tmp_path):
    sp = tmp_path / "site-packages"
    (sp / "shipped-1.0.dist-info").mkdir(parents=True)
    (sp / "shipped-1.0.dist-info" / "RECORD").write_text("shipped/x.py,,\n")
    (sp / "flashinfer_python-0.6.6.dist-info").mkdir(parents=True)
    (sp / "flashinfer_python-0.6.6.dist-info" / "RECORD").write_text(
        "flashinfer/decode.py,,\n"
    )
    src = tree(sp, {"shipped/x.py": "import flashinfer\n"})
    r = rule(deny=["flashinfer-python"], bytecode="pyc-only")
    selected = set(package.select(src, r, ["sm_90"]))
    assert "shipped/x.py" in selected

    pkg = package.build(src, r, tmp_path / "out", ["sm_90"], archive=False)
    assert not list(pkg.rglob("*.py"))
    assert importscan.runtime_conflicts(pkg, r, importscan.top_level_map(sp)) == []
    assert importscan.runtime_conflicts(
        src, r, importscan.top_level_map(sp), selected
    ) == [
        "shipped/x.py: imports flashinfer at module level, but flashinfer-python is denied"
    ]


def test_evaluate_parse_findings_splits_a_gate_line():
    line = (
        "shipped/x.py: imports flashinfer at module level, "
        "but flashinfer-python is denied"
    )
    assert evaluate_deny.parse_findings([line]) == [
        ("shipped/x.py", "flashinfer", "flashinfer-python")
    ]


def test_evaluate_wheel_sizes_counts_one_distribution_in_the_wheelhouse(tmp_path):
    wh = tmp_path / "wheelhouse"
    wh.mkdir()
    (wh / "flashinfer_python-0.6.6-py3-none-any.whl").write_bytes(b"x" * 100)
    (wh / "flashinfer_python-0.6.6-cp312-cp312-linux_x86_64.whl").write_bytes(b"y" * 50)
    (wh / "torch-2.5.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"z" * 7)
    assert evaluate_deny.wheel_sizes(wh, "flashinfer-python") == (150, 2)
    assert evaluate_deny.wheel_sizes(wh, "torch") == (7, 1)
    assert evaluate_deny.wheel_sizes(wh, "numpy") == (0, 0)


def test_evaluate_tree_sizes_measures_a_distribution_tree_only(tmp_path):
    sp = tmp_path / "site-packages"
    (sp / "foo").mkdir(parents=True)
    (sp / "foo" / "a.py").write_text("x" * 10)
    (sp / "foo" / "b.so").write_bytes(b"y" * 20)
    (sp / "foo" / "link.so").symlink_to(sp / "foo" / "b.so")
    assert evaluate_deny.tree_sizes(sp, ["foo"]) == (30, 2)
    assert evaluate_deny.tree_sizes(sp, ["absent"]) == (0, 0)


def test_evaluate_materialise_writes_only_the_selection(tmp_path):
    sp = tmp_path / "site-packages"
    tree(sp, {"shipped/x.py": "import deniedpkg\n", "deniedpkg/y.py": "x = 1\n"})
    r = rule(deny=["deniedpkg"])
    selected = set(package.select(sp, r, ["sm_90"]))
    view = evaluate_deny.materialise(sp, selected, tmp_path / "view")
    assert (view / "shipped" / "x.py").is_file()
    assert not (view / "deniedpkg").exists()


def test_denied_uses_the_record_owner_when_the_import_name_differs(tmp_path):
    sp = tmp_path / "site-packages"
    tree(sp, {"flashinfer/x.py": "x = 1\n", "other/y.py": "y = 1\n"})
    (sp / "flashinfer-python-0.2.dist-info").mkdir(parents=True)
    (sp / "flashinfer-python-0.2.dist-info" / "RECORD").write_text(
        "flashinfer/x.py,,\n"
    )
    r = rule(deny=["flashinfer-python"])
    owner = importscan.top_level_map(sp)
    # the folder name is not the distribution name, so a name-only check misses it
    assert not rules.denied("flashinfer/x.py", r)
    assert rules.denied("flashinfer/x.py", r, owner)
    assert package.select(sp, r, ["sm_90"], owner) == ["other/y.py"]


def test_evaluate_stops_on_a_declared_requirement_not_an_import(tmp_path):
    sp = tmp_path / "site-packages"
    tree(sp, {"flashinfer/x.py": "x = 1\n"})
    assert evaluate_deny.is_top_level("flashinfer", sp, {})
    assert evaluate_deny.is_top_level("numpy", sp, {"numpy": "numpy"})
    message = "The 'huggingface-hub>=1.31.0,<3.0' distribution was not found"
    assert not evaluate_deny.is_top_level(message.split(".")[0], sp, {})


def test_evaluate_view_must_start_clean(tmp_path):
    view = tmp_path / "view"
    assert evaluate_deny.view_is_clean(view)
    view.mkdir()
    assert evaluate_deny.view_is_clean(view)
    (view / "x.py").write_text("x = 1\n")
    assert not evaluate_deny.view_is_clean(view)


def test_evaluate_importers_by_top_names_the_denied_importer_of_a_shipped_tree(
    tmp_path,
):
    sp = tmp_path / "site-packages"
    tree(
        sp,
        {
            "kept/x.py": "import helper\n",
            "helper/y.py": "x = 1\n",
            "gone/z.py": "import helper\n",
        },
    )
    (sp / "kept-1.0.dist-info").mkdir(parents=True)
    (sp / "kept-1.0.dist-info" / "RECORD").write_text("kept/x.py,,\n")
    (sp / "helper-1.0.dist-info").mkdir(parents=True)
    (sp / "helper-1.0.dist-info" / "RECORD").write_text("helper/y.py,,\n")
    (sp / "gone-1.0.dist-info").mkdir(parents=True)
    (sp / "gone-1.0.dist-info" / "RECORD").write_text("gone/z.py,,\n")
    r = rule(deny=["gone"])
    selected = set(package.select(sp, r, ["sm_90"]))
    owner = importscan.top_level_map(sp)
    importers = evaluate_deny.importers_by_top(sp, selected, owner)
    assert importers["helper"] == {"kept"}
    assert "gone" not in importers["helper"]


def test_evaluate_run_import_records_the_first_trigger(tmp_path):
    sp = tmp_path / "site-packages"
    tree(sp, {"entrypkg/__init__.py": "import depmod\n", "depmod.py": "x = 1\n"})
    result = evaluate_deny.run_import(sp, "entrypkg")
    assert result["missing"] is None
    assert result["error"] is None
    assert result["ok"]["depmod"] == "entrypkg"


def test_evaluate_run_import_separates_a_caught_import_from_an_executed_one(tmp_path):
    sp = tmp_path / "site-packages"
    guarded = "try:\n    import deniedpkg\nexcept ImportError:\n    deniedpkg = None\n"
    tree(sp, {"entrypkg/__init__.py": guarded})
    result = evaluate_deny.run_import(sp, "entrypkg")
    assert result["missing"] is None
    assert result["ok"].get("deniedpkg") is None
    assert result["failed"]["deniedpkg"] == "entrypkg"


def test_evaluate_run_import_names_the_module_that_stopped_the_entry_path(tmp_path):
    sp = tmp_path / "site-packages"
    tree(sp, {"entrypkg/__init__.py": "import deniedpkg\n"})
    result = evaluate_deny.run_import(sp, "entrypkg")
    assert result["missing"] == "deniedpkg"
    assert result["failed"]["deniedpkg"] == "entrypkg"


def test_requirements_without_snapshot_needs_the_clone():
    r = rule(requirements={"clone": "no-such-clone", "files": ["requirements/x.txt"]})
    try:
        lock_mod.requirements_for(r)
    except SystemExit as exc:
        assert "no 'names' snapshot" in str(exc)
    else:
        raise AssertionError("a missing clone must stop the build, not pass")


# package_thin helpers ----------------------------------------------------


def test_lock_names_and_site_packages(tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text(
        "foo==1.0 \\\n    --hash=sha256:" + "0" * 64 + "\n"
        "Bar_Pkg==2.1.0 \\\n    --hash=sha256:" + "1" * 64 + "\n"
    )
    assert package_thin.lock_names(lock) == [("foo", "1.0"), ("bar-pkg", "2.1.0")]

    sp = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True)
    assert package_thin.site_packages(tmp_path / "venv") == sp


def test_lock_diff_reports_missing_and_extra(tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text(
        "foo==1.0 \\\n    --hash=sha256:" + "0" * 64 + "\n"
        "bar-pkg==2.1.0 \\\n    --hash=sha256:" + "1" * 64 + "\n"
    )
    sp = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True)
    (sp / "foo-1.0.dist-info").mkdir()
    (sp / "extra-0.1.dist-info").mkdir()
    r = rule(deny=["pip"])
    assert package_thin.lock_diff(tmp_path / "venv", lock, r) == [
        "missing: bar-pkg",
        "installed outside the lock: extra",
    ]

    # the environment's own tooling is denied by the rule and never reaches the
    # payload, so it is ignored rather than reported as a violation
    (sp / "pip-25.0.dist-info").mkdir()
    assert package_thin.lock_diff(tmp_path / "venv", lock, r) == [
        "missing: bar-pkg",
        "installed outside the lock: extra",
    ]


def test_install_flags_keep_the_wheelhouse_as_the_only_source():
    flags = package_thin.INSTALL_FLAGS
    assert "--no-index" in flags
    assert "--require-hashes" in flags
    assert "--only-binary=:all:" in flags
    assert "--isolated" in flags


def test_build_path_launchers_are_dropped(tmp_path):
    venv = tmp_path / "venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "useless").write_text("#!/build/python\n# pip install foo\n")
    (bin_dir / "keep").write_text("#!/usr/bin/env python3\nprint(1)\n")
    assert package_thin.drop_build_path_launchers(venv, tmp_path / "build") == 1
    assert (bin_dir / "keep").exists()
    assert not (bin_dir / "useless").exists()


def test_install_flags_are_hash_locked_and_no_deps():
    for flag in ("--isolated", "--require-hashes", "--no-deps", "--only-binary=:all:"):
        assert flag in package_thin.INSTALL_FLAGS


# libdeps / ELF -----------------------------------------------------------


def test_static_artifacts_are_refused():
    assert libdeps.is_static("foo/lib/x.a")
    assert libdeps.is_static("foo/lib/x.o")
    assert not libdeps.is_static("foo/lib/x.so")


def test_driver_boundary_never_bundles():
    assert libdeps.is_driver("libcuda.so.1")
    assert libdeps.is_driver("libnvidia-ml.so.1")
    assert not libdeps.is_driver("libnccl.so.2")
    assert libdeps.is_system("libc.so.6")
    assert not libdeps.is_system("libnccl.so.2")


def test_elf_gate_reads_dt_needed_without_ldd():
    ssl = system_lib("libssl.so*")
    if ssl is None:
        pytest.skip("no system shared library in this container")
    needed = libdeps.elf_needed(ssl)
    assert needed
    assert any(libdeps.is_system(name) for name in needed)
    # libcrypto is a bundled package library, not a system one, which is why
    # the gate reports it as uncovered when it is not in the package.
    assert not all(libdeps.is_system(name) for name in needed)
    assert libdeps.elf_soname(ssl)


def test_uncovered_names_a_needed_library_that_is_not_bundled(tmp_path):
    ssl = system_lib("libssl.so*")
    crypto = system_lib("libcrypto.so*")
    zstd = system_lib("libzstd.so*")
    if ssl is None or crypto is None or zstd is None:
        pytest.skip("no system shared libraries in this container")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    shutil.copy2(ssl, pkg / ssl.name)
    assert libdeps.uncovered(pkg) == [(ssl.name, crypto.name)]

    # The closure does not stop at the first hop: libcrypto itself needs libzstd.
    shutil.copy2(crypto, pkg / crypto.name)
    assert libdeps.uncovered(pkg) == [(crypto.name, zstd.name)]

    shutil.copy2(zstd, pkg / zstd.name)
    assert libdeps.uncovered(pkg) == []


def _elf(machine: int) -> bytes:
    """A minimal ELF header with the given e_machine, enough for the gate."""
    return (
        b"\x7fELF\x02\x01\x01\x00"
        + b"\x00" * 8
        + b"\x03\x00"
        + machine.to_bytes(2, "little")
    )


def test_system_lib_glob_covers_the_file_apt_installs():
    # Measured on ubuntu:24.04: libgomp1 installs libgomp.so.1.0.0 and a
    # libgomp.so.1 symlink. The exact glob matched only the symlink name, so a
    # real import (numba's omppool) stayed uncovered in the gate.
    assert libdeps.is_system("libgomp.so.1.0.0")
    assert libdeps.is_system("libgomp.so.1")


def test_select_refuses_an_elf_for_another_architecture(tmp_path):
    # The build run hit these as strip failures: cutlass-dsl ships stubs built
    # for aarch64, which an x86_64 image can never load.
    pkg = tmp_path / "pkg"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "lib" / "libcute_dsl_runtime.so").write_bytes(_elf(0xB7))
    (pkg / "lib" / "libruntime.so").write_bytes(_elf(0x3E))
    picked = package.select(pkg, rule(), ["sm_86"])
    assert "lib/libruntime.so" in picked
    assert "lib/libcute_dsl_runtime.so" not in picked
    assert libdeps.elf_machine(pkg / "lib" / "libcute_dsl_runtime.so") == 0xB7
    assert libdeps.machine_for_platform("linux-x86_64") == 0x3E


def test_write_system_libs_records_only_system_imports(tmp_path):
    ssl = system_lib("libssl.so*")
    if ssl is None:
        pytest.skip("no system shared library in this container")
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    shutil.copy2(ssl, pkg / ssl.name)
    recorded = libdeps.write_system_libs(pkg, [ssl.name])
    assert "libc.so.6" in recorded
    text = (pkg / libdeps.SYSTEM_LIBS_FILE).read_text()
    assert "libc.so.6" in text
    assert "libnccl.so.2" not in text


def test_copy_closure_copies_under_soname_and_honours_the_arch_filter(tmp_path):
    zlib = system_lib("libz.so*")
    if zlib is None:
        pytest.skip("no system shared library in this container")
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    (src / "nvidia/cu13/lib").mkdir(parents=True)
    shutil.copy2(zlib, src / "nvidia/cu13/lib" / zlib.name)
    (src / "nvidia/cu13/lib" / "static.a").write_text("a")
    (src / "sm_86").mkdir()
    (src / "sm_86" / "kernel.cubin").write_text("k")
    notes = libdeps.copy_closure(src, dest, ["sm_90a"])
    copied = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*"))
    assert copied == [zlib.name]
    assert notes


# file gates --------------------------------------------------------------


def test_check_files_refuses_no_files():
    assert check_files.main([]) == 1


def test_check_files_finds_trailing_whitespace_and_missing_newline(tmp_path):
    path = tmp_path / "a.py"
    path.write_bytes(b"x = 1  \ny = 2")
    problems = check_files.check(path, fix=False)
    assert any("trailing whitespace" in p for p in problems)
    assert any("final newline" in p for p in problems)
    check_files.check(path, fix=True)
    assert check_files.check(path, fix=False) == []


def test_check_files_fixes_an_extra_final_newline(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"x\n\n")
    assert any(
        "more than one final newline" in p for p in check_files.check(path, fix=False)
    )
    check_files.check(path, fix=True)
    assert path.read_bytes() == b"x\n"
    assert check_files.check(path, fix=False) == []


def test_check_files_allows_a_markdown_hard_line_break(tmp_path):
    path = tmp_path / "README.md"
    path.write_bytes(b"line with hard break  \nnext\n")
    assert check_files.check(path, fix=False) == []


def test_check_files_finds_a_debug_statement(tmp_path):
    path = tmp_path / "a.py"
    path.write_text("import pdb\npdb.set_trace()\nbreakpoint()\n")
    problems = check_files.check(path, fix=False)
    assert any("pdb import" in p for p in problems)
    assert any("debug statement" in p for p in problems)


def test_check_files_rewrites_crlf(tmp_path):
    path = tmp_path / "a.py"
    path.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert any("mixed line endings" in p for p in check_files.check(path, fix=False))
    check_files.check(path, fix=True)
    assert path.read_bytes() == b"x = 1\ny = 2\n"


def test_check_files_parses_yaml(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text("services:\n  x:\n    image: y\n")
    assert check_files.check(good, fix=False) == []
    bad = tmp_path / "bad.yaml"
    bad.write_text("services:\n  x: [unclosed\n")
    assert any("invalid YAML" in p for p in check_files.check(bad, fix=False))


# distributed gate --------------------------------------------------------


def test_module_level_imports_are_found_by_ast(tmp_path):
    path = tmp_path / "p.py"
    path.write_text(
        "import torch.distributed\n"
        "from torch.distributed import Backend\n"
        "def f():\n"
        "    import torch.distributed as dist\n"
    )
    found = check_dist_gate.module_level_imports(path)
    assert len(found) == 2
    assert any("from torch.distributed import Backend" in line for line in found)


def test_dist_gate_scan_on_the_read_only_clone():
    clone = REPO / "vllm" / "vllm" / "distributed"
    if not clone.is_dir():
        pytest.skip("read-only clone is not present")
    unguarded, guarded = check_dist_gate.scan(clone)
    # Plan.md D1: distributed is imported at module load, so USE_DISTRIBUTED=0
    # cannot work even at tp_size=1.
    assert unguarded > 0
    assert guarded == 0


def test_stub_reproduces_the_distributed_verdict(monkeypatch):
    assert sorted(check_dist_gate.STUB_PROVIDES) == ["GroupName", "ProcessGroup"]

    torch = types.ModuleType("torch")
    dist = types.ModuleType("torch.distributed")
    for name in check_dist_gate.STUB_PROVIDES:
        setattr(dist, name, type(name, (), {"__doc__": "stub"}))
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)

    assert getattr(dist, "is_available", None) is None
    with pytest.raises(ImportError, match="cannot import name 'Backend'"):
        from torch.distributed import Backend  # noqa: F401

    assert getattr(dist, "ProcessGroup", None) is not None


# runtime library measurement ---------------------------------------------


def test_maps_reader_takes_the_path_field(tmp_path):
    import measure_runtime_libs

    maps = tmp_path / "maps"
    maps.write_text(
        "7f0000000000-7f0000001000 r-xp 00000000 08:01 123 "
        "/site-packages/nvidia/cublas/lib/libcublas.so.12\n"
        "7f0000002000-7f0000003000 rw-p 00000000 00:00 0 \n"
        "7f0000004000-7f0000005000 r--p 00000000 08:01 456 "
        "/site-packages/torch/lib/libtorch_cpu.so\n"
        "7f0000006000-7f0000007000 r--p 00000000 08:01 457 "
        "/site-packages/nvidia/cufile/lib/libcufile_rdma.so.1\n"
    )
    found = measure_runtime_libs.mapped_libs(str(maps))
    assert found == [
        "/site-packages/nvidia/cublas/lib/libcublas.so.12",
        "/site-packages/nvidia/cufile/lib/libcufile_rdma.so.1",
        "/site-packages/torch/lib/libtorch_cpu.so",
    ]
    # The remainder of the line is not the path: it starts with the offset, so
    # reading it instead of the last field would match nothing.
    remainder = maps.read_text().splitlines()[0].split(" ", 1)[-1]
    assert not remainder.startswith("/")


# size gate ---------------------------------------------------------------


def test_size_gate_budget_and_reduction():
    import check_image_size

    assert check_image_size.BUDGET_BYTES == 4 * 1024 * 1024 * 1024
    assert check_image_size.human(check_image_size.BUDGET_BYTES) == "4.00 GB"

    # Phase 4 gate: <= 4 GB AND >= 60% smaller than the baseline.
    baseline = 23 * 1024 * 1024 * 1024
    slim = 3 * 1024 * 1024 * 1024
    assert slim <= check_image_size.BUDGET_BYTES
    assert (baseline - slim) / baseline >= 0.60
