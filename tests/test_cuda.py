"""The CUDA series is decided by commands, not by a default in a Dockerfile.

Runs inside the tools container: bash docker/run.sh test

These tests cover the reading and the decision. The values themselves come from
runs on the machine, and are recorded in `out/cuda.env` — never asserted here as
if a test could know what a given host has.
"""

import shutil
import subprocess

import pytest
from conftest import REPO, tree

import cuda
import rules

import package


def wheelhouse(root, names):
    for name in names:
        (root / name).write_text("x")
    return root


def test_wheelhouse_series_is_the_series_the_wheels_resolve_to(tmp_path):
    wheelhouse(
        tmp_path,
        [
            "nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl",
            "nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl",
            "nvidia_cutlass_dsl_libs_cu13-0.1.0-py3-none-any.whl",
        ],
    )
    series, how = cuda.wheelhouse_series(tmp_path)
    assert series == 12
    assert "2 of 3" in how


def test_wheelhouse_series_reports_nothing_when_no_wheels_are_cu_tagged(tmp_path):
    wheelhouse(tmp_path, ["pytest-8.3.3-py3-none-any.whl"])
    series, how = cuda.wheelhouse_series(tmp_path)
    assert series is None
    assert "no cu-tagged wheels" in how


def test_driver_ceiling_comes_from_nvidia_smi(monkeypatch):
    table = (
        "NVIDIA Driver Version: 580.95.05\n"
        "| NVIDIA-SMI 580.95.05   Driver Version: 580.95.05    CUDA Version: 13.0  |\n"
    )
    monkeypatch.setattr(cuda, "run", lambda argv, timeout=20: table)
    assert cuda.driver_version() == ("580.95.05", "nvidia-smi")
    assert cuda.driver_ceiling() == (13, 0, "nvidia-smi")


def test_detection_records_that_a_tool_is_not_available(monkeypatch):
    monkeypatch.setattr(cuda, "run", lambda argv, timeout=20: "")
    assert cuda.driver_ceiling() == (None, None, "no driver visible")
    monkeypatch.delenv("VDE_CUDA", raising=False)
    monkeypatch.setattr(cuda.shutil, "which", lambda name: None)
    assert cuda.installed_cuda() == (None, None, "no nvcc on PATH")


def test_override_wins_and_is_recorded_as_an_override(monkeypatch):
    monkeypatch.setenv("VDE_CUDA", "13.1")
    monkeypatch.setattr(cuda, "run", lambda argv, timeout=20: "")
    assert cuda.installed_cuda() == (13, 1, "VDE_CUDA")


def test_nvcc_release_line_is_read_as_the_installed_toolchain(monkeypatch):
    monkeypatch.delenv("VDE_CUDA", raising=False)
    monkeypatch.setattr(cuda.shutil, "which", lambda name: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr(
        cuda,
        "run",
        lambda argv, timeout=20: "Cuda compilation tools, release 12.8, V12.8.93\n",
    )
    assert cuda.installed_cuda() == (12, 8, "/usr/local/cuda/bin/nvcc --version")


def test_wheelhouse_decides_when_the_driver_also_accepts_it(monkeypatch):
    monkeypatch.setattr(cuda, "run", lambda argv, timeout=20: "")
    series, how = cuda.select_series(12, 13, 13)
    assert series == 12
    assert "wheelhouse wins" in how


def test_a_series_the_driver_refuses_is_refused_before_the_image_is_generated():
    # Measured on this host: driver 580.95.05 accepts CUDA 13.0 and refuses
    # `cuda>=13.1` at container start, so a cu13.1 image cannot be generated for
    # a machine whose driver ceiling is 13.0.
    with pytest.raises(SystemExit) as caught:
        cuda.select_series(13, 13, 12)
    assert "cannot be used due to driver issues" in str(caught.value)
    assert "update the driver" in str(caught.value)
    assert "would not start" in str(caught.value)


def test_a_value_the_user_sets_wins_over_detection(monkeypatch):
    # The wheelhouse resolves cu12 and the driver ceiling is 13, but the user
    # asked for 13: the user value is what the image declares.
    monkeypatch.setenv("VDE_CUDA", "13.1")
    monkeypatch.setattr(cuda, "run", lambda argv, timeout=20: "")
    series, how = cuda.select_series(12, 13, 13, 13, "580.95.05")
    assert series == 13
    assert "set by the user" in how
    assert "wheelhouse resolves cu12" in how


def test_a_user_value_below_the_ceiling_does_not_need_the_wheelhouse_warning():
    series, how = cuda.select_series(12, 13, 13, 12, "580.95.05")
    assert series == 12
    assert "set by the user" in how
    assert "wheelhouse resolves" not in how


def test_a_user_value_the_driver_cannot_run_raises_an_error_naming_the_driver():
    # Measured on this host: driver 580.95.05 refuses `cuda>=13.1`, so a user
    # override of 13.1 is refused here, with the driver named as the cause.
    with pytest.raises(SystemExit) as caught:
        cuda.select_series(12, 13, 13, 14, "580.95.05")
    message = str(caught.value)
    assert "cannot be used due to driver issues" in message
    assert "580.95.05" in message
    assert "different CUDA version or update the driver" in message


def test_user_override_reads_the_value_and_records_where_it_came_from(monkeypatch):
    monkeypatch.delenv("VDE_CUDA", raising=False)
    assert cuda.user_override() == (None, None, "not set by the user")
    assert cuda.user_override("13.1") == (13, 1, "set by the user (13.1)")
    monkeypatch.setenv("VDE_CUDA", "12.8")
    assert cuda.user_override()[0] == 12
    assert "set by the user" in cuda.user_override()[2]
    with pytest.raises(SystemExit):
        cuda.user_override("not-a-version")


def test_decide_records_the_user_value_as_a_separate_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(cuda, "run", lambda argv, timeout=20: "")
    monkeypatch.delenv("VDE_CUDA", raising=False)
    wheelhouse(tmp_path, ["nvidia_cublas_cu12-1.0-py3-none-any.whl"])
    values = cuda.decide(tmp_path, None, "13.0")
    assert values["VDE_CUDA_USER"] == "13"
    assert "set by the user (13.0)" in values["VDE_CUDA_USER_HOW"]
    assert values["VDE_CUDA_SERIES"] == "13"
    # With no user value the wheelhouse decides again, as it did before.
    values = cuda.decide(tmp_path, None)
    assert values["VDE_CUDA_USER"] == ""
    assert values["VDE_CUDA_SERIES"] == "12"


def test_a_registry_401_is_reported_as_needing_user_support(monkeypatch):
    # Earlier Docker Hub pulls failed with 401, so a refused tag that came from a
    # 401 is reported differently from a tag the registry simply does not have.
    monkeypatch.setattr(
        cuda.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 1, "", "requested pull returned 401 Unauthorized"
        ),
    )
    how = cuda.verify_tag("nvidia/cuda:13.1.0-devel-ubuntu24.04")
    assert how.startswith("UNVERIFIED")
    assert "401" in how
    assert "user support asked for" in how


def test_a_tag_the_registry_refuses_for_another_reason_stays_a_plain_refusal(
    monkeypatch,
):
    monkeypatch.setattr(
        cuda.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 1, "", "manifest unknown"
        ),
    )
    how = cuda.verify_tag("nvidia/cuda:12.4.1-devel-ubuntu24.04")
    assert how == "UNVERIFIED: registry refused nvidia/cuda:12.4.1-devel-ubuntu24.04"


def test_tree_series_reads_the_cutlass_folders_the_environment_resolves(tmp_path):
    tree(tmp_path, {"nvidia_cutlass_dsl/cu12/lib/x.so": "x"})
    (tmp_path / "nvidia_cutlass_dsl" / "cu13").mkdir()
    series, how = cuda.tree_series(tmp_path)
    assert series == 13
    assert "cu12" in how and "cu13" in how


def test_resolve_prefers_an_explicit_value_then_the_variables_file(monkeypatch):
    monkeypatch.setenv("VDE_CUDA_SERIES", "12")
    monkeypatch.delenv("VDE_CUDA_KEEP", raising=False)
    assert cuda.resolve("13.1") == 13
    assert cuda.resolve("auto") == 12
    monkeypatch.delenv("VDE_CUDA_SERIES")
    assert cuda.resolve("auto") is None
    with pytest.raises(SystemExit):
        cuda.resolve("not-a-series")


def test_the_surviving_copy_is_a_separate_value_from_the_declared_series(monkeypatch):
    # Measured on this machine: the wheelhouse resolves cu12, the environment resolves
    # cu13, so the declared series and the copy that survives are different numbers.
    monkeypatch.setenv("VDE_CUDA_SERIES", "12")
    monkeypatch.setenv("VDE_CUDA_KEEP", "13")
    assert cuda.resolve("auto") == 13
    monkeypatch.delenv("VDE_CUDA_KEEP")
    assert cuda.resolve("auto") == 12
    monkeypatch.setenv("VDE_CUDA_KEEP", "12")
    assert cuda.resolve("auto") == 12
    monkeypatch.delenv("VDE_CUDA_KEEP")
    monkeypatch.delenv("VDE_CUDA_SERIES")


def test_the_duplicate_cutlass_folder_measured_on_this_machine_prunes_the_dead_copy():
    # Both copies are present in the tree, each 41.6 MB, and only one is resolved.
    rule = rules.load("vllm")
    dead = "nvidia_cutlass_dsl/cu12/lib/libcute_dsl_runtime.so"
    live = "nvidia_cutlass_dsl/cu13/lib/libcute_dsl_runtime.so"
    assert rules.cuda_variant(dead, rule, 13)
    assert not rules.cuda_variant(live, rule, 13)


def test_no_series_means_no_prune_decision(monkeypatch):
    monkeypatch.delenv("VDE_CUDA_KEEP", raising=False)
    monkeypatch.delenv("VDE_CUDA_SERIES", raising=False)
    rule = rules.load("vllm")
    rel = "nvidia_cutlass_dsl/cu12/lib/x.so"
    assert not rules.cuda_variant(rel, rule, None)
    assert not rules.cuda_variant(rel, dict(rule, cuda_variant=None), 12)
    # A folder that is not a per-series copy is not touched by this rule.
    assert not rules.cuda_variant("nvidia_cutlass_dsl/dsl.py", rule, 12)


def test_resolve_falls_back_to_the_wheelhouse(tmp_path, monkeypatch):
    monkeypatch.delenv("VDE_CUDA_SERIES", raising=False)
    wheelhouse(tmp_path, ["nvidia_cublas_cu12-1.0-py3-none-any.whl"])
    assert cuda.resolve("auto", tmp_path) == 12


def test_tag_for_a_series_that_was_never_confirmed_is_reported_as_unverified():
    tag, how = cuda.tag_for(14, 0)
    assert tag == "nvidia/cuda:14.0.0-devel-ubuntu24.04"
    assert how.startswith("UNVERIFIED")
    tag, how = cuda.tag_for(12, 8)
    assert tag == "nvidia/cuda:12.8.1-devel-ubuntu24.04"
    assert "verified" in how


def test_a_detected_minor_that_the_registry_does_not_have_falls_back_to_the_confirmed_tag():
    # Measured: nvidia/cuda:12.0.0-devel-ubuntu24.04 is refused by the registry, so
    # a tag built from the detected minor alone is a build that cannot pull. The
    # confirmed tag is used and the difference is recorded rather than hidden.
    tag, how = cuda.tag_for(12, 0)
    assert tag == "nvidia/cuda:12.8.1-devel-ubuntu24.04"
    assert "detected 12.0" in how
    assert not how.startswith("UNVERIFIED")


def test_variables_file_round_trip(tmp_path):
    dest = tmp_path / "out" / "cuda.env"
    cuda.write_env(dest, {"VDE_CUDA_SERIES": "12", "VDE_DRIVER_VERSION": "580.95.05"})
    assert cuda.read_env(dest) == {
        "VDE_CUDA_SERIES": "12",
        "VDE_DRIVER_VERSION": "580.95.05",
    }
    assert cuda.read_env(tmp_path / "missing.env") == {}


def test_a_value_with_spaces_is_quoted_so_sourcing_does_not_execute_it(tmp_path):
    # Measured failure: the unquoted line
    #   VDE_INSTALLED_CUDA_HOW=no nvcc on PATH
    # is not an assignment. The shell reads `VDE_INSTALLED_CUDA_HOW=no` as an
    # environment prefix and then runs `nvcc on PATH` as a command, which is what
    # made the gate run fail with "nvcc fatal: Don't know what to do with 'PATH'".
    dest = tmp_path / "cuda.env"
    values = {
        "VDE_INSTALLED_CUDA_HOW": "no nvcc on PATH",
        "VDE_TREE_SERIES_HOW": "nvidia_cutlass_dsl/['cu12', 'cu13']",
    }
    cuda.write_env(dest, values)
    assert (
        dest.read_text().splitlines()[0] == "VDE_INSTALLED_CUDA_HOW='no nvcc on PATH'"
    )
    assert all(
        line.split("=", 1)[1].startswith("'") for line in dest.read_text().splitlines()
    )
    assert cuda.read_env(dest) == values


def test_the_variables_file_is_valid_shell(tmp_path):
    """The build sources this file, so the only honest test is to source it."""
    dest = tmp_path / "cuda.env"
    values = {
        "VDE_CUDA_SERIES": "12",
        "VDE_CUDA_KEEP": "13",
        "VDE_INSTALLED_CUDA_HOW": "no nvcc on PATH",
        "VDE_TREE_SERIES_HOW": "nvidia_cutlass_dsl/['cu12', 'cu13']",
    }
    cuda.write_env(dest, values)
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash available to source the file")
    done = subprocess.run(
        [
            bash,
            "-c",
            f'. {dest} && printf \'%s|%s\\n\' "$VDE_CUDA_KEEP" "$VDE_INSTALLED_CUDA_HOW"',
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0
    assert done.stdout.strip() == "13|no nvcc on PATH"


def test_rule_prunes_the_copy_of_the_other_series():
    rule = rules.load(REPO / "packaging" / "rules" / "vllm.json")
    live = "nvidia_cutlass_dsl/cu12/lib/libcute_dsl_runtime.so"
    dead = "nvidia_cutlass_dsl/cu13/lib/libcute_dsl_runtime.so"
    assert not rules.cuda_variant(live, rule, 12)
    assert rules.cuda_variant(dead, rule, 12)
    # Without a decided series nothing is pruned: the choice has to come from a
    # measurement, not from a pattern that guesses.
    assert not rules.cuda_variant(dead, rule, None)


def test_selection_prunes_the_dead_copy(tmp_path):
    rule = rules.load(REPO / "packaging" / "rules" / "vllm.json")
    tree(
        tmp_path,
        {
            "nvidia_cutlass_dsl/cu12/lib/x.so": "x",
            "nvidia_cutlass_dsl/cu13/lib/y.so": "y",
            "vllm/engine.py": "x",
        },
    )
    assert package.select(tmp_path, rule, [], None, 12) == [
        "nvidia_cutlass_dsl/cu12/lib/x.so",
        "vllm/engine.py",
    ]
    # A rule that does not name the folder is unaffected by the series.
    plain = rules.load(REPO / "packaging" / "rules" / "vllm-omni.json")
    assert "cuda_variant" not in plain
    assert not rules.cuda_variant("vllm_omni/engine.py", plain, 12)


def test_the_variables_file_writes_the_names_the_build_reads(tmp_path, monkeypatch):
    """The writer and the reader must agree on the names, or the file is decoration.

    This is the whole point of the variables file: `cuda` writes it from commands,
    and the build reads it. A name written that nothing reads, or a name read that
    nothing writes, means the decision never reaches the image.
    """
    monkeypatch.setattr(cuda, "run", lambda argv, timeout=20: "")
    for name in [
        "VDE_CUDA",
        "VDE_CUDA_KEEP",
        "VDE_DRIVER_VERSION",
        "VDE_DRIVER_CUDA",
        "VDE_INSTALLED_CUDA",
        "VDE_WHEELHOUSE_SERIES",
        "VDE_TREE_SERIES",
        "VDE_CUDA_USER",
        "VDE_CUDA_SERIES",
        "VDE_CUDA_BUILDER",
        "VDE_RUNTIME_BASE",
    ]:
        monkeypatch.delenv(name, raising=False)

    wh = wheelhouse(tmp_path, ["nvidia_cublas_cu12-1.0-py3-none-any.whl"])
    tree(tmp_path / "sp", {"nvidia_cutlass_dsl/cu12/lib/x.so": "x"})
    (tmp_path / "sp" / "nvidia_cutlass_dsl" / "cu13").mkdir()

    values = cuda.decide(wh, tmp_path / "sp")
    # Every name resolve() looks up has to be in the file.
    for name in ["VDE_CUDA_SERIES", "VDE_CUDA_KEEP"]:
        assert name in values
    # And every value in the file is a name a consumer can read back.
    dest = tmp_path / "out" / "cuda.env"
    cuda.write_env(dest, values)
    back = cuda.read_env(dest)
    assert back == values
    assert back["VDE_WHEELHOUSE_SERIES"] == "12"
    assert back["VDE_TREE_SERIES"] == "13"
    # The build reads the file, so the value it reads is the value that was decided.
    monkeypatch.setenv("VDE_CUDA_KEEP", back["VDE_CUDA_KEEP"])
    assert cuda.resolve("auto") == 13
    monkeypatch.delenv("VDE_CUDA_KEEP")
    monkeypatch.setenv("VDE_CUDA_SERIES", back["VDE_CUDA_SERIES"])
    assert cuda.resolve("auto") == 12
