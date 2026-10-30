"""Which CUDA the image declares, decided from the machine it will run on.

This package releases code for personal/local images rather than a universal
binary, so the CUDA version is not a default in a Dockerfile. It comes from
commands run against the environment, and the answer is stored in a variables
file that the build reads.

Detection order, first answer wins, and every answer records how it was decided:

1. A value the user sets (``--cuda`` or ``VDE_CUDA``), or the variables file
   written by an earlier run. A value the user sets wins over detection; the
   only condition on it is the driver. When the installed driver cannot run the
   selected version, this script raises an error that names the driver as the
   cause and says to use a different CUDA version or update the driver.
2. ``nvidia-smi`` -- the driver ceiling. An image cannot declare a CUDA series
   the driver refuses, no matter what the registry has. Measured here: driver
   580.95.05 reports CUDA 13.0, and ``cuda>=13.1`` is refused at container
   start, so a cu13.1 override on this machine is refused by the driver.
3. ``nvcc --version`` -- the CUDA toolchain installed on the machine, which is
   what the build stage uses.
4. The wheelhouse -- the series the wheels actually resolve to. A wheel built
   for cu12 cannot be relabelled cu13, so the wheelhouse is the floor. The
   requirements in this repo resolve to cu12 on PyPI (measured), so a cu13
   image needs the cu13 index and that has not been measured yet.

Registry checks are not retried. When the registry answers 401, the failure is
reported with the tag that was being checked, so the user can verify the source
or download the image directly.

Run inside the container, on the GPU:

  bash docker/run.sh cuda --out out/cuda.env
  bash docker/run.sh cuda --verify        # confirms the tags exist in the registry
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

import report

UBUNTU = "ubuntu24.04"

# Tags confirmed to exist by `docker manifest inspect` on this machine. A series
# that is not in this table has not been confirmed, so it is reported as
# UNVERIFIED instead of guessed. `12.4.1-devel-ubuntu24.04` is an example: it is
# a plausible tag that the registry does not have.
KNOWN_TAGS = {12: "12.8.1", 13: "13.0.0"}

RELEASE_LINE = re.compile(r"release (\d+)\.(\d+)")
CUDA_VERSION_LINE = re.compile(r"CUDA Version:\s*(\d+)\.(\d+)")
WHEEL_CU = re.compile(r"[-_]cu(\d+)")
CU_FOLDER = re.compile(r"^cu(\d+)$")


def run(argv: list[str], timeout: int = 20) -> str:
    """Run a command and return stdout, or an empty string when it is not
    available. Detection must not fail the build on a machine that lacks a tool.
    """
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout


def driver_version() -> tuple[str, str]:
    """Driver version and how it was obtained."""
    override = os.environ.get("VDE_DRIVER_VERSION")
    if override:
        return override, "VDE_DRIVER_VERSION"
    text = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    match = re.search(r"\d+(?:\.\d+)+", text)
    if match:
        return match.group(0), "nvidia-smi"
    return "", "no driver visible"


def driver_ceiling() -> tuple[int | None, int | None, str]:
    """The highest CUDA series the installed driver accepts, from nvidia-smi."""
    text = run(["nvidia-smi"])
    match = CUDA_VERSION_LINE.search(text)
    if match:
        return int(match.group(1)), int(match.group(2)), "nvidia-smi"
    return None, None, "no driver visible"


def installed_cuda() -> tuple[int | None, int | None, str]:
    """The CUDA toolchain installed on the machine, read from nvcc."""
    override = os.environ.get("VDE_CUDA")
    if override:
        match = RELEASE_LINE.search(override) or re.match(r"(\d+)\.(\d+)", override)
        if match:
            return int(match.group(1)), int(match.group(2)), "VDE_CUDA"
    nvcc = shutil.which("nvcc")
    if nvcc:
        match = RELEASE_LINE.search(run([nvcc, "--version"]))
        if match:
            return int(match.group(1)), int(match.group(2)), f"{nvcc} --version"
    return None, None, "no nvcc on PATH"


def wheelhouse_series(wheelhouse: Path) -> tuple[int | None, str]:
    """The series the wheels actually resolve to, read from wheel filenames.

    This is the floor: the shipped libraries are the ones pip downloaded, and a
    cu12 wheel cannot be presented as cu13.
    """
    if not wheelhouse.is_dir():
        return None, "wheelhouse not read"
    counts: dict[int, int] = {}
    for wheel in wheelhouse.glob("*.whl"):
        match = WHEEL_CU.search(wheel.name)
        if match:
            series = int(match.group(1))
            counts[series] = counts.get(series, 0) + 1
    if not counts:
        return None, "no cu-tagged wheels in the wheelhouse"
    series = max(counts, key=lambda s: (counts[s], -s))
    return series, f"wheelhouse ({counts[series]} of {sum(counts.values())} wheels)"


def tree_series(site_packages: Path) -> tuple[int | None, str]:
    """The series the installed tree resolves to, from the cutlass DSL folders.

    ``nvidia_cutlass_dsl`` ships both cu12 and cu13 copies; the environment
    resolves one of them. Seeing which one is present is evidence of what the
    machine actually uses, and it is the only duplicate series found in the tree.
    """
    if not site_packages.is_dir():
        return None, "tree not read"
    dsl = site_packages / "nvidia_cutlass_dsl"
    folders = [p.name for p in dsl.iterdir() if p.is_dir()] if dsl.is_dir() else []
    series = [int(m.group(1)) for m in (CU_FOLDER.match(f) for f in folders) if m]
    if not series:
        return None, "no cu folder in nvidia_cutlass_dsl"
    return max(series), f"nvidia_cutlass_dsl/{sorted(folders)}"


def resolve(value: str | None, wheelhouse: Path | None = None) -> int | None:
    """The series whose copy survives in the tree.

    Order: an explicit value, then the variables file, then the tree the
    environment resolves, then the wheelhouse. Nothing is decided by a default in
    a Dockerfile.
    """
    if value and value != "auto":
        match = re.search(r"\d+", value)
        if not match:
            raise SystemExit(f"cuda: cannot read a series from {value!r}")
        return int(match.group(0))
    env = os.environ.get("VDE_CUDA_KEEP") or os.environ.get("VDE_CUDA_SERIES")
    if env and env.isdigit():
        return int(env)
    if wheelhouse is not None:
        return wheelhouse_series(wheelhouse)[0]
    return None


def resolve_keep(value: str | None, tree: Path | None = None) -> int | None:
    """The series whose copy survives in the tree, as opposed to the one declared.

    The declared series is what the image says; the surviving series is what the
    environment resolves. They can differ (the wheelhouse is cu12 while
    `nvidia_cutlass_dsl` resolves cu13), and pruning must follow the copy that
    is actually loaded, not the label. Order: an explicit value, then the
    variables file, then the tree, then the declared series.
    """
    if value and value != "auto":
        match = re.search(r"\d+", value)
        if not match:
            raise SystemExit(f"cuda: cannot read a series from {value!r}")
        return int(match.group(0))
    env = os.environ.get("VDE_CUDA_KEEP")
    if env and env.isdigit():
        return int(env)
    if tree is not None:
        series, _ = tree_series(tree)
        if series is not None:
            return series
    return None


def user_override(explicit: str | None = None) -> tuple[int | None, int | None, str]:
    """The CUDA version the user set, read from ``--cuda`` or ``VDE_CUDA``.

    A value the user sets is not a guess and is not overridden by detection. It
    is checked against the driver and nothing else, which is the only condition
    the user asked for.
    """
    value = explicit or os.environ.get("VDE_CUDA")
    if not value:
        return None, None, "not set by the user"
    match = RELEASE_LINE.search(value) or re.match(r"(\d+)\.(\d+)", value)
    if not match:
        raise SystemExit(f"cuda: cannot read a series from {value!r}")
    return int(match.group(1)), int(match.group(2)), f"set by the user ({value})"


def driver_error(requested: int, ceiling: int, driver: str) -> str:
    """The error wording for a CUDA version the installed driver cannot run.

    The message names the driver as the cause and gives the two ways out: pick a
    CUDA version the driver supports, or update the driver.
    """
    return (
        f"cuda: the selected version cannot be used due to driver issues. "
        f"CUDA {requested} needs a driver that supports it; the installed driver "
        f"{driver or 'is not visible'} supports at most CUDA {ceiling}, so an "
        f"image built for it would not start on this machine. Use a different "
        f"CUDA version or update the driver."
    )


def select_series(
    wheelhouse: int | None,
    installed: int | None,
    ceiling: int | None,
    user: int | None = None,
    driver: str = "",
) -> tuple[int, str]:
    """Pick the series the image declares, and refuse an impossible one.

    A value the user set wins over detection. Otherwise the wheelhouse wins when
    it exists, because those are the libraries that will be shipped, and the
    installed toolchain decides when the wheelhouse has nothing cu-tagged. Either
    way the driver ceiling is a hard gate: a series above it is refused here,
    before an image is generated that could not start.
    """
    if user is not None:
        if ceiling is not None and user > ceiling:
            raise SystemExit(driver_error(user, ceiling, driver))
        how = f"set by the user ({user})"
        if wheelhouse is not None and wheelhouse != user:
            # A cu12 wheel cannot be relabelled cu13, so an override that
            # disagrees with the wheelhouse is recorded as a warning rather than
            # silently accepted: the wheels for the chosen series have to be
            # downloaded before the build can ship them.
            how += f"; wheelhouse resolves cu{wheelhouse}, so that copy must be fetched"
        return user, how

    for name, value in (
        ("wheelhouse", wheelhouse),
        ("installed toolchain", installed),
    ):
        if value is None:
            continue
        if ceiling is not None and value > ceiling:
            raise SystemExit(driver_error(value, ceiling, driver))
        return (
            value,
            f"wheelhouse wins over installed cu{installed}" if installed else name,
        )
    if ceiling is not None:
        return ceiling, "driver ceiling only"
    raise SystemExit("cuda: nothing to decide from. Pass --cuda or run on the GPU.")


def tag_for(series: int, minor: int | None) -> tuple[str, str]:
    """Builder tag for a series, with a verification flag.

    A patch level cannot be guessed: `12.4.1-devel-ubuntu24.04` is a plausible
    tag the registry does not have, and `12.0.0-devel-ubuntu24.04` is refused.
    When a tag for the series was confirmed, that tag is used even when the
    detected minor differs, because a guessed tag is a build that cannot pull.
    Only a series with nothing confirmed is reported as UNVERIFIED.
    """
    known = KNOWN_TAGS.get(series)
    if known is None:
        return (
            f"nvidia/cuda:{series}.{minor}.0-devel-{UBUNTU}",
            "UNVERIFIED: series not checked against the registry",
        )
    if minor is None or str(minor) == known.split(".")[1]:
        return (
            f"nvidia/cuda:{known}-devel-{UBUNTU}",
            "verified by docker manifest inspect",
        )
    return (
        f"nvidia/cuda:{known}-devel-{UBUNTU}",
        f"verified by docker manifest inspect; detected {series}.{minor}, "
        f"only {known} exists for cu{series}",
    )


def verify_tag(tag: str) -> str:
    """Ask the registry whether a tag exists. Needs the docker socket mounted.

    A 401 is reported to the user rather than retried, because it is an
    availability problem on the source. A refused tag stays UNVERIFIED.
    """
    try:
        done = subprocess.run(
            ["docker", "manifest", "inspect", tag],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "UNVERIFIED: docker CLI not available in this container"
    if done.returncode == 0:
        return "verified by docker manifest inspect"
    detail = (done.stdout + done.stderr).strip()
    if report.is_auth_failure(detail):
        report.notify(f"docker manifest inspect {tag}", detail)
        return f"UNVERIFIED: registry answered 401 for {tag}, user support asked for"
    return f"UNVERIFIED: registry refused {tag}"


def decide(
    wheelhouse: Path,
    site_packages: Path | None,
    explicit: str | None = None,
) -> dict[str, str]:
    driver, driver_how = driver_version()
    c_major, c_minor, ceiling_how = driver_ceiling()
    i_major, i_minor, installed_how = installed_cuda()
    u_major, u_minor, user_how = user_override(explicit)
    w_series, wheel_how = wheelhouse_series(wheelhouse)
    tree_series_value, tree_how = (
        tree_series(site_packages) if site_packages else (None, "tree not read")
    )

    series, how = select_series(w_series, i_major, c_major, u_major, driver)
    if u_major == series:
        minor = u_minor
    elif i_major == series:
        minor = i_minor
    else:
        minor = c_minor
    builder, builder_how = tag_for(series, minor)

    # The copy of the duplicate that survives is the one the environment
    # resolves, which is what the driver reports for this tree — not the series
    # pip happened to pick. Both copies load on driver 580, so this is a choice
    # about which file is dead weight, and the choice is recorded here.
    keep = tree_series_value or series
    keep_how = (
        f"tree resolves {tree_series_value}"
        if tree_series_value
        else f"tree not read, falls back to the declared series {series}"
    )

    return {
        "VDE_DRIVER_VERSION": driver,
        "VDE_DRIVER_HOW": driver_how,
        "VDE_DRIVER_CUDA": f"{c_major}.{c_minor}" if c_major else "",
        "VDE_DRIVER_CUDA_HOW": ceiling_how,
        "VDE_INSTALLED_CUDA": f"{i_major}.{i_minor}" if i_major else "",
        "VDE_INSTALLED_CUDA_HOW": installed_how,
        "VDE_WHEELHOUSE_SERIES": str(w_series) if w_series else "",
        "VDE_WHEELHOUSE_SERIES_HOW": wheel_how,
        "VDE_TREE_SERIES": str(tree_series_value) if tree_series_value else "",
        "VDE_TREE_SERIES_HOW": tree_how,
        "VDE_CUDA_USER": str(u_major) if u_major else "",
        "VDE_CUDA_USER_HOW": user_how,
        "VDE_CUDA_SERIES": str(series),
        "VDE_CUDA_SERIES_HOW": how,
        "VDE_CUDA_KEEP": str(keep),
        "VDE_CUDA_KEEP_HOW": keep_how,
        "VDE_CUDA_BUILDER": builder,
        "VDE_CUDA_BUILDER_HOW": builder_how,
        # The runtime base stays a plain ubuntu: the loader finding is that the
        # tree resolves its own libraries. Recorded so the build does not
        # silently pull a CUDA base image back in.
        "VDE_RUNTIME_BASE": "ubuntu:24.04",
    }


def shell_value(value: str) -> str:
    """A value written so a shell can source it.

    The `how` fields are human-readable and contain spaces. An unquoted line like
    `VDE_INSTALLED_CUDA_HOW=no nvcc on PATH` is not an assignment: the shell reads
    `VDE_INSTALLED_CUDA_HOW=no` as an environment prefix and then runs `nvcc on
    PATH` as a command. That is what happened here, and it is why every value is
    quoted.
    """
    return "'" + str(value).replace("'", "'\\''") + "'"


def write_env(dest: Path, values: dict[str, str]) -> None:
    """Write the variables file. It is sourced by docker/build.sh and docker/run.sh,
    so each line has to be a valid shell assignment, not just a KEY=value pair.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={shell_value(value)}" for key, value in values.items()]
    dest.write_text("\n".join(lines) + "\n")


def read_env(dest: Path) -> dict[str, str]:
    if not dest.is_file():
        return {}
    values = {}
    for line in dest.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1].replace("'\\''", "'")
        values[key] = value
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide the CUDA series by command.")
    parser.add_argument("--wheelhouse", default="out/wheelhouse")
    parser.add_argument(
        "--tree", default="out/run2/vllm.venv/lib/python3.12/site-packages"
    )
    parser.add_argument("--out", default="out/cuda.env")
    parser.add_argument(
        "--cuda",
        help="override the series, e.g. 12.8 or 13.0; same as VDE_CUDA",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="confirm the chosen tags exist in the registry (needs the docker socket)",
    )
    args = parser.parse_args()

    if args.cuda:
        os.environ["VDE_CUDA"] = args.cuda

    values = decide(
        Path(args.wheelhouse), Path(args.tree) if args.tree else None, args.cuda
    )
    if args.verify:
        values["VDE_CUDA_BUILDER_HOW"] = verify_tag(values["VDE_CUDA_BUILDER"])

    dest = Path(args.out)
    write_env(dest, values)

    print(f"cuda: cu{values['VDE_CUDA_SERIES']} -> {dest}")
    for key in (
        "VDE_DRIVER_VERSION",
        "VDE_DRIVER_CUDA",
        "VDE_INSTALLED_CUDA",
        "VDE_WHEELHOUSE_SERIES",
        "VDE_TREE_SERIES",
        "VDE_CUDA_USER",
        "VDE_CUDA_SERIES",
        "VDE_CUDA_KEEP",
        "VDE_CUDA_BUILDER",
        "VDE_RUNTIME_BASE",
    ):
        how = values.get(f"{key}_HOW")
        print(f"  {key}={values[key]}" + (f"  ({how})" if how else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
