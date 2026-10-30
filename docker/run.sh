#!/bin/bash
# The only script in this repo that executes anything, and it executes nothing on
# the host: every gate, test, and measurement runs inside a container. The host is
# used only to start containers.
#
#   bash docker/run.sh image
#   bash docker/run.sh gate docker/run.sh tools/check_files.py README.md
#   bash docker/run.sh test
#   bash docker/run.sh run packaging/package.py --verify /workspace/out/...
#   bash docker/run.sh size vllm-slim:dev vllm/vllm-openai:latest
#
# The repo is bind-mounted at /workspace, so gate fixes land back in the workspace.
# The cache volume keeps pinned tool downloads between runs.
set -eu

REPO="$(cd "$(dirname "$0")/.." && pwd)"

# A pull that fails with 401 is reported to the user rather than retried, so the
# wording lives in one shell helper that both this script and build.sh source.
# shellcheck disable=SC1090
. "$REPO/docker/pull.sh"

TOOLS="${VDE_TOOLS_IMAGE:-vde-tools:dev}"
CACHE="${VDE_CACHE_VOLUME:-vde-tools-cache}"
SOCKET="${VDE_DOCKER_SOCKET:-/var/run/docker.sock}"
WORKDIR="/workspace"
GPUS="${VDE_GPUS:-device=0}"
# The CUDA values come from the variables file that the `cuda` task writes, so a
# personal image is built from what this machine actually has. The fallback below
# is for a machine where no detection has run yet.
CUDA_ENV="${VDE_CUDA_ENV:-$REPO/out/cuda.env}"
if [ -f "$CUDA_ENV" ]; then
    # shellcheck disable=SC1090
    . "$CUDA_ENV"
fi

# 12.4.1-devel-ubuntu24.04 does not exist on the registry (manifest unknown);
# 12.8.1-devel-ubuntu24.04 does, and it matches the CUDA 12.8 libraries measured
# in the wheelhouse for this rule.
SPIKE_IMAGE="${VDE_SPIKE_IMAGE:-${VDE_CUDA_BUILDER:-nvidia/cuda:12.8.1-devel-ubuntu24.04}}"

# The bind-mounted repo is owned by the host uid, so git inside the container
# refuses the index as "dubious ownership". Configure it through env rather than
# writing config into the workspace.
GITENV=(-e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace)

TASK="${1:-help}"
if [ $# -gt 0 ]; then
    shift
fi

ensure_tools() {
    if [ -z "$(docker images -q "$TOOLS" 2>/dev/null)" ]; then
        echo "building $TOOLS"
        local detail
        if ! detail="$(docker build -f "$REPO/docker/Dockerfile.tools" -t "$TOOLS" "$REPO" 2>&1)"; then
            # The build pulls its base image from the registry, so a 401 shows up
            # here rather than in a separate pull step.
            report_failure "docker build of $TOOLS (base of docker/Dockerfile.tools)" "$detail"
            return 1
        fi
        printf '%s\n' "$detail"
    fi
}

# VDE_HOST_REPO is the path as the daemon sees it. A mount source written as
# /workspace/... inside a container is resolved on the host, so a script that
# mounts a workspace subdirectory must use the host path, not the container path.
HOSTENV=(-e VDE_HOST_REPO="$REPO")

run() {
    docker run --rm \
        -v "$REPO:$WORKDIR" \
        -v "$CACHE:/root/.cache" \
        -e PYTHONPATH="$WORKDIR" \
        -e VDE_WORKDIR="$WORKDIR" \
        "${GITENV[@]}" \
        "${HOSTENV[@]}" \
        "$TOOLS" "$@"
}

run_docker() {
    # Tasks that must talk to the daemon: they need the docker CLI (in the tools
    # image) and the socket. Nothing here runs a container from the host shell.
    docker run --rm \
        -v "$REPO:$WORKDIR" \
        -v "$CACHE:/root/.cache" \
        -v "$SOCKET:/var/run/docker.sock" \
        -e PYTHONPATH="$WORKDIR" \
        -e VDE_WORKDIR="$WORKDIR" \
        "${GITENV[@]}" \
        "${HOSTENV[@]}" \
        "$TOOLS" "$@"
}

run_gpu() {
    # Quoting matters: --gpus '"device=0"' breaks the daemon. Pass it unquoted.
    # shellcheck disable=SC2086
    docker run --rm \
        --gpus ${GPUS} \
        -v "$REPO:$WORKDIR" \
        -v "$CACHE:/root/.cache" \
        -v "$SOCKET:/var/run/docker.sock" \
        -e PYTHONPATH="$WORKDIR" \
        -e VDE_WORKDIR="$WORKDIR" \
        "${HOSTENV[@]}" \
        "$TOOLS" "$@"
}

case "$TASK" in
    image)
        ensure_tools
        ;;
    build)
        # build.sh calls the docker CLI, so it needs the socket mounted.
        # Everything still runs inside the container.
        ensure_tools
        run_docker bash "$WORKDIR/docker/build.sh" "$@"
        ;;
    gate)
        ensure_tools
        run bash "$WORKDIR/docker/gates.sh" "$@"
        ;;
    test)
        ensure_tools
        run python3 -m pytest "$WORKDIR/tests" -q "$@"
        ;;
    script)
        ensure_tools
        run python3 "$WORKDIR/$1" "${@:2}"
        ;;
    py)
        # Run a snippet inside the container. Nothing runs on the host.
        ensure_tools
        run python3 -c "$1"
        ;;
    dist)
        ensure_tools
        run python3 "$WORKDIR/tools/check_dist_gate.py" "$@"
        ;;
    size)
        # Both gate scripts take the image as a flag, so the documented positional
        # form is translated here rather than at every call site. A call that already
        # uses flags is passed through unchanged.
        ensure_tools
        if [ -z "${1:-}" ] || [ "${1#-}" != "$1" ]; then
            run_docker python3 "$WORKDIR/tools/check_image_size.py" "$@"
        else
            run_docker python3 "$WORKDIR/tools/check_image_size.py" \
                --image "$1" "${@:2}"
        fi
        ;;
    runtime)
        ensure_tools
        if [ -z "${1:-}" ] || [ "${1#-}" != "$1" ]; then
            run_docker python3 "$WORKDIR/tools/check_runtime_image.py" "$@"
        else
            run_docker python3 "$WORKDIR/tools/check_runtime_image.py" \
                --image "$1" "${@:2}"
        fi
        ;;
    measure)
        ensure_tools
        run_gpu bash "$WORKDIR/tools/measure.sh" "$@"
        ;;
    serve-smoke)
        ensure_tools
        run_gpu bash "$WORKDIR/docker/smoke-test.sh" "$@"
        ;;
    spike)
        # Needs nvcc, so this runs in a CUDA devel container, not the tools image.
        check_image "$SPIKE_IMAGE" || exit 1
        # shellcheck disable=SC2086
        docker run --rm \
            --gpus ${GPUS} \
            -v "$REPO:$WORKDIR" \
            -v "$CACHE:/root/.cache" \
            -v "$SOCKET:/var/run/docker.sock" \
            -e PYTHONPATH="$WORKDIR" \
            -e VDE_WORKDIR="$WORKDIR" \
            --entrypoint bash \
            "$SPIKE_IMAGE" -c "bash $WORKDIR/tools/spike_torch_distributed.sh $*"
        ;;
    compile)
        # The tools image has no C compiler on purpose (R01). This task installs
        # gcc and the C headers for one run only, to measure what a build stage
        # can pre-produce and what the runtime image then needs. The pinned image
        # is not changed. triton needs both: gcc alone fails on <stdlib.h>.
        # shellcheck disable=SC2086
        docker run --rm \
            --gpus ${GPUS} \
            -v "$REPO:$WORKDIR" \
            -v "$CACHE:/root/.cache" \
            -e PYTHONPATH="$WORKDIR" \
            -e VDE_WORKDIR="$WORKDIR" \
            "$TOOLS" bash -c "apt-get update -qq >/dev/null && apt-get install -y -qq --no-install-recommends gcc libc6-dev >/dev/null && $*"
        ;;
    warm)
        # The build-stage half of the JIT decision, run through the harness so the
        # host stays a launcher. Two things this task supplies that the tool itself
        # cannot: a C compiler for the run (the tools image has none, per R01, and
        # triton's build needs gcc plus the C headers), and the arch the cache is
        # warmed for, read from the card rather than guessed.
        # shellcheck disable=SC2086
        docker run --rm \
            --gpus ${GPUS} \
            -v "$REPO:$WORKDIR" \
            -v "$CACHE:/root/.cache" \
            -e PYTHONPATH="$WORKDIR" \
            -e VDE_WORKDIR="$WORKDIR" \
            "${HOSTENV[@]}" \
            "$TOOLS" bash -c '
                apt-get update -qq >/dev/null
                apt-get install -y -qq --no-install-recommends gcc libc6-dev >/dev/null
                cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ".")
                python3 "$VDE_WORKDIR/tools/warm_cache.py" --arch "sm_$cap" "$@"
            ' bash "$@"
        ;;
    spike-run)
        # A command in the CUDA devel container. Needed for measurements that
        # require a compiler (triton builds a C extension at runtime) or nvcc.
        check_image "$SPIKE_IMAGE" || exit 1
        # shellcheck disable=SC2086
        docker run --rm \
            --gpus ${GPUS} \
            -v "$REPO:$WORKDIR" \
            -v "$CACHE:/root/.cache" \
            -v "$SOCKET:/var/run/docker.sock" \
            -e PYTHONPATH="$WORKDIR" \
            -e VDE_WORKDIR="$WORKDIR" \
            --entrypoint bash \
            "$SPIKE_IMAGE" -c "$*"
        ;;
    cuda)
        # Decide the CUDA series by command: driver, toolchain, wheelhouse. The
        # answer is written to the variables file that build.sh then reads.
        # Needs the GPU for nvidia-smi and the socket for --verify.
        ensure_tools
        run_gpu python3 "$WORKDIR/packaging/cuda.py" \
            --out "$WORKDIR/out/cuda.env" "$@"
        ;;
    gpu)
        # A command in the tools container with a GPU attached, for arch detection.
        # shellcheck disable=SC2086
        run_gpu bash -c "$*"
        ;;
    docker)
        # A docker CLI command inside the container. Used to move content between
        # images (create + cp) without executing anything on the host.
        ensure_tools
        run_docker bash -c "$*"
        ;;
    pull)
        # Pull an image through the helper so a 401 is reported to the user
        # instead of retried. Nothing is downloaded on the host.
        check_image "$1" || exit 1
        ;;
    shell)
        ensure_tools
        docker run --rm -it \
            -v "$REPO:$WORKDIR" \
            -v "$CACHE:/root/.cache" \
            -e PYTHONPATH="$WORKDIR" \
            -e VDE_WORKDIR="$WORKDIR" \
            "$TOOLS" bash
        ;;
    help | *)
        cat <<'USAGE'
docker/run.sh <task> [args]

  image                 build the pinned gate/test container
  build [args]          run docker/build.sh inside the container
  gate [files...]       ruff, typos, markdownlint, shellcheck, file checks
                        (files must be passed explicitly; no files is not a pass)
  test [pytest args]    the packaging suite, inside the container
  script <file> [args]  run a repo python script inside the container
  py "<snippet>"        run a snippet inside the container
  dist [args]           distributed gate against the read-only clone
  size <image> [base]   size + budget gate (passes --image / --baseline)
  runtime <image>       runtime image config gate (R01, passes --image)
  measure <image>       cold start, tokens/s, peak GPU memory
  serve-smoke <image>   fixed-seed inference against a running container
  spike [variant]       libtorch build variants (CUDA devel container)
  spike-run "<command>" run a command in the CUDA devel container (needs a compiler)
  compile "<command>"   run a command with gcc + libc6-dev installed for one run
  warm [args]           warm the JIT cache on the GPU with a compiler for the run,
                        arch read from the card, then relocate to the runtime path
  cuda [args]           decide the CUDA series by command, write out/cuda.env
  gpu "<command>"       run a command in the tools container with a GPU attached
  docker "<command>"    docker CLI command inside the container, for moving
                        content between images (create + cp)
  pull <image>            pull an image through the helper: a 401 is reported to
                        the user, not retried
  shell                 interactive shell inside the container

Nothing in this script runs on the host interpreter.
USAGE
        ;;
esac
