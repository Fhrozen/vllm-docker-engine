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
TOOLS="${VDE_TOOLS_IMAGE:-vde-tools:dev}"
CACHE="${VDE_CACHE_VOLUME:-vde-tools-cache}"
SOCKET="${VDE_DOCKER_SOCKET:-/var/run/docker.sock}"
WORKDIR="/workspace"
GPUS="${VDE_GPUS:-device=0}"
SPIKE_IMAGE="${VDE_SPIKE_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu24.04}"

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
        docker build -f "$REPO/docker/Dockerfile.tools" -t "$TOOLS" "$REPO"
    fi
}

run() {
    docker run --rm \
        -v "$REPO:$WORKDIR" \
        -v "$CACHE:/root/.cache" \
        -e PYTHONPATH="$WORKDIR" \
        -e VDE_WORKDIR="$WORKDIR" \
        "${GITENV[@]}" \
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
        ensure_tools
        run_docker python3 "$WORKDIR/tools/check_image_size.py" "$@"
        ;;
    runtime)
        ensure_tools
        run_docker python3 "$WORKDIR/tools/check_runtime_image.py" "$@"
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
    gpu)
        # A command in the tools container with a GPU attached, for arch detection.
        # shellcheck disable=SC2086
        run_gpu bash -c "$*"
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
  size <image> [base]   size + budget gate
  runtime <image>       runtime image config gate (R01)
  measure <image>       cold start, tokens/s, peak GPU memory
  serve-smoke <image>   fixed-seed inference against a running container
  spike [variant]       libtorch build variants (CUDA devel container)
  gpu "<command>"       run a command in the tools container with a GPU attached
  shell                 interactive shell inside the container

Nothing in this script runs on the host interpreter.
USAGE
        ;;
esac
