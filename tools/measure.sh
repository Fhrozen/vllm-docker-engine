#!/bin/bash
# Measurement harness for the Phase 4 gate.
#
# Every number printed here comes from a run. Nothing is estimated, and a value
# that cannot be read is printed as UNVERIFIED instead of inferred.
#
# Metrics: image size, compressed download, cold start (container start to
# healthcheck green), tokens/s on a fixed seed, peak GPU memory.
set -eu

REPO="$(cd "$(dirname "$0")/.." && pwd)"

IMAGE="${1:-vllm-slim:dev}"
MODEL="${VDE_MODEL:-Qwen/Qwen3.5-0.8B}"
SEED="${VDE_SEED:-0}"
MAX_MODEL_LEN="${VDE_MAX_MODEL_LEN:-2048}"
GPUS="${VDE_GPUS:-device=0}"
PORT="${VDE_PORT:-8000}"
TIMEOUT="${VDE_TIMEOUT:-600}"
NAME="vde-measure-$(date +%s)-$$"
# Two different paths, and confusing them produced a download failure:
#   - the mount SOURCE must be the host path, because the daemon resolves a
#     bind-mount source on the host (docker/run.sh passes it as VDE_HOST_REPO);
#   - HF_HOME must be the path the measured CONTAINER sees, i.e. the mount target.
# Setting HF_HOME to the host path made the server try to write outside its own
# filesystem and fail with "PermissionError at /export".
HOST_CACHE="${VDE_HF_CACHE:-${VDE_HOST_REPO:-$REPO}/out/hf-cache}"
CACHE_MOUNT="/opt/hf-cache"
# The same folder, as this container sees it through the workspace mount.
LOCAL_CACHE="$REPO/out/hf-cache"
CACHE_STATE=empty
CACHE_FILES=0
CACHE_KIB=0
# Warm must mean "the model is actually here", not "the folder has some file in it":
# an earlier run reported warm on a folder holding only a probe file, which made a
# download-inclusive cold start look like a warm-cache number.
if [ -n "$(find "$LOCAL_CACHE" -path '*/models--*' -type f 2>/dev/null | head -1)" ]; then
    CACHE_STATE=warm
    CACHE_FILES=$(find "$LOCAL_CACHE" -type f | wc -l)
    CACHE_KIB=$(du -sk "$LOCAL_CACHE" | cut -f1)
fi
if [ ! -d "$LOCAL_CACHE" ]; then
    # Created here it would be root-owned, and the measured container runs as
    # uid 1000. The harness reports it rather than silently failing at download.
    echo "hf cache: MISSING ($LOCAL_CACHE) — create it on the host before running" >&2
fi

# The image declares a VOLUME for the HF cache, but a named volume is created owned
# by root while the container runs as uid 1000: measured, `touch` inside the mounted
# volume returned Permission denied. A bind mount of a workspace directory is writable
# by the runtime user, so the harness uses one. The cache state is printed because a
# cold cache puts the model download inside the cold-start number.

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not available; the harness cannot run" >&2
    exit 1
fi

echo "image: $IMAGE"
echo "model: $MODEL (seed $SEED, max_model_len $MAX_MODEL_LEN)"

unpacked=$(docker image inspect --format '{{.Size}}' "$IMAGE")
echo "unpacked bytes: $unpacked"
mkdir -p "$REPO/out/logs"
LOG_FILE="$REPO/out/logs/$(echo "$IMAGE" | tr ':/' '__').log"

echo "hf cache: $CACHE_STATE ($CACHE_FILES files, $CACHE_KIB KiB) mounted at $CACHE_MOUNT — a cold cache puts the model download inside cold start"

if command -v python3 >/dev/null 2>&1; then
    python3 "$REPO/tools/check_image_size.py" --image "$IMAGE" --json |
        python3 -c '
import json, sys
r = json.load(sys.stdin)
print("compressed bytes:", r["compressed_bytes"] or "UNVERIFIED",
      "(" + r["compressed_how"] + ")")
'
else
    echo "compressed bytes: UNVERIFIED (python3 unavailable)"
fi

# --entrypoint is overridden because the image ENTRYPOINT is the server itself;
# the harness needs to control startup timing and run the smoke test separately.
# shellcheck disable=SC2086  # --gpus must not be quoted; quoting breaks the daemon
docker run -d \
    --name "$NAME" \
    --gpus ${GPUS} \
    --shm-size 1gb \
    --ipc host \
    -p "${PORT}:8000" \
    -e VLLM_MODEL="$MODEL" \
    -e VDE_SEED="$SEED" \
    -e HF_HOME="$CACHE_MOUNT" \
    -v "$HOST_CACHE:$CACHE_MOUNT" \
    "$IMAGE" \
    --model "$MODEL" --seed "$SEED" --max-model-len "$MAX_MODEL_LEN" >/dev/null

cleanup() {
    docker stop "$NAME" >/dev/null 2>&1 || true
    docker rm "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

start=$(date +%s)
status="starting"
while [ "$status" != "healthy" ]; do
    status=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$NAME")
    if [ "$status" = "none" ]; then
        echo "cold start: UNVERIFIED (image has no HEALTHCHECK)"
        break
    fi
    if [ $(( $(date +%s) - start )) -gt "$TIMEOUT" ]; then
        echo "cold start: TIMEOUT after ${TIMEOUT}s (status $status)"
        break
    fi
    sleep 5
done
if [ "$status" = "healthy" ]; then
    echo "cold start seconds: $(( $(date +%s) - start ))"
else
    # The reason a boot fails is in the container log, not in the health status.
    # The last 40 lines alone are not enough: vLLM prints the real cause earlier,
    # in the engine-core worker, and tail hid it.
    docker logs "$NAME" > "$LOG_FILE" 2>&1
    echo "--- container log (last 40 lines; full log in $LOG_FILE) ---"
    tail -40 "$LOG_FILE"
fi

peak=0
if command -v nvidia-smi >/dev/null 2>&1; then
    for _ in $(seq 1 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        [ -n "$used" ] && [ "$used" -gt "$peak" ] && peak=$used
        sleep 2
    done
    echo "peak GPU memory MiB: $peak"
else
    echo "peak GPU memory MiB: UNVERIFIED (nvidia-smi unavailable)"
fi

bash "$REPO/docker/smoke-test.sh"
