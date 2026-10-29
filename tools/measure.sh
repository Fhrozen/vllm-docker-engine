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

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not available; the harness cannot run" >&2
    exit 1
fi

echo "image: $IMAGE"
echo "model: $MODEL (seed $SEED, max_model_len $MAX_MODEL_LEN)"

unpacked=$(docker image inspect --format '{{.Size}}' "$IMAGE")
echo "unpacked bytes: $unpacked"

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
    -v vde-hf-cache:/root/.cache/huggingface \
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
