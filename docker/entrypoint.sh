#!/bin/bash
# Runtime entrypoint. Nothing here installs, downloads, or compiles (gate R01).
set -eu

MODULE="${VLLM_ENTRY_MODULE:-vllm.entrypoints.cli.main}"

if [ -n "${VLLM_MODEL:-}" ]; then
    exec python3 -m "$MODULE" serve --model "$VLLM_MODEL" "$@"
fi

exec python3 -m "$MODULE" serve "$@"
