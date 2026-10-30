#!/bin/bash
# Smoke test: one fixed-seed completion against a running server.
# Prints tokens/s so the measurement table in docs/SIZES.md can be filled from
# an actual run rather than an estimate.
set -eu

URL="${VLLM_SMOKE_URL:-http://127.0.0.1:${PORT:-8000}}"
MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-0.8B}"
PROMPT="${VLLM_SMOKE_PROMPT:-The capital of France is}"
SEED="${VLLM_SMOKE_SEED:-0}"
MAX_TOKENS="${VLLM_SMOKE_MAX_TOKENS:-64}"

python3 - "$URL" "$MODEL" "$PROMPT" "$SEED" "$MAX_TOKENS" <<'SMOKE'
import json
import sys
import time
import urllib.request

url, model, prompt, seed, max_tokens = sys.argv[1:6]
payload = json.dumps(
    {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "seed": int(seed),
        "temperature": 0.0,
    }
).encode()

request = urllib.request.Request(url + "/v1/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
start = time.monotonic()
with urllib.request.urlopen(request, timeout=300) as response:
    body = json.load(response)
elapsed = time.monotonic() - start

usage = body.get("usage") or {}
count = usage.get("completion_tokens") or 0
if not count:
    print("smoke: no completion tokens returned", file=sys.stderr)
    sys.exit(1)

print(f"smoke: {count} tokens in {elapsed:.2f}s -> {count / elapsed:.2f} tokens/s")
print(f"smoke: {body.get('choices', [{}])[0].get('text', '').strip()!r}")
SMOKE
