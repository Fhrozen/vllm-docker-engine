#!/bin/bash
# Healthcheck: exit 0 only on HTTP 200 with a non-empty data list.
# Uses stdlib only, so the runtime image needs no extra package.
set -eu

URL="${VLLM_HEALTH_URL:-http://127.0.0.1:${PORT:-8000}/v1/models}"

python3 - "$URL" <<'CHECK'
import json
import sys
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        if response.status != 200:
            sys.exit(1)
        body = json.load(response)
except Exception:
    sys.exit(1)

if not body.get("data"):
    sys.exit(1)
sys.exit(0)
CHECK
