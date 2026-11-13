#!/usr/bin/env bash
# RFC-0001 request smoke check; works with real or explicitly test-only servers.
# Response shape alone does not verify that a real Qwen model is running.
set -euo pipefail

export BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b}"

python - <<'PY'
import json
import os
import time
import urllib.error
import urllib.request

base_url = os.environ["BASE_URL"].rstrip("/")
deadline = time.monotonic() + 300
while True:
    try:
        with urllib.request.urlopen(base_url + "/ready", timeout=2) as response:
            if response.status == 200:
                break
    except (urllib.error.URLError, TimeoutError):
        pass
    if time.monotonic() >= deadline:
        raise SystemExit("Server did not become ready within 300 seconds")
    time.sleep(1)

payload = {
    "model": os.environ["SERVED_MODEL_NAME"],
    "messages": [{"role": "user", "content": "请用一句话介绍你自己。"}],
    "temperature": 0.6,
    "max_tokens": 64,
    "stream": False,
}
request = urllib.request.Request(
    base_url + "/v1/chat/completions",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
except urllib.error.HTTPError as exc:
    raise SystemExit(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")

assert result["object"] == "chat.completion", result
assert result["model"] == payload["model"], result
assert len(result["choices"]) == 1, result
choice = result["choices"][0]
assert choice["message"]["role"] == "assistant", result
assert isinstance(choice["message"]["content"], str), result
assert choice["finish_reason"] in ("stop", "length"), result
usage = result["usage"]
assert usage["prompt_tokens"] > 0 and usage["completion_tokens"] > 0, result
assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], result
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
