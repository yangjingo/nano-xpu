# 服务启动与请求示例

脚本与配置保存在以下命名代码块中，按 filename 保存后使用。两类拓扑都通过 `nano-xpu serve -f` 启动；单 worker 也经 router。CPU harness 只验证接口，生产 NIXL/P/D 尚未完成。运行要求见[部署说明](../docs/deployment.md)。

```powershell filename=serve-cpu.ps1
# Test-only HTTP server: synthetic output, no Qwen weights are loaded.
param(
    [string]$Python = 'python',
    [int]$Port = 8000,
    [string]$RepoRoot = (Get-Location).Path
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$previousPythonPath = $env:PYTHONPATH
Push-Location $repoRoot
try {
    if (Test-Path -LiteralPath '.cpu-deps') {
        $env:PYTHONPATH = (Join-Path $repoRoot '.cpu-deps') + [IO.Path]::PathSeparator + $env:PYTHONPATH
    }
    & $Python -m tests.helpers.cpu_server _worker . --port $Port --served-model-name qwen3-0.6b
    if ($LASTEXITCODE -ne 0) { throw "CPU test server exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
```

```powershell filename=request-cpu.ps1
# Call the explicitly test-only server; optionally request a clean shutdown.
param(
    [string]$BaseUrl = 'http://127.0.0.1:8000',
    [switch]$Shutdown
)
$ErrorActionPreference = 'Stop'
$BaseUrl = $BaseUrl.TrimEnd('/')
Invoke-RestMethod "$BaseUrl/ready"
$body = @{
    model = 'qwen3-0.6b'
    messages = @(@{ role = 'user'; content = '你好' })
    temperature = 0.6
    max_tokens = 16
} | ConvertTo-Json -Depth 5
$result = Invoke-RestMethod "$BaseUrl/v1/chat/completions" -Method Post -ContentType 'application/json; charset=utf-8' -Body $body
if ($result.choices[0].message.content -notlike '[[]CPU TEST ENGINE:*') {
    throw 'Expected an explicitly marked CPU test engine response'
}
$result | ConvertTo-Json -Depth 8
if ($Shutdown) {
    Invoke-RestMethod "$BaseUrl/__test__/shutdown" -Method Post
}
```

```bash filename=serve-qwen.sh
#!/usr/bin/env bash
# All model/device settings belong to the cluster YAML and its runtime profile.
# Real CUDA/router verification is still required.
set -euo pipefail
: "${CLUSTER_CONFIG:?Set CLUSTER_CONFIG to a cluster YAML file}"
if [[ ! -f "$CLUSTER_CONFIG" ]]; then
  echo "Cluster configuration does not exist: $CLUSTER_CONFIG" >&2
  exit 1
fi
exec nano-xpu serve -f "$CLUSTER_CONFIG" "$@"
```

```bash filename=request-qwen.sh
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
```
