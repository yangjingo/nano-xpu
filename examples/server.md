# 服务启动与请求示例

第一次调试服务时，往往还没有准备好模型权重和 GPU 环境，但已经需要确认客户端能否连接、请求格式是否正确，以及服务能否正常退出。为此，本页先用 CPU 测试服务走通一次“启动 → 请求 → 停止”，再说明准备好模型环境以后，怎样通过配置启动 Qwen，并把请求发送给 router。

CPU 示例依赖完整的 Nano-XPU checkout，使用仓库里的测试引擎生成带标记的固定响应，便于检查 HTTP 链路。Qwen 示例则通过公开的 `nano-xpu serve -f` 入口管理 router 和 workers，节点、模型和卡号来自[部署配置](deploy.md)。两条路径的用途不同：CPU 检查通过后，真实模型的输出、设备资源和跨节点链路仍需分别验证。目前真实 CUDA/Qwen 服务和生产 NIXL/P/D 尚未完成设备验收。

下面的脚本都完整保存在代码块中。按各节说明保存为对应文件名后即可运行；本目录只保留这一份文档来源。测试和 CI 也会从这些代码块提取临时文件进行检查。

## CPU 服务启动

这一步适合先验证接口、客户端或本地开发环境。按 [CPU 验证指南](../docs/testing.md)准备 Python 3.12 和测试依赖，并保留仓库的 `tests/` 目录；只安装生产 wheel 不包含这里使用的测试引擎。

将下面代码保存为 `serve-cpu.ps1`。脚本进入指定 checkout，如果存在 `.cpu-deps` 就把它加入当前子进程的 Python 搜索路径，然后启动测试 HTTP 服务。它默认监听本机 `127.0.0.1:8000`，服务名为 `qwen3-0.6b`。这个名字用于请求匹配，启动过程不会读取同名模型权重。脚本退出时会恢复原来的目录和 Python 搜索路径。

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

如果将脚本保存到仓库根目录，并已在当前 Python 环境安装测试依赖，可以在第一个 PowerShell 终端运行：

```powershell
.\serve-cpu.ps1 -Python python
```

使用 CPU 指南中的独立解释器时，把 `python` 换成该解释器的完整路径，例如已经设置好的 `$python312`。若脚本保存在别处，增加 `-RepoRoot` 指向 checkout；8000 端口被占用时，可以增加 `-Port 8001`，并在后续请求中使用同一个端口。

看到 `CPU TEST SERVER: synthetic engine; no Qwen model loaded` 和 HTTP 监听日志后，保留这个终端，在第二个终端执行下一节。提示缺少模块时，先检查所选解释器及其依赖；提示地址被占用时，检查端口或已有服务，不要同时启动两个监听同一端口的实例。

## CPU 请求与停止

服务启动后，需要确认它不仅打开了端口，还能处理一条完整的聊天请求。下面的脚本先访问 `/ready`，再向 `/v1/chat/completions` 发送一句“你好”，最后检查响应是否带有 `CPU TEST ENGINE` 标记。这个检查也能防止误把同一端口上的其他服务当成本次测试服务。

将代码保存为 `request-cpu.ps1`。默认只请求并打印结果；传入 `-Shutdown` 时，它会在请求检查成功后调用测试服务专用的退出端点。请求失败或返回内容不符合预期时，脚本会报错，不会继续执行后面的退出请求。

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

在第二个 PowerShell 终端运行：

```powershell
.
equest-cpu.ps1
# 完成检查后，再发送一次请求并停止测试服务。
.
equest-cpu.ps1 -Shutdown
```

如果启动时选择了 8001 端口，使用 `-BaseUrl http://127.0.0.1:8001`。成功时会先看到 ready 的结果，再看到带测试标记的聊天 JSON；加上 `-Shutdown` 后，第一个终端的服务应正常退出。此脚本不等待启动完成，连接失败或 ready 未通过时应先回到启动终端查看日志。

这里的退出端点只存在于 CPU 测试服务中。通过配置启动的 Qwen 集群使用后文的 `serve down` 停止。

## Qwen 配置启动

准备实际模型服务时，先阅读[部署配置](deploy.md)，保存拓扑与公共 profile，并准备节点上的模型、Python 环境、router 和 SSH。启动脚本只负责把选定配置交给 Controller；模型不会由它下载，远端依赖也不会由它自动安装。

将下面代码保存为 `serve-qwen.sh`。`CLUSTER_CONFIG` 指定要使用的拓扑 YAML，脚本检查文件存在后执行 `nano-xpu serve -f`。后续参数原样传给 CLI，因此前台运行和 `--detach` 使用同一套配置。`exec` 让脚本进程交由 CLI 接管，前台使用 Ctrl+C 时由 Controller 执行停止流程。

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

在已安装 Nano-XPU、可调用 `nano-xpu` 的 Linux 环境中，进入保存脚本和 YAML 的工作目录，先检查计划，再启动：

```bash
nano-xpu serve plan -f ./cluster-colocated.yaml
CLUSTER_CONFIG=./cluster-colocated.yaml bash ./serve-qwen.sh
```

确认 plan 中的地址、卡号、命令及未解决项后再启动。Controller 会先检查节点，启动并核对 workers，最后启动 router；只有完整服务就绪后才返回 router 地址。默认前台继续监测，Ctrl+C 会停止所管理的服务。需要后台运行时，在命令末尾添加 `--detach`，返回后仍可查询和停止：

```bash
nano-xpu serve status --name qwen-colocated
nano-xpu serve logs --name qwen-colocated
nano-xpu serve down --name qwen-colocated
```

这里的名称来自拓扑中的 `name`，并且这些命令须使用启动时的同一个状态目录。后台模式表示 Controller 完成就绪检查后返回，不表示可以省略后续状态检查。若启动报错，先根据错误检查 profile、SSH 或节点日志；没有得到就绪结果时，不应继续把这次部署当成可用服务。

## Qwen 请求检查

集群就绪后，客户端应连接返回的 router 地址。worker 的内部端口用于转发和探测，通常不作为用户请求入口。下面的脚本适合做第一条请求检查：它等待 `/ready`，发送一条非流式聊天请求，并检查响应中的模型名、assistant 消息、结束原因和 token 计数是否自洽。

将代码保存为 `request-qwen.sh`，运行环境需要 Bash，以及 PATH 中可用的 `python`。`BASE_URL` 指定服务地址，`SERVED_MODEL_NAME` 必须与 profile 中的 `served_model_name` 一致；它是对外服务名，可以与模型目录名不同。

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

如果按部署示例把 router 配置在 `10.0.0.10:8000`，在能访问该地址的终端运行：

```bash
BASE_URL=http://10.0.0.10:8000 SERVED_MODEL_NAME=qwen3-0.6b bash ./request-qwen.sh
```

脚本最多等待 ready 约 300 秒，聊天请求的超时为 120 秒。成功时打印完整 JSON；HTTP 错误会打印状态码与响应内容，字段检查失败则报断言错误。ready 超时通常需要回到 `serve status` 和日志检查服务状态；模型名不匹配时，核对客户端变量与 profile。

这个脚本也可以连接前面的 CPU 测试服务，此时保留默认的 `http://127.0.0.1:8000` 即可。通过检查只说明所连接服务完成了这条 HTTP 请求，响应结构符合预期；真实 Qwen 是否生成正确内容，还需要结合模型加载日志、实际输出和设备验收结果判断。
