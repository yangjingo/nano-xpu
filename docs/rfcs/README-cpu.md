# 本地 CPU 验证：nano-xpu serve 服务入口与切分参数

> 入口更新：本文保留早期服务设计记录。当前配置启动和 CPU 示例以[部署说明](../deployment.md)及其链接为准；旧直接模型命令已停止支持。

本说明对应 [RFC-0001](0001-serve-qwen-colocated.md) 的 S0。测试引擎只在 `tests/` 中提供合成响应，**不会加载 Qwen，不证明真实模型推理或 TP 多卡正确**。

自动化工作流、wheel 安装验证与 CI 验收结果见 [CPU CI 指南](../ci.md)。

## 标准 Python 3.12 环境

在仓库根目录执行；CPU 环境不安装 CUDA 核心依赖：

```bash
python3.12 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install -r tests/requirements-cpu.txt
python -m pip install --no-deps -e .
python -m pytest -q tests
nano-xpu serve --help
```

`--no-deps` 仅用于服务层 CPU 测试安装。生产 Qwen 服务仍需要项目声明的 CUDA 推理依赖，不应使用此安装方式冒充完整推理环境。

## 本机 Windows 的已用方式

本机可直接运行 uv 管理的 CPython 3.12.13，但新建虚拟环境中的启动器被 Windows Application Control 拒绝。因此使用原解释器配合局部依赖目录，不更改系统策略：

```powershell
$python312 = "$env:APPDATA\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe"
uv pip install --python $python312 --target .cpu-deps -r tests/requirements-cpu.txt
uv pip install --python $python312 --target .cpu-deps --no-deps -e .
$env:PYTHONPATH = Join-Path (Get-Location) '.cpu-deps'
& $python312 -m pytest -q tests
& $python312 -m nanoxpu serve --help
```

`.cpu-deps` 和 `.venv-server` 均不提交。安装元数据中的 `nano-xpu` 入口已由测试加载校验，本机 CLI 行为通过等价的 `python -m nanoxpu` 入口验证；Windows 生成的 console launcher 已尝试启动但被 Application Control 拦截，不能标为运行通过。

## 手动启动 CPU 测试服务

也可直接使用 [启动脚本](examples/serve-cpu.ps1) 与 [调用脚本](examples/request-cpu.ps1)。从仓库根目录，在两个终端分别执行：

```powershell
# 终端 1，Python 路径使用前文的 $python312，或你自己的 Python 3.12 环境。
.\docs\rfcs\examples\serve-cpu.ps1 -Python $python312
# 终端 2，调用并正常关闭测试服务。
.\docs\rfcs\examples\request-cpu.ps1 -Shutdown
```

终端 1（使用上面准备的 Python/PYTHONPATH）：

```powershell
& $python312 -m tests.helpers.cpu_server serve . --served-model-name qwen3-0.6b --port 8000
```

这会使用当前目录作为测试配置路径，**不读取模型权重**。控制台和输出文本都会标记 CPU TEST。该 harness 使用与生产相同的 HTTP app 和 ManagedServer，仅替换引擎，并附带仅测试环境存在的 shutdown 路由。

终端 2：

```powershell
$body = @{
    model = 'qwen3-0.6b'
    messages = @(@{role = 'user'; content = '你好'})
    max_tokens = 16
    temperature = 0.6
} | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:8000/ready
Invoke-RestMethod http://127.0.0.1:8000/v1/chat/completions -Method Post -ContentType 'application/json; charset=utf-8' -Body $body
Invoke-RestMethod http://127.0.0.1:8000/__test__/shutdown -Method Post
```

Linux 上可用 `python -m tests.helpers.cpu_server serve . --served-model-name qwen3-0.6b` 拉起同一测试服务，再执行 [`request-qwen.sh`](examples/request-qwen.sh)。该调用脚本只校验接口响应，不自行断言服务后端是真实模型；真实 Qwen 验收必须另附模型、设备和服务启动证据。

## 验证范围

- CLI/HTTP 不依赖 torch/transformers/triton/flash-attn 的导入路径；切分参数、输入错误和响应格式。
- 专属线程、一次一个生成请求、断连后保留槽位、故障拒绝新请求、启动失败清理。
- 子进程真实 HTTP 请求、示例调用脚本、正常关闭和同端口重启。
- CUDA 适配器通过桩对象验证模板 token IDs、TP/token 预算传递与长度/KV 容量拒绝；不执行 CUDA。
- 原 Scheduler 处理 `4 + 4 + 2` 个 Prefill token 后进入 Decode，结束时释放活动 block 引用。

真实 Qwen 输出、CUDA Graph、NCCL/TP、设备 OOM、SIGTERM 设备资源释放和性能仍待目标设备验证。测试引擎不进入生产 `nano-xpu serve`，默认入口没有 CPU fallback。
