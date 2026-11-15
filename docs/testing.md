# CPU 验证指南与结果

本页集中记录复现方式、历史 S0 与当前 CPU 验证。CI 安装、完整测试和 Linux 集成命令见 [CI 指南](ci.md)，部署操作见 [部署说明](deployment.md)。测试引擎不会加载 Qwen，CPU 结果不能代替真实模型、CUDA/NCCL、NIXL 或跨机验收。

## Windows 本地复现

本机新建虚拟环境和 console launcher 曾被 Application Control 拦截，使用可运行的 CPython 3.12 原解释器及独立依赖目录，不修改系统策略。在仓库根目录执行：

```powershell
$python312 = "$env:APPDATA\uv\python\cpython-3.12.13-windows-x86_64-none\python.exe"
uv pip install --python $python312 --target .cpu-deps -r tests/requirements-cpu.txt
uv pip install --python $python312 --target .cpu-deps --no-deps -e .
$env:PYTHONPATH = Join-Path (Get-Location) '.cpu-deps'
& $python312 -m pytest -q tests --ignore=tests/integration --strict-markers
& $python312 -m nanoxpu serve --help
```

`--no-deps` 只适用于这条 CPU 测试安装路径；生产模型需要完整推理依赖。`.cpu-deps` 不提交，Python 模块入口通过也不代表受限的 `.exe` launcher 通过。

先按文档将代码分别保存为仓库根目录的 `serve-cpu.ps1` 与 `request-cpu.ps1`（仅作本地临时文件，不提交），再在两个终端运行 [启动示例](../examples/server.md#cpu-服务启动) 和 [请求示例](../examples/server.md#cpu-请求与停止)：

```powershell
# 终端 1，沿用上述 Python 路径。
.\serve-cpu.ps1 -Python $python312
# 终端 2，完成请求后停止测试服务。
.\request-cpu.ps1 -Shutdown
```

Linux 请求检查也先按[请求示例](../examples/server.md#qwen-请求检查)保存 `request-qwen.sh`。服务可执行 `python -m tests.helpers.cpu_server _worker . --served-model-name qwen3-0.6b`，再运行 `bash ./request-qwen.sh`。这些请求只验证 HTTP 行为；真实 Qwen 启动必须使用 [CUDA 示例](../examples/server.md#qwen-配置启动) 并另附设备证据。

## 历史 S0 记录

2026-11-13，基线 `40a001f99181321d8b2ab30a0b8a1cfcdba0c9d5` 加当时工作区实现，在 Windows / CPython 3.12.13 运行 **43 passed in 6.39s**，零失败、零跳过。服务入口、配置传递、HTTP 路由、并发/断连、退出重启与 Scheduler 分块通过 CPU 验证；不证明 Qwen/TP 正确。对应实现随后记录于 `b734b0b797d58e9b5e470dec420c187709d4f4fd`。

原始 `cpu-test-results.xml` 与旧说明保存在工作区 `output/rfc-cleanup-20261115/`；旧文档也可从 Git 历史追溯。原始 XML、运行日志及发布记录不放入 RFC 目录。

## 当前 CPU 部署验证

状态：本地阶段性验证，完整 CPU 集成尚未完成。2026-11-16，Windows + CPython 3.12.13；基线为 `2d61a9c8aa6eaf7eb6a928b1331f67418b5bd79b`，以下结果包含 `feat/rfc-cpu-deployment` 工作区未提交的实现。新改动没有远程 CI 结果。

最新基础套件 **268 passed in 90.51s**，零失败、错误或跳过。命令为 `python -m pytest -q tests --ignore=tests/integration --strict-markers --junitxml=../../output/examples-markdown-20261116/cpu.xml --basetemp=../../output/examples-markdown-20261116/pytest`；原始报告仅保存在工作区，CI 报告由 Actions artifact 保存。耗时是单次正确性运行记录，不作性能比较。

Ruff、compileall、6 项 CI 报告校验器测试及 actionlint 通过。最终 wheel 重新构建、独立目录安装，在源码目录外通过 8 项 CLI 检查：公开/内部 help、runtime-info、旧入口拒绝、两类配置 plan；未安装 torch、transformers、triton、flash_attn。Windows console launcher 限制仍在，Linux console 与 18 项真实 router/SSH 集成仍待 CI。

## 统一启动与测试目录整理

公开入口统一为 `serve -f`，查询使用 `serve plan/status/logs/down`；内部 `_worker` 只启动单 worker。CLI 测试验证参数错误、两种拓扑、前台/后台生命周期及管理分发；实际本地 helper→supervisor→HTTP worker 验证内部入口与独立 checkout 的来源检查。Controller 的 router 探测仍为注入，不作真实 router 通过声明。

测试从根目录的分散文件和 `integration/cpu_linux` 合并到按职责组织的 `tests/`。布局、vLLM 对照提交与运行命令见 [测试目录指南](../tests/README.md)。按迁移映射逐项比较，迁移前后均为 **285 个相同用例**：267 个基础用例与 18 个 Linux 集成互不重叠，合并覆盖全部用例。共享数据与进程夹具改从 `tests.helpers` 导入，不从另一个测试模块导入。证据：工作区 `output/serve-unification-20261115/collection-comparison.json`、`organized-cpu.xml`、`wheel-smoke.json`。

统一入口回归期间，Windows ledger 替换问题再次出现；修复前两轮完整测试均为 260 通过、1 失败（92.06s / 88.44s）。随后使用“保持旧 state.json 的读句柄，再原子替换”稳定复现 `WinError 5`。`deployment/state.py::atomic_json` 对 Windows 的 5/32/33 错误最多重试一秒，期间保留旧 ledger；持续拒绝或其他权限错误继续上抛，清理临时文件。6 项新增回归覆盖短暂占用、永久拒绝和实际读句柄；修复后状态/controller 16 项通过，迁移前完整 267 项通过（89.45s），当轮测试目录迁移后 267 项通过（75.50s）。未跳过失败断言，也不宣称解决所有访问拒绝原因。证据：同一工作区目录中的 `atomic-repro-before.json`、`atomic-repro-after.xml`、`cpu-tests.xml`、`cpu-recheck.xml`、`final-cpu.xml`。

## 目录清理复测

`docs/rfcs/` 仅保留两份 RFC 和索引；7 个示例迁至 `examples/serve/`、`examples/deployment/`，当轮同步了 PowerShell 启动脚本路径；现已改为 Markdown 内嵌示例，详见后文。全仓 85 个本地 Markdown 链接存在，18 项 Linux 集成仍仅完成收集。

首次全套为 **240 通过、1 失败 / 72.01s**：`test_controller_manages_real_cpu_worker_after_helpers_exit` 中 supervisor 原子替换 `state.json` 遇到 Windows `WinError 5`（访问拒绝），没有样例路径缺失。该用例单独复测通过（11.99s），随后完整复测得到 241 项通过（83.96s）。当时没有修改或跳过该用例的断言，也没有改动生产状态文件实现；当时未定位触发原因，复测通过不能视为修复；后续复现与修复见下节。失败与复测原始报告分别保存在工作区 `output/rfc-cleanup-20261115/cleanup-junit.xml`、`process-recheck.xml`、`cleanup-rerun-junit.xml`。

## RFC 一致性核查与兼容性修复

RFC-0001/0002 已分开记录历史基线、方案状态、实施状态与验证状态；修正 executor 归属、ACK/forward 顺序、同步阶段接口、namespace、ledger 状态和 STOPPED 记录保留策略。Roadmap 同步当前 CPU 范围；NIXL、显存与真实数据通路的未实现要求仍保留，没有通过缩小原设备 Gate 消除差异。

发现的实际兼容性缺陷是：项目声明 Python `>=3.10,<3.13`，模型指纹与 router 收据却调用了 [Python 3.11 新增的 `hashlib.file_digest`](https://docs.python.org/3.11/library/hashlib.html#hashlib.file_digest)。在 3.12 测试中移除此 API，两个路径均先复现 AttributeError，再改用 `utils/hashing.py::sha256_file` 的 1 MiB 分块 SHA256；原始内容摘要语义保持不变。

针对性回归 2 项通过，并纳入当时 241 项全套结果：模型文件修改会改变指纹；router 收据匹配时通过、二进制被修改后拒绝。独立 wheel 还检查了空文件与 2,250,000 字节跨块文件的摘要。工作区证据为 `output/rfc-cpu-20261115/hash-compat-before.xml`、`hash-compat-after.xml`、`wheel-rfc-review-smoke.json`（不随仓库发布）。这证明具体缺陷已修复，不代表完整 Python 3.10 环境已验收。

此前 `deploy plan` 非法配置退出码从 2 变为 1 的回归已修复，相关用例仍在全套测试中。

## 已验证的行为

| 范围 | 实际执行和证据边界 |
|---|---|
| T2-01 配置规划 | 两类 YAML、严格字段与类型、角色、IP/GPU/端口冲突、确定性参数数组及 CLI 拒绝；`tests/deployment/test_planner.py` |
| T2-02 实例隔离 | instance/generation 参数传递、真实 CPU SharedMemory 不冲突；未证明 CUDA/NCCL 资源隔离 |
| T2-03/04 进程管理 | 固定 helper 请求、实际 detached supervisor/worker、重复启动停止、错误 owner/PID 身份拒绝、子进程清理、脱敏日志、预留冲突、延迟 start 的取消 nonce；`tests/deployment/test_helper.py` |
| controller | 全节点 preflight 先于启动、router 最后启动、generation/模型不匹配拒绝、启动回复丢失后回滚、不可达资源保留后重试、router→D→P 停止顺序；`tests/deployment/test_controller.py` 使用注入的 RPC/HTTP，不是真实网络/router 证据 |
| 实际 CPU 服务链路 | controller→独立 helper→supervisor→实际 HTTP worker；helper 退出后仍能请求，重复 up 不增进程，logs/down 可用；`tests/deployment/test_processes.py` 的 router 探测明确为模拟，未验证 router 转发 |
| T2-06 角色 HTTP | P/D 独立严格 schema、内部凭据、request ID、P generation、有效期、元数据大小；未初始化 connector 不能 ready，最终响应不能带内部 KV 字段；`tests/server/test_stage_http.py` |
| 模型身份 | 对实际文件字节计算权重/config、tokenizer、模板和 RoPE 摘要；测试使用小型字节夹具，不是实际模型；部署 preflight 与启动元数据一致性有校验 |
| 运行包来源 | `runtime-info` 报告实际导入包的路径、Git revision 和代码修改状态；helper 对自身与实际 worker 入口分别检查，拒绝异地包、错误提交和未提交代码。`tests/test_runtime_identity.py` 在临时、已提交 checkout 中验证；模型/进程集成也使用这个检查 |
| T2-09/11 资源协议 | 描述符布局/身份/摘要、D 自有块映射、尾块与 L=1、lease 预算、ACK/读取完成顺序、到期 fence/隔离；`tests/kv/test_kv_protocol.py` |
| 服务周期维护 | 无 HTTP 流量时仍在引擎所有者线程处理过期/ACK；读取未结束不释放，不能 fence 时保留 allocation 并使服务不可用；`tests/server/test_lease_service.py`。已接入下述阶段控制，但没有真实 NIXL 证据 |
| T2-12/13 的 CPU 阶段控制 | `StageEngine` 使用实际 Scheduler/BlockManager；P 源块由 lease 保留，D 预留自己的私有块且不发布到 prefix cache。CPU 数组实际复制与计算事件用于验证导入完成前不 forward、L−1 恢复、原始预算、EOS/length/usage、与完整 Prefill 基线输出一致；`tests/engine/test_stage_engine.py` |
| 阶段异常与回收 | 复制/摘要/ACK/计算失败、客户端断连、已消费 lease 重放、prompt 内容不匹配、预算耗尽均有测试。无法 fence 时保留目标或计算占用块，确认 fence 后才释放；服务保持不可用直到重启。CPU fence 事件不能代替设备 fence |
| RFC-0001 回归 | 既有服务/CLI 用例、实际 CPU worker 退出、Scheduler/BlockManager 边界、抢占、prefix 复用与引用回收 |

`nano-xpu serve plan/up/status/logs/down` 已有入口。`up` 在版本占位未替换时拒绝启动；所有角色完成 preflight 后才启动 worker，再核对其身份与能力并启动 router。生产 P/D 后端尚未实现，`NanoEngine.start()` 明确拒绝 P/D，不会以测试引擎或完整 Prefill 伪装成功。

当前运行来源验收要求 worker 和 helper 从 profile 指定的干净 Git checkout 导入，可通过该 checkout 的 editable 安装配置；没有构建来源信息的独立 wheel 不能通过这一部署检查。文档修改不影响代码清洁状态；`nanoxpu/` 或 `pyproject.toml` 的未提交变更会被拒绝。CLI 帮助与一般 wheel 安装测试不受影响。

## 上游合同核查

本次 router 证据固定为 `vllm-project/router@07c20da66c255ba65a97ab531f9fb5f60413a7d4`，没有使用上游工作区修改。以上 Nano-XPU 结果来自前述基线加本地未提交实现，不是两个已发布版本的兼容性结论。

[`src/routers/http/vllm_pd_router.rs::prepare_prefill_request`](https://github.com/vllm-project/router/blob/07c20da66c255ba65a97ab531f9fb5f60413a7d4/src/routers/http/vllm_pd_router.rs) 同时写入 `max_tokens=1`、`min_tokens=1` 和 `stream=false`，本地 Prefill schema 已纳入这个输入。真实 router 的字段透传、token 预算和不重放行为仍需二进制集成测试。

[`src/protocols/spec.rs::ChatCompletionRequest`](https://github.com/vllm-project/router/blob/07c20da66c255ba65a97ab531f9fb5f60413a7d4/src/protocols/spec.rs) 序列化时还会带上九个默认布尔字段，包括 `logprobs=false`、`ignore_eos=false`、`skip_special_tokens=true`。服务只接受与既有行为一致的固定值；相反值及整数冒充布尔值被拒绝，没有因此开启 reasoning、hidden-state 或 logprob 功能。

`.github/ci/build_router.py` 已编写，固定源码、Cargo.lock 和 Rust 1.95.0，构建后生成二进制 SHA256 与源码提交收据。Rust 版本来自同提交的 [`.buildkite/test-ready.yml`](https://github.com/vllm-project/router/blob/07c20da66c255ba65a97ab531f9fb5f60413a7d4/.buildkite/test-ready.yml)。`tests/integration/` 共 **18 项** Linux 集成用例，**尚未构建或运行，不能标为通过**：

- 8 项真实 router 用例：单/双 worker、P/D metadata/token 预算、P/D 失败或 429 不重放、客户端断连后继续服务。
- 7 项 SSH/controller 用例：严格 host key 正反验证、真实 SSH→helper→supervisor→worker/router、重复 up、独立 CLI 恢复 status/logs/down、preflight/worker/router 启动失败清理、P/D 能力拒绝和有序停止。
- 3 项 router→HTTP StageEngine→CPU 数组用例：L=1/5/17、原始预算、完整 Prefill 基线输出相同、结束后资源归零。P/D 分别运行在同一测试进程中的独立引擎线程，KV 是独立 NumPy 数组，router 为外部真实二进制；没有跨进程 KV 传输结论。

新增 StageEngine HTTP fixture 已在 Windows 实际监听 localhost，直接请求 P、取得 metadata 后请求 D，三组输入全部通过；证据为工作区 `output/rfc-cpu-20261115/stage-http-network.json`（不随仓库发布）。该检查没有运行 router，不能计入上述 Linux 集成通过数。

P/D controller 正向用例在测试内部注入 `cpu-test` 期望能力，与 worker 实际报告一致；同时断言生产 YAML schema 拒绝该值。它验证编排控制，不通过伪报 NIXL 能力或新增生产模拟开关获得成功。合部恢复 CLI 使用正常 profile，P/D 测试 profile 不用于生产 CLI。

工作流已加入必跑 `cpu-integration` job，并纳入 `ci-gate` 的三个必需依赖。SSH 测试密钥和 private state 位于独立临时目录，不在上传路径内；上传限于构建来源、help、日志、安装清单和 JUnit。当前只有静态语法检查，尚无本批托管 CI 结果。

[`src/server.rs::build_app` 的路由注册及 `list_workers_rest`](https://github.com/vllm-project/router/blob/07c20da66c255ba65a97ab531f9fb5f60413a7d4/src/server.rs) 提供 `/health` 和 `/workers`，没有自身 `/ready`。controller 核对完整 worker URL/角色和健康标记；`--health-check-endpoint /ready` 是 router 检查后端 worker 的地址。这项实现当前只经探测夹具验证。

## 必须继续的验收

1. 在 Linux 构建固定源码 router，实际执行上述 18 项集成，记录通过或修复真实失败。当前 Windows 无可用 Linux 环境，collect-only 不是运行证据。
2. 发布前提供最终 diff/PR 预览；用户确认后运行新 CI，核对 job、gate 和制品，再回填托管结果。不得用 skip 处理缺少的 router 或 SSH。
3. 远程 CI 若暴露实现、协议或 Linux 生命周期问题，应修复并复测；不能沿用原 main 的绿色结果覆盖本批修改。
4. CUDA/Qwen、NIXL GPU 传输、TP 和跨机真实 P/D 保持设备 Gate，不能由本报告宣布完成。

## 2026-11-16 包名与测试模块对应整改

源码迁移到 `nanoxpu/`，发行包与 CLI 为 `nano-xpu`；更新所有导入、子进程入口、来源检查和安装元数据。测试使用 server/engine/deployment/kv 对应源码，根级 CLI 与 runtime_identity 测试回到 tests 根目录，共享夹具改为 helpers。迁移前基线为上一节 267 项通过；新命名的完整 CPU 复测 **267 passed / 67.72s**，零失败、错误或跳过；迁移前后 **285 个相同用例**，其中 18 个 Linux 集成仍仅收集。47 个生产 Python 文件逐个与迁移前版本对照，只发生包命名及安装提示替换。新 wheel 独立安装后确认包名、入口、包文件清单和旧包缺失，SamplingParams 导入及 8 项 CLI/plan 检查通过，无加速器依赖。上游来源名称仍保留，实际运行入口不保留旧包别名。

本轮首次完整测试为 **266 通过、1 失败 / 103.64s**：`tests/server/test_app.py::test_real_http_subprocess_and_same_port_restart` 在第二次启动后的退出等待超时（10 秒），服务日志出现 Windows Proactor 的 `WinError 10054` 连接重置。单项诊断复测 1 passed / 3.98s，随后完整复测得到上面的 267 项通过；未修改断言、延长超时或跳过测试。尚未稳定复现或定位该间歇退出问题，复测通过不代表修复，也不能证明包改名是原因。失败报告、日志与复测均保留在工作区 `output/nanoxpu-rename-20261116/`：`cpu.xml`、`pytest/test_real_http_subprocess_and_0/cpu-server.log`、`http-recheck.xml`、`cpu-recheck.xml`；另有 `collection-comparison.json`、`source-equivalence.json`、`wheel-smoke.json`。这些原始产物不放进 RFC 或源码包。

## 2026-11-16 示例全部改为 Markdown

`examples/` 仅保留 `server.md` 和 `deploy.md` 两份文档，分别汇总服务代码与部署配置；YAML、Shell 和 PowerShell 嵌入带 filename 标注的代码块。`tests/helpers/examples.py` 把文档代码提取到临时目录，生产 planner 仍只接收 YAML。CI 源码外的安装包 plan、配置测试、HTTP 请求和 PowerShell 启动均使用提取结果，不维护第二份脚本。

新增目录约束测试后，**286 项可收集：268 项基础 CPU 全部通过（90.51s），18 项 Linux 集成仍未运行**。原有 285 项全部保留；新增一项验证目录仅 Markdown 且 7 份代码均可生成。两份 Bash 代码块通过语法检查，配置启动 wrapper 的参数传递通过；真实模型仍未验证。静态检查、实际 gate 64 种组合及 117 个本地 Markdown 链接通过。证据保存在工作区 `output/examples-markdown-20261116/` 的 `cpu.xml`、`collection-comparison.json`、`shell-checks.json`。前述 Windows 间歇退出超时未在本轮出现，根因仍未定位。

## 示例收敛为两份文档

按最新目录约定，examples 只保留 `server.md`（四段服务启动/请求代码）与 `deploy.md`（三段 YAML）；原 README 和两个子目录已删除。提取器支持一份文档中的多个命名代码块，生成路径保持不变，CI 无需保存独立脚本。7 段代码与合并前逐字一致，19 个章节链接通过。此次运行目录约束、CLI、planner 与 HTTP 相关测试：**98 passed / 14.70s**；没有再次运行完整套件，最近一次完整结果仍为上文的 268 项通过。当前 112 个本地 Markdown 链接与 gate 64 种组合通过。证据：工作区 `output/examples-two-docs-20261116/targeted.xml`、`checks.json`。
