# RFC-0001：nano-xpu serve 服务化与切分支持——先完成本地 CPU 验证

- 状态：方案草案；S0（CPU 服务层）已实现，43 项测试通过；CUDA/TP/真实 Qwen 结果待设备验收。
- 日期：2026-11-13
- 代码证据基线：`40a001f99181321d8b2ab30a0b8a1cfcdba0c9d5`
- 关联：[总体 roadmap](../roadmap.md)、[CUDA 耦合审计](../cuda-coupling-audit.md)

## 1. 建议的首个 Issue

**标题：`feat: 建立 nano-xpu serve 服务入口与切分参数，先完成 CPU 服务链路验证`**

当前只有 `LLM.generate()` 离线调用，无法通过 server 接收请求。增加 `nano-xpu serve <model-path>`，加载本地 Qwen3 模型，提供非流式文本聊天、健康检查和模型查询接口，并给出可复现的启动与调用脚本。

本 Issue 的“混合部署”明确为 **Prefill 与 Decode 在同一个引擎实例中合部执行**。先交付 S0：本机 CPU 验证服务层；真实推理的第一条交付路径是 Linux + NVIDIA CUDA + Qwen3-0.6B + TP=1。多副本共存、Ascend 与跨厂商协同分别在后续阶段推进，不能由这项验收推导出已支持异构混部。

### Issue 工作清单

- [x] 增加 CLI 入口与 server 可选依赖，保留 `from nanoxpu import LLM` 的现有用法。
- [x] 支持模型路径、模型服务名、HTTP 地址、推理参数及 NCCL rendezvous 端口配置。
- [x] 实现单引擎独占执行、就绪状态与正常/异常退出。
- [x] 实现 `/health`、`/ready`、`/v1/models`、非流式 `/v1/chat/completions`。
- [x] 实现输入与显存块容量校验、并发忙碌拒绝和结构化错误（CPU/适配器桩验证，真实 KV 分配待设备验证）。
- [ ] 将本 RFC 示例升级为实测脚本，补齐启动、调用、退出证据。
- [ ] 验证原 `example.py` 与离线 API 无回归，回填第 8 节验收表。

完成定义：在固定环境中，安装后使用一条 `nano-xpu serve` 命令加载 Qwen3-0.6B；ready 后 HTTP 请求返回模型实际生成的文本；SIGTERM 后释放 GPU 与通信资源，可以重新启动。所有验收项有证据后关闭 Issue。

### 当前实施边界：S0 本地 CPU 验证

本轮授权实施 CLI、HTTP server、独占执行线程、可注入引擎边界和 CPU 测试。默认生产入口仍指向 CUDA nano-xpu 引擎；测试引擎仅保存在 `tests/`，不增加 `--backend mock`，无 CUDA 时禁止自动返回模拟文本。CPU 验证不需要下载 Qwen 权重，也不能证明 Qwen 推理、KV 算子或 TP 通信正确。

S0 完成定义：Python 3.12 环境中运行 CLI 帮助与参数校验；通过测试引擎启动真实 HTTP 服务并验证四个路由、并发拒绝、断连后的槽位所有权、故障变为不可用、关闭与重启；所有 CPU 测试通过并记录证据。完整 Issue 仍需 S1 的真实 Qwen 验收。

### 切分方式与验收边界

| 方式 | 含义与配置 | S0（CPU） | 设备验收 |
|---|---|---|---|
| Tensor Parallelism（TP） | 按模型层内张量维度切到多张 GPU；`--tensor-parallel-size 1..8` | 校验范围并确认参数传递，不能证明多卡计算正确 | S1 验证 TP=1 和 TP=2；同时检查注意力头等维度能被 TP 整除 |
| chunked prefill | 调度器按 `--max-num-batched-tokens` 分块处理长 prompt，KV 留在同一实例 | 校验正整数、参数传递；可单独测试原 Scheduler 的分块状态 | S1 对照分块与不分块的真实 Qwen 输出、边界和显存 |
| P/D 分离 | Prefill 与 Decode 放到独立实例，需传 KV 与请求状态 | 不实现；无 `--pd` 开关 | S5 单独设计并验收 KVConnector |

新增 `--max-num-seqs` 控制引擎调度序列上限；它不改变首版 HTTP 单活动请求策略。TP 不能等同于流水线并行（PP）；本 RFC 不添加 PP 或同卡多模型共享显存能力。

目标多卡示例（待 GPU 验收）：

```bash
CUDA_VISIBLE_DEVICES=0,1 nano-xpu serve /models/Qwen3-8B \
  --tensor-parallel-size 2 --max-model-len 8192 \
  --max-num-batched-tokens 2048 --max-num-seqs 32
```

## 2. 基线行为与问题

| 已核查事实 | 代码证据 | 对 server 的影响 |
|---|---|---|
| 未配置命令行入口，依赖中没有 HTTP server | [`pyproject.toml`](../../pyproject.toml) | 当前没有 `nano-xpu serve` 命令 |
| `LLM` 直接继承 `LLMEngine`，`generate()` 同步执行至结束 | [`llm.py`](../../nanoxpu/llm.py)、[`LLMEngine.generate`](../../nanoxpu/engine/llm_engine.py) | 不能直接阻塞 HTTP 事件循环，也不能让多个请求并发修改引擎 |
| 同一 Scheduler/ModelRunner 完成 P+D，KV 在实例内部保存 | [`LLMEngine.step`](../../nanoxpu/engine/llm_engine.py)、[`ModelRunner.allocate_kv_cache`](../../nanoxpu/engine/model_runner.py) | 可以复用合部计算路径，不需要新增 KV 传输 |
| NCCL 总会初始化，地址固定 `localhost:2333`，TP 共享内存名固定 `nanoxpu` | [`ModelRunner.__init__`](../../nanoxpu/engine/model_runner.py) | TP=1 也有端口冲突；后续 TP 多实例还需隔离共享内存 |
| 当前采样禁止 temperature=0 | [`SamplingParams.__post_init__`](../../nanoxpu/sampling_params.py) | 首版显式拒绝零温度，不声称支持 Greedy |
| 模型初始化与 Attention 硬绑定 CUDA、NCCL、flash-attn | [`model_runner.py`](../../nanoxpu/engine/model_runner.py)、[`attention.py`](../../nanoxpu/layers/attention.py) | 首版仅声明 CUDA 支持 |

现有 `generate()` 会在一次调用中提交整个输入列表，执行结束才返回。首版每次只提交一个请求，通过 HTTP 提供可用的合部服务；动态跨请求批处理另行实现。

## 3. 首版范围与取舍

首版支持一个 server 进程、一个模型、一个引擎；TP 参数接受 1–8，S0 仅验证配置传递，S1 在设备上分别验收 TP=1/2；同时仅允许一个生成请求占用引擎。额外生成请求立即返回 HTTP 429，无隐藏无界队列。健康与就绪接口仍可响应。

首版提供 OpenAI 风格的有限聊天接口，仅支持文本 `system/user/assistant` 消息、单个候选 `n=1` 和 `stream=false`。不承诺完整 OpenAI SDK 兼容性。暂不提供 SSE、工具调用、多模态、鉴权、多模型路由、跨请求 continuous batching、P/D 分离或同卡资源共享调度。

选择这一路径是为了先验证 CLI → HTTP → 真实模型 → HTTP 响应的完整链路。相较于在事件循环里直接调用 `generate()`，专属执行线程可以让健康检查继续工作；相较于首版就实现服务进程池，它减少了 IPC、调度及资源恢复状态。HTTP 层使用 FastAPI/Uvicorn，作为 `server` extra；版本需在实现时固定并验证，现有 CUDA 核心依赖保持原状。

## 4. 目标启动接口

以下为真实模型目标命令；S0 不把它标记为已完成设备验收：

```bash
CUDA_VISIBLE_DEVICES=0 nano-xpu serve /models/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 \
  --distributed-port 2333 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager
```

| 参数 | 约定与校验 |
|---|---|
| `<model-path>` | 必填本地目录；由用户提前准备完整权重、配置与 tokenizer，不隐式联网下载 |
| `--served-model-name` | 默认模型目录名；请求 `model` 必须与之匹配 |
| `--host` / `--port` | 默认 `127.0.0.1:8000`；地址/端口无效时启动失败 |
| `--tensor-parallel-size` | 1–8，默认 1；设备验收前不声称多卡可用 |
| `--max-num-batched-tokens` | 正整数，默认 16384；传给 Scheduler 控制 Prefill 分块预算 |
| `--max-num-seqs` | 正整数，默认 512；引擎调度序列上限，不等于 HTTP 并发数 |
| `--distributed-port` | 默认 2333；传入 Config/ModelRunner，不能继续硬编码；冲突时失败并退出 |
| `--max-model-len` | 默认 4096；正整数且不超过模型配置的上限，超出显式报错 |
| `--gpu-memory-utilization` | 默认 0.9，必须在 `(0, 1)`；它是本实例预算，不是多实例隔离保证 |
| `--enforce-eager` | 显式关闭 CUDA Graph；首次验收使用此参数，默认仍沿用引擎行为 |

以 `CUDA_VISIBLE_DEVICES` 选择设备，程序内部继续使用逻辑 rank=0。首版不额外增加可能与该环境变量冲突的 `--device`。模型 dtype 沿用配置并记录，不能把未适配 dtype 自动当作已支持。

已实施包改动（CPU 验证）：为现有 `nano-vllm` distribution 增加 `[project.scripts]` 的 `nano-xpu = "nanoxpu.cli:main"`，不在本 Issue 中重命名 distribution。`nanoxpu.cli` 延迟导入推理与服务模块，`--help` 不加载模型；缺少 server extra 时给出明确安装提示。

## 5. 进程、调用与资源所有权

```text
CLI: 校验启动参数 → 启动 HTTP app（仅一个 web worker，不使用 reload）
                              │ startup
                              ▼
                    专属 engine 执行线程
                    创建 LLM / tokenizer / CUDA / NCCL
                              │ 初始化成功后 ready=true
HTTP chat → 校验 schema → 尝试获取唯一生成槽位
                              │ 成功
                              ▼
                 同一个执行线程：chat template → token IDs
                    → 长度/容量校验 → LLM.generate
                    → Decode 输出与 usage → HTTP 响应
                              │ finally
                              ▼
                         释放生成槽位
```

### 5.1 已实现模块与设备验证边界

| 模块 | 职责与调用时机 |
|---|---|
| `nanoxpu/cli.py`（新增，CPU 已验证） | 解析 `serve` 参数，构造 ServerConfig/引擎 Config，启动单 HTTP worker |
| `nanoxpu/server/app.py`（新增，CPU 已验证） | lifespan、路由、schema 校验、错误映射、唯一生成槽位；禁止调用进程全局 `generate()` |
| `nanoxpu/server/engine.py`（新增，CPU 已验证） | 持有单线程执行器；`start()` 创建引擎，`complete(request)` 执行完整生成，`close()` 在同线程释放引擎 |
| `nanoxpu/config.py` / `engine/model_runner.py` | 引入并消费 `distributed_port`，保持离线默认 2333 |
| `engine/llm_engine.py` | 为显式关闭补幂等保护，使正常 shutdown 与已有 atexit 不重复释放 |
| `pyproject.toml` | CLI 入口、server extra；已有离线导入路径保持兼容 |

`start()` 的前置条件是配置校验成功；后置条件是模型权重、KV、warmup（及启用时的 Graph）已完成，才能置 ready。初始化失败以非零退出码结束，不报告 ready。

`complete()` 的前置条件是 ready 且已持有唯一槽位。同一专属线程完成 tokenizer 与引擎操作；从模板直接生成 token IDs 传给 `generate()`，避免二次 tokenize 或重复加入特殊 token。Qwen3 首版固定 `enable_thinking=False`、`add_generation_prompt=True`，示例调用与离线对照使用相同设置。

除检查 `prompt_tokens + max_tokens <= max_model_len` 外，还须按当前 `block_size` 检查最坏情况下所需 KV blocks 不超过本实例总 blocks；不得把永远无法分配的请求留给 Scheduler 触发断言。成功后该请求的活动序列与引用被释放；允许已释放 block 上的 prefix cache 数据留存，不能将缓存保留误判为活动资源泄漏。

关键调度伪代码：

```python
async def chat(request):
    validate_supported_fields(request)
    if not ready:
        return error(503, "engine_not_ready")
    if not try_acquire_generation_slot():
        return error(429, "engine_busy")
    # 槽位所有权交给后台任务：即使 HTTP handler 因断连退出也不能提前释放。
    task = create_task(run_complete_on_engine_thread(request))
    task.add_done_callback(release_slot_and_observe_failure)
    return await shield(task)
```

这是调用顺序说明，不是可复制的实现。实现需保证槽位获取在同一事件循环中原子执行；通过 `max_workers=1` 的专用 executor 串行执行，避免默认线程池将 CUDA 调用分发到不同线程。

### 5.2 故障与关闭

- schema、模型名、长度或容量错误：在生成前拒绝，不修改 Scheduler 状态，下一请求可继续。
- 请求断连：首版继续完成已开始的生成并丢弃返回值，完成后才释放槽位；不宣称立即取消 GPU 运算。首版不提供执行中的硬超时。
- 模型执行异常（含 OOM）：将引擎标记 unhealthy/ready=false，当前请求返回 500，后续生成返回 503；触发受控关闭并以非零码退出，由外部重新拉起，不在可能污染的 Scheduler 上继续生成。
- SIGTERM/SIGINT：ready=false，拒绝新生成，最多等待活动任务 30 秒，随后在所属线程关闭引擎。超时则由 CLI 的退出 watchdog 记录异常并强制非零退出，避免 executor 等待卡住的 GPU 线程而永远无法结束；强制退出依赖 OS 回收进程资源，不能声称仍执行了 Python 清理。S0 的测试引擎不创建 TP 子进程；TP>1 的强制退出与子进程资源回收尚待 S1 实现和设备验证。
- `close()` 幂等；正常退出显式销毁通信组并释放模型引用。TP 子进程退出/超时机制在 S1 的 TP=2 Gate 验证，未通过前不能声明多卡服务已交付。

## 6. HTTP 契约

| 路由 | 返回行为 |
|---|---|
| `GET /health` | 活动 HTTP 进程且无致命引擎故障时 200；致命故障后 503 |
| `GET /ready` | 模型可接受请求时 200，否则 503；生成忙碌仍可返回 200，忙碌由 429 表达 |
| `GET /v1/models` | 返回唯一 served-model-name，模型可服务前返回 503 |
| `POST /v1/chat/completions` | 返回单个文本 completion、finish_reason、usage；首版非流式 |

请求样例：

```json
{
  "model": "qwen3-0.6b",
  "messages": [{"role": "user", "content": "请用一句话介绍你自己。"}],
  "temperature": 0.6,
  "max_tokens": 64,
  "stream": false
}
```

请求默认 `temperature=0.6`、`max_tokens=64`、`stream=false`、`n=1`。temperature 必须有限且大于 `1e-10`；max_tokens 为正整数。messages 必须非空，content 仅支持字符串，模板产生的 prompt 不得为空。`temperature=0`、`stream=true`、`n!=1`、多模态 content、tools/stop 等未知字段均明确返回 400，不能静默忽略。

成功响应包含 `id`、`object="chat.completion"`、`created`（Unix 秒）、`model`、`choices[0].message={role:"assistant",content:...}`、`choices[0].index=0`、`finish_reason` 和 `usage`。usage 使用实际模板输入 token 数和实际生成 token 数，total 为二者之和，不用字符数估算；遇 EOS 为 `stop`，到 max_tokens 为 `length`，同时满足时优先 `stop`。

错误统一为 `{"error":{"message":"...","type":"...","code":"..."}}`。非法参数返回 400，模型名不匹配返回 404，忙碌返回 429（附 `Retry-After: 1`），尚未就绪返回 503，推理失败返回 500。内部堆栈写日志，不放入响应。

## 7. 示例脚本与执行方式

本目录附两份接口示例；调用脚本已对 CPU 测试服务执行，真实模型启动脚本仍待 CUDA 设备验收：

- [`examples/serve-qwen.sh`](examples/serve-qwen.sh)：前台启动，用 `exec` 保持终止信号直达服务进程。
- [`examples/request-qwen.sh`](examples/request-qwen.sh)：有限时间等待 ready，调用真实聊天接口，验证响应结构。

目标环境为 Linux + NVIDIA CUDA；开发机 Windows 上完成文档检查不算设备验收。Python 遵循当前项目 `>=3.10,<3.13` 限制。模型下载、CUDA/PyTorch/flash-attn 配套安装在运行前完成，实测后记录完整版本与 GPU 型号。

```bash
# 真实 CUDA 环境的安装与启动方式，尚未设备验收；CPU 测试请看 README-cpu.md。
python -m pip install -e '.[server]'

# 终端 1，位于仓库根目录：
MODEL_PATH=/models/Qwen3-0.6B bash docs/rfcs/examples/serve-qwen.sh

# 终端 2：
bash docs/rfcs/examples/request-qwen.sh
```

脚本默认仅绑定 loopback。外部访问由使用者显式配置 HOST；首版不包含公网部署所需的访问控制。示例的最大等待时间用于客户端启动探测，不等于推理取消机制。

## 8. 实施顺序与验收

### S0 本地 CPU 验收结果

| 用例 | 验证场景与判定 | 实际结果 |
|---|---|---|
| CPU-01 | 不安装 torch/transformers/triton/flash-attn；CLI help 与服务模块仍可导入 | 通过；见下方测试报告 |
| CPU-02 | 非法 TP、token 预算、显存比例、端口、模型路径被 CLI 拒绝；合法参数传给引擎 | 通过；见下方测试报告 |
| CPU-03 | 注入测试引擎；models/chat/health/ready 正常，响应格式与 usage 正确 | 通过；见下方测试报告 |
| CPU-04 | 非法请求 400、错误模型 404、忙碌 429、故障后 503；合法请求不受输入错误污染 | 通过；见下方测试报告 |
| CPU-05 | 请求取消后线程仍执行，槽位保持占用，完成后可接受下一请求 | 通过；见下方测试报告 |
| CPU-06 | 引擎初始化/生成/关闭在同一线程；关闭一次；启动失败能报告错误 | 通过；见下方测试报告 |
| CPU-07 | 子进程真实监听 localhost，调用示例可用；正常停止后可同端口重启 | 通过；见下方测试报告 |
| CPU-08 | 原 Scheduler 在 CPU 下分块推进；仅验证调度状态，不验证 Attention 或数值 | 通过；见下方测试报告 |

2026-11-13，在 Windows + CPython 3.12.13 的独立依赖目录中执行：**43 passed in 6.39s**，无 CUDA 推理依赖。证据：[JUnit XML](cpu-test-results.xml)、[验证说明](README-cpu.md)、[结果报告](cpu-validation.md)。其中 CPU-07 使用真实 HTTP 子进程，测试了同端口重启；watchdog 的强制非零退出另有子进程测试。Windows `nano-xpu.exe` 被系统 Application Control 拦截，已验证安装入口元数据及 `python -m nanoxpu` 的等价 CLI 路径，不把 launcher 标为通过。以下 TC-01–08 保留真实 CUDA 验收，不用 CPU 结果替代。

| 步骤 | 产物 | 依赖与完成条件 |
|---|---|---|
| S1.1 CLI 与配置 | CLI、server extra、distributed-port 参数、参数帮助 | `nano-xpu serve --help` 无模型加载；非法配置非零退出 |
| S1.2 引擎生命周期 | 专属执行线程、start/complete/close、幂等退出 | 初始化前不就绪；关闭可重启；离线 API 保持兼容 |
| S1.3 HTTP 合部链路 | 四个路由、模板处理、容量校验、单请求槽位与错误 | 依赖 S1.2；真实 Qwen 请求返回 completion |
| S1.4 交付验证 | 两份实测脚本、环境清单、验收结果与日志 | 依赖以上步骤；回填下表后完成 Issue |

| 用例 | 输入/执行 | 预期判定 | 实际结果与证据 |
|---|---|---|---|
| TC-01 单卡拉起 | 固定 Qwen3-0.6B 快照，按示例启动 | 模型仅加载一次，ready=200；调用返回实际文本与正确 usage | 未执行，待回填 |
| TC-02 离线对照 | 同模型、依赖、模板、种子与采样参数，对照离线 token IDs | 确认模板输入一致；受控随机条件下比对输出，存在非确定性须定位并记录，不凭文本像样判定正确 | 未执行，待回填 |
| TC-03 边界校验 | 空消息、错模型名、temperature=0、stream=true、未知字段、超上下文/总 KV 容量 | 按契约拒绝，下一条合法请求成功；无 Scheduler 断言 | 未执行，待回填 |
| TC-04 并发与断连 | 长请求运行时发第二条请求、探测健康；断开第一条连接 | 第二条 429，健康可响应；断连后至 GPU 任务完成前仍忙，随后恢复 | 未执行，待回填 |
| TC-05 启动失败 | 不存在的模型、占用 HTTP/NCCL 端口、不可用 CUDA | 非零退出，明确报错，不报告 ready，不残留服务进程 | 未执行，待回填 |
| TC-06 退出重启 | 正常请求后 SIGTERM，再以同端口启动 | 正常关闭无重复销毁，GPU/端口释放，可再次请求；另测活动任务超过 30 秒的退出路径 | 未执行，待回填 |
| TC-07 执行故障 | 在引擎执行边界注入异常，补真实设备 OOM 场景 | 当前 500、后续 503，非零退出，重新启动后恢复 | 未执行，待回填 |
| TC-08 连续请求与回归 | 连续 100 次合法请求，再运行原 example.py | 活动序列/块引用归零，无逐请求显存增长；原离线 API 正常 | 未执行，待回填 |

验收记录包含实现 commit、GPU/显存、驱动、CUDA、Python、torch、transformers、triton、flash-attn 版本，模型与 tokenizer 快照标识、完整命令、响应与日志路径。首版无吞吐/延迟硬目标，可记录单请求端到端耗时（ms）作为后续基线，不声称性能收益。

## 9. 服务化 Roadmap

| 阶段 | 可交付能力 | 前置与 Gate |
|---|---|---|
| **S0：CPU 服务层验证（已完成）** | CLI、HTTP、测试引擎注入、异常与生命周期、切分参数 | CPU-01–08；测试引擎不进入生产入口 |
| **S1：CUDA 单实例 P/D 合部与 TP 验收** | `nano-xpu serve` + Qwen3 TP=1/2 + chunked prefill + 非流式聊天 | 第 8 节验收；先完成这一条端到端路径 |
| **S2：同机多实例隔离** | 显式 instance-id、共享内存命名隔离、TP worker 生命周期；不同 GPU 分别启动副本 | S1 完成；两实例 HTTP/NCCL/共享内存无冲突，停止一个不影响另一个；TP=2 单实例先在 S1 验收 |
| **S3：在线请求调度** | 有界排队、取消/超时、SSE、跨请求 continuous batching 与基础指标 | S1/S2 证据稳定；逐项定义协议与负载验收；替换首版单请求拒绝策略 |
| **S4：Ascend P/D 合部服务** | 复用 serve 接口接入 Ascend 后端，先 A2 后 A3 | 总体 roadmap Phase 1 的实际设备 Gate；HTTP 可用不能替代算子和数值正确性 |
| **S5：P/D 分离及异构部署** | 独立 P/D worker、RequestState、KVConnector，逐步同代→跨代→跨厂商 | 总体 roadmap Phase 2–4；须新增 KV 传输与接续正确性证据 |

S 阶段描述服务化交付，既有 Phase 描述异构运行时能力，两者不是同一套编号。S0 已完成 CPU 验证，下一阶段进入 S1；不要求先实现完整 Platform/Backend 抽象才交付 CUDA HTTP 链路。S2/S3 可拆后续 Issue，S4/S5 的接口按当时后端证据细化。

## 10. 反串讲与尚待验证事项

以“请用一句话介绍你自己”为例：HTTP 层接收文本并取得唯一生成槽位，专属线程中的 tokenizer 产生带 Qwen chat template 的 token IDs；同一线程的 LLMEngine 将序列放入 waiting，Prefill 写入实例本地 KV，随后 Decode 读取这些 block 逐 token 生成。结束后释放活动 block 引用，将输出和实际 token 计数交给 HTTP 层。客户端断开不改变 KV 所有权；计算结束前槽位仍归这条任务持有。

**为什么无需 KVConnector？** P 与 D 都由同一 ModelRunner 执行，本地 KV 没有发生跨实例交接。**首版为何不直接接受并发生成？** Scheduler、Sequence 和全局 context 均未提供 HTTP 并发访问契约；串行所有权让已有状态机保持一致，代价是忙碌拒绝与吞吐受限。**怎样证伪方案？** 通过并发/断连测试观察是否提前释放槽位，通过连续请求和重启测试验证是否存在活动资源泄漏；失败则不能关闭 Issue。

待设备验证：专属线程内 CUDA/NCCL 初始化与退出的实际行为、所选 Qwen3 快照的模板兼容性、默认上下文长度下的显存预算、固定种子离线对照的确定性。若专属线程生命周期验证失败，应在 S1.2 改为独占 engine 子进程并同步故障与 IPC 设计，再进入后续验收。

## 11. 设计依据与本地验证记录

服务生命周期采用 [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)，CPU 接口测试参考 [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)，请求字段使用 [Pydantic strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/) 避免把字符串或布尔值隐式转为数值参数。

本 RFC 已回填 CPU 验证结果。本机 Python 3.12 可直接运行；新建虚拟环境的启动器被 Windows Application Control 拒绝，因此本轮使用可运行的 Python 3.12 原解释器与仓库内独立依赖目录，不修改系统策略。
