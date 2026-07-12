# CUDA / 平台耦合审计清单

> **Phase 0 / Week 1 必交付物**。逐点拆除 nano-vllm 与 CUDA 的强绑定，为 `Platform` / `AttentionBackend` / `KVCacheBackend` / `CommBackend` 抽象提供改造依据。
>
> - 证据基线：nano-vllm 源码（fork 基线 commit `bb823b3`，以实际固定 commit 为准）。
> - 每条带 **文件:行号** 证据，标注 **抽象目标 / 复用层级（L0–L4，见 roadmap 4.1）/ 目标阶段 / Ascend 降级**。
> - 与 [roadmap.md](roadmap.md) 第 4–6、10 节交叉引用。

## 总览改造矩阵

| 类别 | 文件 | 绑定数 | 抽象目标 | 复用层级 | 目标阶段 |
|---|---|---|---|---|---|
| 设备运行时 | `engine/model_runner.py` | 7+ | `Platform` | — | Week 1 |
| CUDA Graph | `engine/model_runner.py` | 3 | `Platform.graph`（降级 `enforce_eager`） | — | Week 1 |
| Attention | `layers/attention.py` | 3 | `AttentionBackend` | L1–L4 | Week 1（KV Write）/ Phase 1（attn） |
| 采样 | `layers/sampler.py`、`sampling_params.py` | 2 | `OperatorBackend` + **新增 Greedy** | 自研 | Week 1 |
| 通信 | `layers/linear.py`、`layers/embed_head.py`、`model_runner.py` | 8 | `CommBackend`（nccl/hccl + rank 注入） | — | Week 1 / Phase 1 |
| Host→Device | `engine/model_runner.py` | 9 | `platform.device` | — | Week 1 |
| 配置 / 依赖 | `config.py`、`pyproject.toml` | 3+ | device-agnostic 命名 + optional extras | — | Week 1 |

---

## 1. 设备运行时 → `Platform`

> 目标：所有 `torch.cuda.*` 设备/内存 API 收口到 `Platform`，由 `HardwareProfile` 选择 `CudaPlatform` / `AscendPlatform`。能力点：`set_device` / `device` / `mem_get_info` / `memory_stats` / `empty_cache` / `synchronize` / `set_default_device`（见 roadmap 第 5 节接口表）。

| 位置 | 现状 | 目标 | Ascend 降级 |
|---|---|---|---|
| `engine/model_runner.py:27` | `torch.cuda.set_device(rank)` | `platform.set_device(rank)` | `torch.npu.set_device` |
| `engine/model_runner.py:30` | `torch.set_default_device("cuda")` | `platform.set_default_device()` | `"npu"` |
| `engine/model_runner.py:58` | `torch.cuda.synchronize()`（`exit`） | `platform.synchronize()` | `torch.npu.synchronize` |
| `engine/model_runner.py:92,101` | `torch.cuda.empty_cache()`（warmup） | `platform.empty_cache()` | `torch.npu.empty_cache` |
| `engine/model_runner.py:93` | `torch.cuda.reset_peak_memory_stats()` | `platform.reset_peak_memory_stats()` | `torch.npu.reset_peak_memory_stats`（若存在，否则 no-op） |
| `engine/model_runner.py:106` | `torch.cuda.mem_get_info()` | `platform.mem_get_info()` | `torch.npu.mem_get_info` |
| `engine/model_runner.py:108-109` | `torch.cuda.memory_stats()[...]` | `platform.memory_stats()` | `torch.npu.memory_stats` |

---

## 2. CUDA Graph → `Platform.graph`（Week 1 降级 `enforce_eager`）

> 现状：`ModelRunner.__init__` 在 `enforce_eager=False`（默认）时于构造期即调用 `capture_cudagraph`（`model_runner.py:34→37`），Ascend 上 `torch.cuda.CUDAGraph` 不可用，**会在实例化阶段直接失败**。

| 位置 | 现状 | 目标 | Ascend 降级 |
|---|---|---|---|
| `engine/model_runner.py:34,37` | 构造期无条件 `capture_cudagraph()`（受 `enforce_eager` 守卫，但默认 False） | `if platform.supports_graph() and not enforce_eager:` | Ascend `supports_graph()=False` → `enforce_eager=True` 跳过 |
| `engine/model_runner.py:223-248` | `capture_cudagraph` 全程 `torch.cuda.CUDAGraph` | `platform.capture_graph(...)` / `platform.replay_graph(...)` | 整段跳过 |
| `engine/model_runner.py:239` | `torch.cuda.CUDAGraph()` | `platform.create_graph()` | — |
| `engine/model_runner.py:242` | `torch.cuda.graph(graph, self.graph_pool)` | `platform.graph_capture_ctx(graph, pool)` | — |
| `engine/model_runner.py:247` | `torch.cuda.synchronize()` | `platform.synchronize()` | — |

**Week 1 最小动作**：让 Ascend 走 `enforce_eager=True` 即可绕过整段 Graph，不必在 Week 1 实现 NPU Graph 捕获。

---

## 3. Attention → `AttentionBackend`

> 现状：`attention.py` 顶层硬 `import flash_attn`（`Dao-AILab` CUDA-only 包），prefill/decode 均走 flash-attn。Ascend 不支持独立 `flash_attn`。`store_kvcache` 是唯一已是 Triton 的算子，可经 Triton-Ascend 复用。

| 位置 | 现状 | 目标 | 复用层级 / Ascend 来源 |
|---|---|---|---|
| `layers/attention.py:6` | `from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache`（顶层硬 import） | 收进 `FlashAttnBackend`，顶层不再硬 import；`flash-attn` 降为可选 | — |
| `layers/attention.py:67` | prefill `flash_attn_varlen_func(...)` | `AttentionBackend.prefill(...)` | Ascend：`torch_npu` 推理 FA `npu_fused_infer_attention_score` / vLLM-Ascend DFlash（**非 Triton**） |
| `layers/attention.py:72` | decode `flash_attn_with_kvcache(...)` | `AttentionBackend.decode(...)` | Ascend：vLLM-Ascend native paged attention / `torch_npu` 推理 FA |
| `layers/attention.py:3-4,10-40` | `store_kvcache_kernel`（Triton） | `KVCacheBackend.write(...)` | **L1–L2**：经 Triton-Ascend 重编译，改 Launcher/Layout |

> ⚠️ **陷阱**：`torch_npu` 的 `npu_fusion_attention` 面向**训练**（含 dropout/keep_prob），推理须用 `npu_fused_infer_attention_score`（torchair）。直接复用训练算子在 A2/A3 会算子不可用或语义不符。

---

## 4. 采样 → `OperatorBackend`（含新增 Greedy）

> 现状：采样器是 Gumbel-max 随机采样且 `@torch.compile`，`SamplingParams` 显式禁止 greedy。但 Phase 0 Golden / 所有 Phase Gate 都依赖确定性 Greedy 基准 —— **Greedy 是必须新增的能力，非复用**。

| 位置 | 现状 | 目标 | 备注 |
|---|---|---|---|
| `layers/sampler.py:7` | `@torch.compile` | 经 `OperatorBackend` 在 Ascend 替换为 `torch_npu.compile` 或回退 eager | compile 后端是 RMSNorm/RoPE/SiLU/Sampler 共同绑定点 |
| `layers/sampler.py:8-11` | Gumbel-max 随机采样（`exponential_(1)` + `argmax`） | **新增确定性 Greedy 路径**：`temperature==0 → logits.argmax` | 随机路径保留；Greedy 作默认基准 |
| `sampling_params.py:11` | `assert self.temperature > 1e-10`（禁止 greedy） | 允许 `temperature==0` 触发 Greedy | 解除 assert |

---

## 5. 通信 → `CommBackend`（`dist.*` → nccl/hccl + rank 注入）

> 现状：`dist.*` 贯穿构造期与 forward。Ascend 切 hccl 时，**不止 `init_process_group`，下列每点都会逐个失败**。目标抽象为 `CommBackend`，backend 由 `HardwareProfile` 决定；构造期 `tp_rank`/`tp_size` 改由 `config`/`HardwareProfile` 注入（避免未初始化 dist 时无法构造）。

| 位置 | 现状 | 目标 |
|---|---|---|
| `engine/model_runner.py:26` | `dist.init_process_group("nccl", ...)` | `CommBackend.init(backend=platform.distributed_backend())`（nccl/hccl） |
| `layers/linear.py:23` | `self.tp_rank = dist.get_rank()`（`LinearBase.__init__`） | 从 `config`/`HardwareProfile` 注入 `tp_rank` |
| `layers/linear.py:24` | `self.tp_size = dist.get_world_size()` | 注入 `tp_size` |
| `layers/linear.py:62,106,139` | `dist.get_world_size()`（Column/QKV/Row 构造期） | 注入 `tp_size` |
| `layers/linear.py:155` | `dist.all_reduce(y)`（`RowParallelLinear.forward`） | `comm_backend.all_reduce(y)` |
| `layers/embed_head.py:17,18` | `dist.get_rank()` / `get_world_size()`（`VocabParallelEmbedding.__init__`） | 注入 |
| `layers/embed_head.py:41` | `dist.all_reduce(y)`（forward） | `comm_backend.all_reduce(y)` |
| `layers/embed_head.py:63-65` | `dist.gather(logits, all_logits, 0)` + `torch.cat`（`ParallelLMHead.forward`） | `comm_backend.gather(...)` |

---

## 6. Host→Device 拷贝 → `platform.device`

> 现状：`prepare_prefill` / `prepare_decode` / `prepare_sample` / `prepare_block_tables` 里大量 `.cuda(non_blocking=True)` 硬编码目标设备。目标统一为 `.to(platform.device, non_blocking=True)`。

| 位置 | 行 |
|---|---|
| `engine/model_runner.py:126` | `prepare_block_tables`：`block_tables` |
| `engine/model_runner.py:164-168` | `prepare_prefill`：`input_ids` / `positions` / `cu_seqlens_q` / `cu_seqlens_k` / `slot_mapping` |
| `engine/model_runner.py:182-185` | `prepare_decode`：`input_ids` / `positions` / `slot_mapping` / `context_lens` |
| `engine/model_runner.py:192` | `prepare_sample`：`temperatures` |

---

## 7. 配置与依赖

| 位置 | 现状 | 目标 |
|---|---|---|
| `config.py:12` | `gpu_memory_utilization` | device-agnostic 命名（如 `memory_utilization`），保留别名 |
| `config.py:13` | `tensor_parallel_size` | 保留（语义中性） |
| `config.py:17,22` | `kvcache_block_size=256`，`assert % 256 == 0` | 保留；作为 Golden 矩阵 Block 边界 255/256/257 的依据 |
| `pyproject.toml:18` | `"flash-attn"`（无版本约束硬依赖，Ascend 无 wheel，`pip install` 直接失败） | 移入 `[project.optional-dependencies] cuda=[...]` |
| `pyproject.toml:16` | `"triton>=3.0.0"` | cuda extras 保留 `triton`；ascend extras 用 `triton-ascend` |
| — | core 依赖 | 仅留 `torch` / `transformers` / `xxhash` |

---

## Week 1 最小解除范围（"四类硬绑定最小 diff"）

roadmap Week 1 要求的"解除 flash-attn / torch.compile / NCCL / CUDA Graph 四类硬绑定的最小 diff"对应：

1. **flash-attn**：`attention.py:6` 顶层 import 改为延迟/可选；`pyproject.toml` flash-attn 移入 cuda extras。
2. **torch.compile**：`sampler.py:7`（及 RMSNorm/RoPE/SiLU 的 compile）经 `OperatorBackend` 选择后端或回退 eager。
3. **NCCL**：`model_runner.py:26` backend 可配置；`linear.py:23` / `embed_head.py:17` 的 `dist.get_rank()` 改注入。
4. **CUDA Graph**：Ascend `supports_graph()=False` → `enforce_eager=True` 跳过 `capture_cudagraph`。

> 配合新增 **Greedy 路径**（第 4 节），Week 1 结束时 CUDA baseline 零回退，且 Ascend 可加载 `AscendPlatform` skeleton 并解除上述四类硬绑定（真机数值对齐属 Phase 1）。

## 暂不在范围（Phase 1+）

- Attention / KV 的 Ascend 数值对齐与候选算子落地（Phase 1）。
- `CommBackend` 在 hccl 下的真机多卡验证（Phase 1）。
- NPU Graph 捕获（Phase 1，先全程 eager）。
- `LLMEngine` 拆分为 Router + 多 `StageEngine`（Phase 2，Week 1 仅预留接口）。
