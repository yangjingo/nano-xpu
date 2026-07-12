# Nano-XPU Roadmap

> **面向跨代际、跨厂商加速器与阶段解耦的大模型异构推理运行时。**

Nano-XPU 基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 演进。项目不追求一开始完整替代 vLLM 或 SGLang，而是优先形成一个：

- 架构清晰、易于理解和修改；
- 最大程度复用社区已有算子与后端；
- 能真实跑通 Ascend A2/A3 和跨厂商 PD；
- 数据协议可验证、后端可扩展；
- 适合作为异构推理研究与工程验证底座；

的轻量级推理运行时。

> 核心工程原则：**Nano-XPU 先成为社区算子的集成器与异构运行时，再成为性能算子的创新者。**

---

## 1. 项目定位

> **让不同代际、不同厂商的 AI 加速器，在最适合的推理阶段协同工作。**

项目重点不是"同一套代码分别运行在不同设备"，而是让不同硬件池共同完成同一个模型请求：

```text
Ascend A3 Prefill → Ascend A2 Decode
RTX 5090   Prefill → Ascend A2 Decode      （正确性与协议验证 Case）
NVIDIA     Encoder → Ascend Prefill → Ascend Decode
```

当前优先级：

```text
PD → EPD → AFD
```

---

## 2. 阶段解耦架构

| 架构 | 全称 | 适用模型 | 阶段图 | 核心传输对象 |
|---|---|---|---|---|
| **PD** | Prefill–Decode | 文本 LLM | `P → D` | KV Cache、请求状态 |
| **EPD** | Encode–Prefill–Decode | 含独立 Vision / Audio / Video Encoder 的多模态模型 | `E → P → D` | Feature、KV Cache |
| **AFD** | Attention–FFN Disaggregation | Dense Transformer、MoE | `A ↔ F` | Activation、路由元数据 |

> **EPD 的边界**：`E` 指多模态编码器，不是文本 LLM 的某种额外阶段。含 Encoder 的请求统一走 EPD；其内部 LM 段的 `P→D` 复用 `KVConnector`，但 StageContract 必须携带 multimodal RequestState 标记，与纯文本 PD 区分。

三类边界各有一条数据通路，分别用独立 Connector 承载，**禁止用一个通用 Connector 模糊数据语义**：

```text
E → P ：Multimodal Feature（embedding、grid/position metadata、modality 类型） → FeatureConnector / FeatureDescriptor
P → D ：KV Cache（K/V、序列状态、采样状态、Block 描述）                        → KVConnector / KVDescriptor
A ↔ F ：Activation（每层、每 microbatch、每 decode step）                        → ActivationConnector / ActivationDescriptor
```

采样状态、Block 描述归入 `KVDescriptor` 的子字段（见 4.6）。

---

## 3. 总体演进路径

### 主线 A：文本 LLM 的 PD

```text
A2 单体
  → A2(P) → A2(D)
  → A3 单体
  → A3(P) → A3(D)
  → A3(P) → A2(D)
  → RTX 5090(P) → Ascend A2(D)
  → 数据中心 NVIDIA(P) → Ascend(D)
```

> **RTX 5090 + Ascend 仅作跨厂商协议与正确性验证 Case**：消费卡与数据中心 GPU 特性差异大，不代表性能收益结论，且 BF16 跨厂商 FlashAttention 累加顺序不同必有数值漂移。数值漂移超容差的降级方案见 [9.7 Risk Register](#97-risk-register)。

### 主线 B：多模态模型的 EPD

```text
多模态单体 → E/PD → E/P/D → A2+A3 跨代 EPD → NVIDIA+Ascend 跨厂商 EPD
```

### 方向 C：AFD

AFD 涉及每层、每个 Microbatch 和每个 Decode Step 的 Activation 交换，对互联带宽、阶段配比和模型结构要求更高，因此在 PD、EPD 稳定后推进。

---

## 4. 核心设计原则

### 4.1 社区优先：先集成，再适配，最后自研

不以重新实现 Attention / RMSNorm / RoPE / KV Cache 等算子为第一目标。复用优先级：

```text
L0  原样复用社区已有实现
 ↓
L1  增加薄适配层（参数、Shape、Layout、Launcher）—— 即 4.2 定义的 OperatorBackend / AttentionBackend / KVCacheBackend 的实现
 ↓
L2  对社区实现做少量兼容性 Patch
 ↓
L3  Fork 社区 Kernel 做设备专项适配
 ↓
L4  Profiling 证明存在瓶颈后，才自研算子
```

第一阶段优先调研与复用：nano-vllm 已有实现 → Triton / Triton-Ascend 已适配算子 → FlagGems 通用 Triton 算子 → SGLang / vLLM 社区 Attention 与 KV Kernel → vLLM-Ascend、`torch_npu` 等 Ascend 原生社区实现 → PyTorch / `torch_npu` Reference 实现作为正确性兜底。

> **代码现状约束**：经核实，nano-vllm 中**唯一**可在 Ascend 经 Triton-Ascend 重编译复用的是 KV Cache Write（`store_kvcache`，L1–L2）。Prefill/Decode Attention 当前走 Dao-AILab CUDA-only `flash_attn` 包（`attention.py:6/67/72`），**Ascend 上没有可直接编译的 Triton attention 候选**，须复用 `torch_npu` 推理 FA / vLLM-Ascend DFlash（见第 6 节）。

### 4.2 统一算子语义，而不是强制统一 Kernel

统一的是接口与语义（`OperatorBackend` / `AttentionBackend` / `KVCacheBackend`），不同硬件可选择不同社区实现：

```text
AttentionBackend
├── FlashAttnBackend        （CUDA，nano-vllm 原路径）
├── TritonCudaBackend       （NVIDIA 侧探索性）
├── TritonAscendBackend     （Ascend 侧探索性，目前仅 KV Write 可用）
├── AscendNativeBackend     （torch_npu 推理 FA / vLLM-Ascend DFlash，Ascend 首选）
└── ReferenceBackend        （PyTorch / torch_npu eager，正确性兜底，CPU/无 NPU CI 可跑）
```

Triton / Triton-Ascend 是优先复用路线，但**不要求所有设备强制使用同一份 Kernel**。

### 4.3 先正确，再高性能

MVP 可接受：`BF16 / TP=1 / Batch=1 / Eager Mode / Host Staging / TCP / Canonical KV`。
暂不优先：`RDMA / Device Direct / FP8·INT8 KV / Continuous Batching / 跨 TP KV 重分片 / Graph Mode / 极致 Kernel 性能`。

> **Greedy 是新增能力，非复用**：nano-vllm 现采样器是 Gumbel-max 随机采样（`sampler.py:7-11`，`@torch.compile` 装饰），且 `sampling_params.py:11` 显式 `assert temperature>1e-10` 禁止 greedy。Phase 0 Golden 依赖的确定性 Greedy 路径（`temperature==0 → logits.argmax`）必须新增，并随 `OperatorBackend` / `Platform` 把 `@torch.compile` 在 Ascend 替换为 `torch_npu.compile` 或回退。

### 4.4 先同代，再跨代，最后跨厂商

```text
A2(P) → A2(D) → A3(P) → A3(D) → A3(P) → A2(D) → NVIDIA(P) → Ascend(D)
```

逐层隔离：PD 状态机 → Ascend 后端兼容 → 跨代能力差异 → 跨厂商 Layout / 传输 / 数值差异。

### 4.5 异构发生在 Stage 边界

不在同一个 TP / PP / EP Collective Group 内混用不同厂商设备。Stage 之间通过 **StageContract** 协同，而不是让 NCCL 与 HCCL 组成统一通信域。

**StageContract 最小定义**（跨 Stage 边界的一等契约）：

```text
1. RequestState schema 版本（两端 Stage 协商）
2. 跨边界传输对象类型（Feature / KV / Activation）
3. KVDescriptor / KVPayloadFormat 协商字段
4. 失败 / 取消语义（transfer_complete / transfer_failed 回调）
```

**通信抽象的范围远不止 `init_process_group`**，须覆盖三类（Ascend 切 hccl 时逐点替换）：

```text
(1) init_process_group（backend: nccl / hccl）
(2) 构造期 get_rank / get_world_size → 改由 config / HardwareProfile 注入 tp_rank / tp_size
(3) forward 期 all_reduce / all_gather 原语
```

> 代码现状：`dist.*` 贯穿 `linear.py:23/24/62/106/139`（构造期）、`linear.py:155` 与 `embed_head.py:41/63-65`（forward 期）。Week 1 至少把 `LinearBase.__init__` 的 `dist.get_rank()` 纳入清单。

### 4.6 逻辑 KV 与物理 KV 解耦

P/D 节点不共享：`Block ID / Block Table / 设备地址 / Paged KV 物理布局 / Runtime 对象`。

三层契约（语义层 / 格式层 / 传输执行层）：

```text
KVDescriptor     ：描述数据语义（带版本号的结构体）
KVPayloadFormat  ：描述实际传输格式
KVConnector      ：执行传输（TCP / Host Staging / RDMA / Mooncake …）
```

`KVDescriptor` 至少含：

```text
version
num_layers / layer_range                 # 支持 LAYERWISE
num_tokens / token_range / cu_seqlens    # 支持 CHUNKED
num_kv_heads / kv_head_grouping          # GQA | MLA-latent | MLA-reconstructed
head_dim
rope_applied (bool) + rope_theta         # 禁止一端 absorb 一端不 absorb
dtype
quant_scheme (enum) + scale_layout       # 支持 QUANTIZED；scale/zp 存放位置
block_size (可选)                         # 支持 SOURCE_PAGED；block_table 属 Payload 元数据
endian                                   # BF16 跨厂商路径做 byteswap
```

MVP：`CANONICAL_CONTIGUOUS + BF16 + Host Staging`，逻辑格式 `[K/V, Layer, Token, KVHead, HeadDim]`。后续扩展 `LAYERWISE / CHUNKED / SOURCE_PAGED / QUANTIZED / NATIVE_FAST_PATH`。

> 物理侧来源：`KVDescriptor` 以 `utils/context.py` 的 `Context`（slot_mapping / context_lens / block_tables）为物理侧来源，`PDRequestState` 以 `engine/sequence.py` 的 `Sequence`（`num_cached_tokens` / `num_scheduled_tokens` / `block_table` / `temperature`）为承载，避免再造第四套对象。物理 paged layout `[2,L,B,bs,H,d]` → 逻辑 canonical `[2,L,T,H,d]` 的导出算子用 `store_kvcache` 逆操作（scatter→gather）作 Reference 兜底。

**跨厂商 Layout 与数值一致性**（Phase 4 重点）：

- **Paged 布局转换器**：CUDA paged `K=[num_blocks,H,head_size/x,block_size,x]`、`V=[num_blocks,H,head_size,block_size]`（`x=16` 向量化因子）由 GPU coalescing 决定，与 CANN 布局不同，不可逐字节搬运，须作为 Connector 内必备 stage。
- **MLA / decoupled-RoPE**：单列 KV payload 协议，声明传输 RoPE 前/后 K、latent/reconstructed K，禁止跨厂商一端 absorb 一端不 absorb（会产生结构性错误，非数值漂移）。
- **Model Fingerprint** 纳入 `rope_theta` + RoPE 实现变体 + Q/K/V 权重 hash + `num_kv_heads` + `head_dim` + `block_size` + `dtype` + `endian`；GQA/MLA head 映射升为独立校验项。

### 4.7 Stage / Backend / Connector 三层解耦

```text
┌──────────────────────────────────────────┐
│ StageEngine<abstract>                    │   ← Prefill/Decode/Encode
│   PrefillEngine / DecodeEngine / …       │     （未来 Attention/FFN）
├──────────────────────────────────────────┤
│ StageContract（跨边界一等契约）            │
├──────────────────────────────────────────┤
│ Connector                                │
│   FeatureConnector / KVConnector /       │
│   ActivationConnector (Phase A)          │
├──────────────────────────────────────────┤
│ Hardware Backend                         │
│   NVIDIA / Ascend / Future               │
│   └─ Platform（设备运行时门面，见第 5 节）  │
└──────────────────────────────────────────┘
```

### 4.8 A2 与 A3 共用上层协议

不分别维护 `engine_a2.py` / `engine_a3.py`。通过 Profile 选择算子实现 / 内存 API / Stream API / 通信 Backend / Graph 能力 / 量化能力。

**Profile 派生关系**（静态 → 动态）：

```text
HardwareProfile   （静态设备 SKU，不可变）
   ↓ 派生
CapabilityProfile （由 HardwareProfile 派生的算子/内存/通信能力位）
   ↓ 匹配
StageProfile      （某 Stage 在某设备上的实测/预期性能，PlacementPolicy 输入）
   ↓ 决策
PlacementPolicy   （匹配 StageProfile 与 CapabilityProfile 决定放置）
```

环境矩阵是项目一等资产，至少记录：设备 SKU、服务器型号、驱动/固件、CANN、PyTorch / `torch_npu`、Triton-Ascend、`torchair`、Host 架构、网络拓扑。

---

## 5. 核心软件架构

```text
                       ┌──────────────────┐
Request ──────────────▶│ Stage Router     │   （可按 EPD 顺序路由 E→P→D）
                       └────────┬─────────┘
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
     EncodeEngine        PrefillEngine        DecodeEngine
       (Phase E)                               │
            └────────── StageContract ─────────┘        （未来: AttentionEngine ↔ FFNEngine, Phase A）
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
   FeatureConnector  KVConnector  ActivationConnector (Phase A)
            └─────────────┼─────────────┘
                          ▼
                   Hardware Backend
              NVIDIA / Ascend / Future
                   └─ Platform
```

### 关键接口

| 接口 | 职责 | 备注 |
|---|---|---|
| `Platform` | 设备运行时门面：device set/select、`mem_get_info`、`memory_stats`、`empty_cache`、`synchronize`、graph(capture/replay)、`set_default_device` | 对应 vLLM Worker/Platform；读 `HardwareProfile` 决定行为；`capture_cudagraph`/`enforce_eager` 为 Ascend 降级入口（`enforce_eager=True`） |
| `OperatorBackend` | 通用算子（RMSNorm/RoPE/SiLU/…）统一入口 | 薄封装社区实现 |
| `AttentionBackend` | `prefill(...)` / `decode(...)` | 见 4.2 五种实现 |
| `KVCacheBackend` | `write` / `gather` / `import_logical` / `export_logical` | 收口 `store_kvcache` |
| `StageEngine`<abstract> | 单阶段执行 | 下挂 Prefill/Decode/Encode |
| `StageContract` | 跨 Stage 边界契约 | 见 4.5 |
| `RequestState` | 请求状态抽象 | 下挂 `PDRequestState` / `MultimodalRequestState`；是 StageContract 的 payload，`KVDescriptor`/`FeatureDescriptor` 是其子字段 |
| `KVConnector` / `FeatureConnector` / `ActivationConnector` | 三类数据通路传输 | 见第 2 节 |
| `HardwareProfile` / `CapabilityProfile` / `StageProfile` / `PlacementPolicy` | 设备能力与放置决策链 | 见 4.8 派生关系 |

---

## 6. 社区算子复用策略

| 能力 | 首选来源（Ascend） | 备选 | 复用原则 / 代码现状 |
|---|---|---|---|
| KV Cache Write | nano-vllm Triton Kernel 经 **Triton-Ascend 重编译** | SGLang / vLLM-Ascend | **L1–L2 非 L0**：改 Launcher 与 Layout（`attention.py:10`，原 CUDA Triton） |
| Prefill Attention | **`torch_npu` 推理 FA `npu_fused_infer_attention_score`（torchair）/ vLLM-Ascend DFlash** | Triton-Ascend（探索性） | 现 `flash_attn_varlen_func`（`attention.py:67`，CUDA-only）；**Ascend 无现成 Triton 候选**，不自研 |
| Decode Attention | **vLLM-Ascend native paged attention / `torch_npu` 推理 FA** | Triton-Ascend（探索性） | 现 `flash_attn_with_kvcache`（`attention.py:72`，CUDA-only）；优先 BF16/TP=1/Batch=1 |
| RMSNorm / RoPE / SiLU | 换 `@torch.compile` 后端（inductor→`torch_npu`/FlagGems 注册） | `torch_npu` 原生 | 现为 `@torch.compile` 的 PyTorch eager（`layernorm.py:16/28`、`rotary_embedding.py:37`、`activation.py:8`）；**绑定点是 compile 后端而非 kernel**；FlagGems Ascend 支持渐进式，逐算子 BF16 对齐 |
| Softmax | — | — | **移出矩阵**：仅内联在 sampler/attention，随 `AttentionBackend` 走 |
| Linear / GEMM | PyTorch / `torch_npu` | 厂商高性能实现 | 第一阶段不自研 GEMM；通信见 4.5 |
| Sampling | nano-vllm / PyTorch | SGLang / vLLM | **新增确定性 Greedy 路径**（见 4.3）；现 Gumbel-max 禁止 greedy |
| 通信 | TCP / Host Staging（MVP） | **Mooncake + Ascend Transport（Ascend RDMA 目标）/ NIXL（仅 NVIDIA 侧）** | 先验证协议，再优化数据面；Mooncake=Apache-2.0，已成 SGLang/vLLM/vLLM-Ascend 共同采用的事实标准，已支持 Ascend |

> **`torch_npu` Attention 算子区分**：推理用 `npu_fused_infer_attention_score`（torchair）；`npu_fusion_attention` 面向训练卡（含 dropout/keep_prob），**仅作正确性参考，不作推理后端**——A2/A3 推理直接复用会算子不可用，是 Ascend 最常见复用陷阱。

每个复用算子必须记录：`Source Repository / Commit / Source File / License / 输入输出 Layout / 支持 dtype / A2·A3 状态 / 修改内容 / 已知限制`。

---

## 7. 分阶段 Roadmap

> Phase 0–5 为 PD 主线（顺序推进）；方向 E / 方向 A 为并行方向，在 PD 稳定后启动。

### 主线 A — 文本 LLM 的 PD

| 阶段 | 主要目标 | 核心交付 | 阶段理想态 |
|---|---|---|---|
| **Phase 0：项目基础** | 可持续 Fork + Golden Baseline + 接口骨架 | upstream 管理、CI、环境矩阵、Qwen3 CUDA Golden、确定性 Greedy、基础接口（Platform/Backend/StageContract/RequestState 雏形） | CUDA 原路径零回退，开发环境可复现 |
| **Phase 1：社区算子复用与 Ascend 单体** | 最大程度复用 Triton-Ascend / Ascend 社区算子 | 算子复用矩阵、Backend 适配、解除四类硬绑定、A2/A3 单体推理 | Qwen3 在 A2/A3 完成 P+D，输出与 Golden 对齐 |
| **Phase 2：Ascend 同代 PD** | A2(P)→A2(D)、A3(P)→A3(D) | P/D Engine、RequestState、KVDescriptor v0、KVConnector、`transfer_complete/failed` 回调、block 引用计数 | 单请求 PD 正确，多请求基础稳定 |
| **Phase 3：Ascend 跨代 PD** | A3(P)→A2(D) | HardwareProfile/CapabilityProfile/StageProfile/PlacementPolicy | 能按阶段 Profile 选 A2/A3 放置 |
| **Phase 4：跨厂商 PD** | RTX 5090(P)→Ascend A2(D)（正确性 Case） | Canonical KV、Paged 布局转换器、endian、Model Fingerprint、MLA/RoPE 协议 | 跨厂商输出与单体 baseline 在容差内对齐 |
| **Phase 5：PD 生产能力** | 吞吐、容错、可观测 | Continuous Batching、Router、Backpressure、取消/重试、Mooncake RDMA、可观测、厂商融合算子可选 backend | 多 P/多 D 稳定，满足 TTFT/TPOT SLO |

### 方向 E — 多模态 EPD（PD 稳定后）

首个多模态模型建议 Qwen2.5-VL / Qwen3-VL / InternVL；从 `E/PD` 起步演进到 `E/P/D`，支持独立扩缩容，含 `FeatureConnector` 与多模态 `RequestState`，可跨代、跨厂商组合部署。

### 方向 A — AFD（EPD 后）

`ActivationConnector` / A-F Pipeline / 阶段配比模型；只在 Profiling 与互联带宽证明收益时启用。

---

## 8. Phase Gate 与验收标准

所有 Gate 的"对齐"统一改为可执行断言，量化判据见 [Test & Golden Strategy](#82-test--golden-strategy)。

### 8.1 Test & Golden Strategy

```text
(1) Greedy 前 64 token 100% exact-match          （hard gate，确定性）
(2) 张量级 max_abs_err / mean_abs_err / cos_sim  （软指标；BF16 跨厂商 cos_sim ≥ 0.999）
(3) KV 逐元素 atol = 5e-3                         （隔离"传输损坏" vs "计算漂移"）
(4) 容差表：同厂商 BF16 atol/rtol、跨厂商漂移区间、FP8 KV 放宽
(5) Golden 覆盖矩阵：Prompt 长度档 {1, Block 边界 255/256/257, 多 Block, 4K} × Batch × 模型档
    （Block Size 默认 256，与 config.py:22 `assert kvcache_block_size % 256 == 0` 对齐）
(6) Golden 制品版本化，绑定环境矩阵 hash
```

### 8.2 各阶段退出条件

- **Phase 0**：CUDA Golden 可复现；环境矩阵锁定；接口骨架就位；确定性 Greedy 可用。
- **Phase 1（Ascend 单体）**：Qwen3-0.6B BF16/TP=1/Greedy 可运行；前 64 token 与 CUDA Golden exact-match；Prompt 覆盖 1/Block 边界/多 Block；无 NaN/Inf；A2 与 A3 同套上层接口。
- **Phase 2（Ascend PD）**：首 Token 语义正确；KV 导出/传输/导入/Block 重建正确；单体与 PD 输出一致；Block 边界 255/256/257 正确；**StageContract v0 可序列化、两端可独立解析**；单请求重复 N 次内存断言归零（显式状态泄漏检测）。
- **Phase 3（跨代 PD）**：A3(P)→A2(D) 与 baseline 在容差内对齐。
- **Phase 4（跨厂商 PD）**：Model Fingerprint + Protocol Version 校验通过；BF16 Canonical KV 正确；CUDA/CANN Layout 转换正确；跨厂商 cos_sim ≥ 0.999；失败请求可清理。
- **Phase 5（生产 PD）**：Continuous Batching 下多 P/多 D 稳定；TTFT/TPOT 达标；容错重试可用。
- **方向 E / A**：Gate 待 Profiling 数据后补充。

---

## 9. 工程基础

### 9.1 Upstream Tracking

长期 fork 模型。每 upstream release tag 开一次 rebase / cherry-pick window；core 保持与 upstream 可 merge 结构，nano-xpu 专属逻辑集中在 `HardwareProfile` / Backend / Connector 扩展点以缩小冲突面；建立 visible 的 upstream-sync issue/label 与冲突处理 SOP（谁 rebase、conflict 走 PR、breaking change 降级）。

### 9.2 License Compliance

fork 对外仍 MIT，但含按各自 license 再分发的第三方算子。建立 `THIRD_PARTY_LICENSES` / `NOTICE` 机制；复用矩阵每算子加 `NOTICE-required` / `Patent-clause` 两列；维护 license 兼容性矩阵（哪些可静态链入 MIT、哪些须独立模块动态调用）；算子引入 review gate（谁签 off、GPL/AGPL 拦截）。

已知：Mooncake=Apache-2.0、Triton-Ascend=MIT、FlagGems=BSD、vLLM/vLLM-Ascend/SGLang=Apache-2.0、flash-attn=自定义（非 MIT 兼容，须独立模块）。

### 9.3 Fault Tolerance & Recovery

故障分类：D 节点 crash、KV 传输 fail、校验 fail、请求 cancel、慢节点。检测信号 + 恢复动作（retry / fallback 单体 / KV 重新 prefill）。Phase 2 即引入 `KV block 引用计数 + ack-then-free`（不依赖 RDMA）作为 Phase 5 容错前置；`KVConnector` 定义 `transfer_complete` / `transfer_failed` 回调；per-request resource ledger（idle 后强制 GC，CI 长时压力断言 ledger 归零）。

### 9.4 Observability

阶段级 latency breakdown（request→stage wall-clock + KV transfer bytes/duration）作为 StageContract 一部分；metrics 最小集：TTFT / TPOT / per-stage time / KV transfer throughput / block utilization / OOM drop rate；每 request 一个 `trace_id` 跨 P/D 传播（走 StageContract header）；导出 OpenMetrics/Prometheus + OpenTelemetry。Phase 5 写具体 SLO 数值（TTFT≤x ms、TPOT≤y ms）。

### 9.5 Performance Baseline & Regression

bench 矩阵：离线 correctness bench（固定 Golden）+ 在线 throughput/latency bench（多 batch/seq-len）；SLO 表（每代硬件一个目标，如 A2 Qwen3-0.6B tok/s 与 TTFT）；回归判定（相对 baseline 允许回退 ≤3% warn，超出 block）；结果绑定环境矩阵 hash；Phase 0 即建 CUDA baseline。

### 9.6 Configuration & Reproducibility / Ascend CI 策略

环境矩阵以可执行制品固化（Dockerfile + requirements lock + 矩阵 yaml，每条目可 hash）。Ascend CI 三档：

```text
(1) 无 NPU：跑 ReferenceBackend/CPU 路径的接口与协议测试（PDRequestState/KVDescriptor 序列化、StageContract 校验全覆盖）
(2) Triton-Ascend 编译期检查（NPU 前置 gate）
(3) 真机 smoke：自建 / Ascend 云 runner 按 label 触发（按资源投入决定，当前 optional）
```

### 9.7 Risk Register

每条含 触发条件 / 影响面 / 缓解 / 降级 / 负责节点，随阶段更新：

| 风险 | 降级方案 |
|---|---|
| Triton-Ascend 算子覆盖缺口 | 降级 `torch_npu` native / ReferenceBackend；触发 L4 自研门 |
| 跨厂商 BF16 KV 数值漂移超容差 | 扩容差 / fp32 Canonical KV / 退回同厂商 PD |
| RTX 5090 + Ascend 数值不可接受 | 仅作协议验证，不作性能结论；换数据中心 NVIDIA Case |
| HCCL/NCCL 无法组统一通信域 | 维持 Stage 边界异构（已是默认） |
| Host Staging TCP 带宽天花板致 PD 收益为负 | 切 Mooncake + Ascend Transport RDMA |
| Upstream 漂移 | 分层 rebase 策略（见 9.1） |
| 单一硬件来源依赖 | 环境矩阵多 SKU 覆盖 |

### 9.8 Governance & Contributing

接口分级：**core 接口（StageContract / RequestState）Phase 2 末（PD 跑通后）冻结为 stable，Phase 0–2 标 experimental**；实现层（Backend / Connector 签名）持续 experimental。外部 backend/connector 贡献须附环境矩阵覆盖、Golden 对齐证据、license/NOTICE 清单、复用矩阵登记。明确当前阶段不接受哪类贡献（如 Phase 0 不接受新 attention kernel）。

---

## 10. Week 1 计划

> Week 1 不自研新 Kernel，而是建立社区算子复用体系，并最大可能拼接出 Ascend 最小执行路径。

| 工作点 | 主要内容 | 注意事项 | 理想态 |
|---|---|---|---|
| **Fork 与 Baseline** | 固定 nano-vllm commit、建 upstream、跑通 Qwen3 CUDA Golden、**新增确定性 Greedy** | 不改原 CUDA 行为；固定模型/依赖/随机性 | 原路径零回退，Golden 可重复 |
| **Ascend 环境矩阵** | 锁 A2/A3 的 SKU、CANN、PyTorch、`torch_npu`、Triton-Ascend、`torchair` | 不用模糊"A2/A3"标签；版本可复现 | A2/A3 Smoke Test 可运行 |
| **社区算子盘点** | 建 nano-vllm / Triton-Ascend / FlagGems / SGLang / vLLM-Ascend 复用矩阵 | 记录 Commit/License/Layout/dtype/已知限制 | Qwen3 关键算子都有候选 |
| **Backend 薄封装** | 建 `OperatorBackend`/`AttentionBackend`/`KVCacheBackend` + `Platform` 能力清单；`pyproject.toml` 改 `flash-attn`/`triton` 为 optional extras（cuda/ascend），core 仅留 torch/transformers/xxhash | 接口包裹社区实现，不重写；**附解除 flash-attn/torch.compile/NCCL/CUDA Graph 四类硬绑定的最小 diff** | CUDA 与 Ascend 候选可经统一接口调用 |
| **Triton / Ascend 算子验证** | KV Write 经 Triton-Ascend 编译；Prefill/Decode Attention 候选锁定 `torch_npu` FA / vLLM-Ascend DFlash（**非假定 Triton**） | 先 BF16/TP=1/Batch=1；编译成功≠数值正确 | KV Write + 一个 Prefill + 一个 Decode 候选在 A2 数值正确 |
| **基础算子拼接** | 验证 RMSNorm/RoPE/SiLU/Embedding/Sampling（换 compile 后端） | 优先 FlagGems/`torch_npu`/社区 Triton | Qwen3 单层 Prefill/Decode 可执行 |
| **PD 协议草案** | 定义 `PDRequestState v0`、`KVDescriptor v0`、首 Token 语义、StageContract v0 | 区分逻辑协议与物理传输格式 | Week 2 可基于协议接入执行链 |
| **冲刺（Stretch）** | 尝试 Qwen3-0.6B A2 单体产生 1 个正确 Greedy token | **不计入必须完成，失败无碍进入 W2** | A2 生成若干正确 token |

### Week 1 必须完成

```text
CUDA Golden Baseline（含确定性 Greedy）
A2/A3 环境与版本矩阵
社区算子复用矩阵
Backend + Platform 接口骨架（解除四类硬绑定的最小 diff）
KV Write 在 A2 经 Triton-Ascend 编译且数值正确
一个 Prefill 与一个 Decode Attention 候选（torch_npu FA / vLLM-Ascend）在 A2 跑通单次 forward
Qwen3 单层执行链
PDRequestState v0 / KVDescriptor v0 / StageContract v0
```

### Week 1 不做

```text
自研完整 Attention Kernel          Continuous Batching
算子极致性能优化                    RDMA / Device Direct
PD 网络端到端                       FP8 KV
整模型单体 P+D（顺延 W2）           跨 TP KV 重分片
Graph Mode                          AFD
```

---

## 11. 近期节奏

Phase ↔ Week 映射（每周末对照 Phase Gate 复核）：

```text
W1   社区算子发现/复用/兼容性验证；Platform 骨架；A2 单层路径          ≈ Phase 0–1 起步
W2   补齐算子适配；A2/A3 单体 P+D                                      ≈ Phase 1
W3   A2(P) → A2(D)                                                     ≈ Phase 2A
W4   A3(P) → A3(D)                                                     ≈ Phase 2B
W5   A3(P) → A2(D)                                                     ≈ Phase 3
W6+  RTX 5090(P) → Ascend A2(D)；跨厂商协议正确性验证                   ≈ Phase 4
```

PD 主线稳定后：多模态 EPD（E/PD → E/P/D → 跨代 → 跨厂商）；AFD 在 Profiling 与互联收益模型成立后推进。

---

## 12. 项目理想态

不是维护多套硬件 Fork，而是形成：

```text
统一 StageContract          可插拔 Hardware Backend
统一 RequestState           可插拔社区算子 Backend
统一 KV/Feature/Activation  可插拔 Connector
基于 HardwareProfile 的阶段放置
```

最终可表达：

```text
Ascend A3 Prefill → Ascend A2 Decode
NVIDIA   Prefill  → Ascend Decode
NVIDIA   Encoder  → Ascend Prefill → Ascend Decode
Ascend   Attention ↔ 高算力 FFN Pool
```

---

## 13. 术语与缩写

| 缩写 | 全称 / 含义 |
|---|---|
| PD / EPD / AFD | Prefill–Decode / Encode–Prefill–Decode / Attention–FFN Disaggregation |
| P / D / E / A / F | Prefill / Decode / Encode / Attention / FFN |
| TP / PP / EP | Tensor / Pipeline / Expert Parallelism |
| MoE | Mixture-of-Experts |
| HBM | High Bandwidth Memory |
| TTFT / TPOT | Time To First Token / Time Per Output Token |
| NCCL / HCCL | NVIDIA / Huawei Collective Communication Library |
| CANN | Compute Architecture for Neural Networks（Ascend 软件栈） |
| FA | FlashAttention |
| GQA / MLA | Grouped-Query / Multi-head Latent Attention |
| E/P/D、E/PD | EPD 的三阶段分离 / Encoder 独立 + P+D 共置 |

> 注：`HardwareProfile` 等类型名中的 Profile 指"能力描述对象"；表性能评估/收益测量处统一用 **Profiling**，避免一词两义。

---

## References

### PD / 分离推理

- [Disaggregated Prefill — vLLM Ascend](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/disaggregated_prefill/)
- [SGLang — PD Disaggregation](https://docs.sglang.ai/advanced_features/pd_disaggregation.html)

### EPD / 多模态

- [Efficiently Serving Large Multimodal Models Using EPD Disaggregation (arXiv 2501.05460)](https://arxiv.org/abs/2501.05460)
- [EPD-Serve: Multimodal EPD Disaggregation on Ascend (arXiv 2601.11590, 2026)](https://arxiv.org/abs/2601.11590)
- [HydraInfer: Hybrid Disaggregated Scheduling for Multimodal LLMs (arXiv 2505.12658)](https://arxiv.org/abs/2505.12658)
- [SGLang — EPD Disaggregation](https://docs.sglang.ai/advanced_features/epd_disaggregation.html)（旧域 `sgl-project.github.io` 已永久迁移至 `docs.sglang.ai`）
- [EPD Disaggregation — vLLM Ascend](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/epd_disaggregation/)
- [Encoder Disaggregation — NVIDIA Dynamo](https://docs.nvidia.com/dynamo/user-guides/multimodal/encoder-disaggregation)

### 社区算子与后端

- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [Triton-Ascend (triton-lang/triton-ascend, MIT, 3.2.1/CANN 9.0.0, 覆盖 A2&A3)](https://github.com/triton-lang/triton-ascend) — 注：`Ascend/triton-ascend` 仅为 GitCode 镜像且版本滞后；`Ascend/triton-ascend-ops` 为预调优算子库（备选来源）
- [FlagGems (BSD)](https://github.com/FlagGems/FlagGems)
- [SGLang (Apache-2.0)](https://github.com/sgl-project/sglang)
- [vLLM Ascend (Apache-2.0)](https://github.com/vllm-project/vllm-ascend) — 已支持 PD（Mooncake, A2/A3）与 EPD（MooncakeLayerwise），随 vLLM 0.11.1 原生 EPD
- [Mooncake Transfer Engine (Apache-2.0)](https://github.com/kvcache-ai/Mooncake) — Ascend 侧 RDMA/KV 传输目标路径
- [Ascend Extension for PyTorch (torch_npu)](https://github.com/Ascend/pytorch) — 注：GitHub 为镜像，主仓在 gitcode.com/Ascend/pytorch
- [torch_npu.npu_fusion_attention — Ascend Docs](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000139.html)（训练语义，推理用 `npu_fused_infer_attention_score`）
