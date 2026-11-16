# RFC-0002：引入 vllm-router，一条 CLI 启动 P/D 合部与分离集群

- 方案状态：草案；已有 CPU 范围实施授权，正式接受记录未在本 RFC 登记。
- 实施状态：部分实现。配置规划、进程控制、角色 HTTP、descriptor/lease 和 CPU 阶段引擎已有实现；生产 NIXL/P/D 后端未实现。
- 验证状态：本地 CPU 回归已有通过记录，最新版本与范围见 [验证报告](../testing.md)；18 项 Linux 集成仅完成收集，真实 router/SSH 与设备验证未完成。
- 作者 / 维护人：初稿署名未记录；本次由 Codex 按用户要求修订；维护人待确认。
- 创建 / 更新日期：原稿创建日未单独记录 / 2026-11-15。
- 历史问题基线：`yangjingo/nano-xpu@53e77bb993291ccc28d8e3c943ab618687aa32e2`。
- 当前核查版本：`2d61a9c8aa6eaf7eb6a928b1331f67418b5bd79b` + `feat/rfc-cpu-deployment` 工作区未提交修改；不代表默认分支已发布能力。
- 关联与依赖：[RFC-0001](0001-serve-qwen-colocated.md)、[Roadmap](../roadmap.md)；真实 CUDA/TP 前置尚待设备验收，历史 43 项服务层测试不能替代。独立远程 Issue/PR 未登记。
- 反馈对象与待决问题：仓库维护者（具体人选待确认）；NIXL 版本/API、真实数据通路与设备 Gate。任务标题：`feat: 用统一部署 CLI 编排 vllm-router 与 nano-xpu P/D workers`。

## 1. 要交付什么

**用户配置各角色的 IP 和卡号，再执行一条命令获得统一推理地址。** 首次准备模型、Python 环境、router 和 SSH 后，日常部署只改拓扑；模型路径、依赖版本和端口规则复用 profile。CLI 负责启动整组进程，每个计算角色由内部 `nano-xpu _worker` 拉起。

| 模式 | 角色 | 请求经过哪里 | 首次设备 Gate |
|---|---|---|---|
| `colocated`，P/D 合部 | router + 一个或多个 `colocated` worker | router → 任一完整 P+D worker | CUDA TP=1，两副本转发与隔离 |
| `disaggregated`，P/D 分离 | router + `prefill` + `decode` | router → P → KV 交接 → D | 同型号 CUDA、1P1D、两端 TP=1 |

**两种模式都必须接入 vllm-router，包括单张 5090、仅一个合部 worker 的情况。** `serve -f` 必须同时管理 router 和全部 workers；router 未启动或未就绪时，整组部署不能报告成功，不提供跳过 router 的部署选项。用户调用 router 的 8000 端口；worker 的 8100 起始端口只用于内部转发、探测和调试。公开 CLI 拒绝直接 `serve MODEL`；单 worker 合部也使用同一份部署配置。

“合部”不表示不同模型共享一张卡。“分离”不表示只给两个完整推理服务加角色标签。首版不做 GPU/NPU 跨厂商、跨 TP KV 重分片、PP、EPD、SSE、自动扩缩容、Kubernetes 或在线切换拓扑。两种模式分别部署，不能在同一集群混填完整 worker 与 P/D 池；未来混合路由另开设计。

## 2. 证据：已有的入口还缺什么

**历史基线已有 HTTP 服务，但没有 P/D 引擎或 KV 交接。** 下表描述 `53e77bb993291ccc28d8e3c943ab618687aa32e2` 的行为，当次核查工作区干净；当前分支已有 CPU 控制实现，见第 6 节，不能把历史描述当作现状。相对代码链接用于定位当前模块，历史内容以该完整提交为准。

| 证据 | 历史基线行为 | 原始改动目标 |
|---|---|---|
| [`server/cli.py::parse_config`](../../nanoxpu/server/cli.py) | 一个模型、一个本地服务 | 统一 serve 配置入口、角色与集群配置 |
| [`schema.py::ChatRequest`](../../nanoxpu/server/schema.py) | `extra=forbid`，拒绝 `kv_transfer_params` | 角色专用且受约束的内部请求模型，不能全局放开未知字段 |
| [`app.py::EngineService`](../../nanoxpu/server/app.py) | 一条活动生成、专属执行线程 | P 留存 KV lease、D 等待导入，保持引擎线程所有权 |
| [`LLMEngine.step`](../../nanoxpu/engine/llm_engine.py)、[`BlockManager.deallocate`](../../nanoxpu/engine/block_manager.py) | 同一实例 P+D；请求结束立即释放 block 引用 | 持有、导出、导入、确认、释放的独立生命周期 |
| [`Sequence.__getstate__`](../../nanoxpu/engine/sequence.py) | TP 的本地控制数据序列化 | 不是跨实例恢复协议；另定义版本化描述符 |

**router 负责两阶段 HTTP 协调，KV 搬运属于推理后端。** 本 RFC 正式依赖候选为上游 [vllm-project/router](https://github.com/vllm-project/router)，固定核查提交 `07c20da66c255ba65a97ab531f9fb5f60413a7d4`。不把任意同名 PyPI 包或 `latest` 当作同一版本。

上游该提交的 [`vllm_pd_router.rs`](https://github.com/vllm-project/router/blob/07c20da66c255ba65a97ab531f9fb5f60413a7d4/src/routers/http/vllm_pd_router.rs) 中，`prepare_prefill_request` 把 P 请求改成 `max_tokens=1, min_tokens=1, stream=false`；`build_prefill_kv_transfer_params` 注入连接器参数；NIXL pull 分支的 `build_decode_kv_transfer_params` 克隆 P 响应中的 `kv_transfer_params`，再与原始请求交给 D。**只启动 router 不会让 nano-xpu 自动获得这些能力。** 对比范围仅为 HTTP P/D 协议，不复用其他调度扩展。

本地还参考了 `code/cann-deepinfer-recipe/router/`，其所属仓库提交 `1d11df307b25d8ff665dcc8acdd05b4f34932ae5`、router 路径无未提交修改；该副本仅用于发现入口，不作为本项目的打包来源。旧 recipe RFC 与本方案无接口或授权关系。

## 3. 用户如何启动

### 3.0 统一配置入口（已实现，本地 CPU 已验证）

用户只使用 `nano-xpu serve -f cluster.yaml [--detach]` 启动整组服务；`serve plan -f ...` 只规划，`serve status/logs/down --name ...` 查询或停止同一个 ledger。合部与分离只由 YAML 的 mode、workers 和 profile 区分，单 worker 合部也必须经 router。旧 `deploy` 和 `serve MODEL` 不保留公开别名，返回参数错误和迁移提示，不能默默改变旧脚本的行为。

内部保持两层：`deployment/cli.py` 校验参数并调用 Controller；`server/cli.py` 解析单 worker 参数并启动 HTTP/引擎。planner 生成 `nano-xpu _worker MODEL ...`，helper 的来源检查与 controller 的模型路径提取同步改为 `_worker`。该入口不列入公开帮助，只供受控编排和明确标记的测试 harness 使用，不是鉴权边界。`runtime-info` 保留给实际导入包来源检查。配置格式和 ledger 结构不变；离线 `LLM` API 不变。

反串讲：同一份合部 YAML 经 serve → planner → 全节点 preflight → supervisor → `_worker`，确认 worker 就绪后才启动 router；成功返回 router URL。更换为 P/D YAML 仍走同一流程。若只是把 deploy 改名，却继续让 `serve MODEL` 绕过 router，就没有统一启动语义。内部 worker 必须避免再次进入 serve 编排，否则会递归创建集群。

| 用例 | 执行与预期 | 实际结果 |
|---|---|---|
| U01 | 两种配置经 `serve plan` 生成同原语义的完整 router/worker 计划，不导入加速器、不连 SSH | 本地通过；见 [验证记录](../testing.md) |
| U02 | `serve -f` 及 detach/前台停止进入同一 Controller；status/logs/down 使用既有 ledger；规划无副作用 | 本地通过；见 [验证记录](../testing.md) |
| U03 | 缺配置、旧 deploy、直接模型路径、混用启动和管理参数返回 2；help 不加载引擎 | 本地通过；见 [验证记录](../testing.md) |
| U04 | planner、preflight、实际 CPU worker 与示例脚本使用内部入口；独立 CLI 恢复仍可用 | 本地 CPU worker/来源检查通过；真实 SSH/router 与独立 CLI 恢复留待 Linux CI |
| U05 | Windows 读句柄暂时占用 ledger；持续权限拒绝；更新原子性和临时文件清理 | 6 项回归通过；限时重试，持续失败保留旧状态并上抛 |

测试整理不改变验收范围：基础用例按模块归入 `tests/`，真实 Linux 验证放入 `tests/integration/`；267 + 18 个用例迁移前后集合一致，CI 两个 job 共同覆盖全部。版本对照与共享夹具约定见 [测试目录指南](../../tests/README.md)。

实施顺序：更新入口与 worker 参数解析 → planner/preflight/controller → 脚本与测试 → 文档和安装 wheel 验证。此次明确改变命令兼容性，旧启动脚本必须按迁移表更新；生产 NIXL 和设备 Gate 不因此完成。

**统一配置入口，worker 收为内部命令。** 以下是本次确认的接口；`plan` 可本地执行，`up` 需要锁定版本并通过远端 preflight。当前 CPU 测试不能证明真实 router 或 GPU 部署已可用：

```bash
# 核对 IP、GPU 映射、端口和即将执行的命令；不创建远程进程
nano-xpu serve plan -f cluster.yaml
# 一键拉起，所有必要角色就绪后返回统一地址；失败返回非零
nano-xpu serve -f cluster.yaml --detach
nano-xpu serve status --name qwen-lab
nano-xpu serve logs --name qwen-lab --role prefill
nano-xpu serve down --name qwen-lab
```

不加 `--detach` 时控制台持续跟踪运行状态，Ctrl+C 执行同一停止流程；每个角色有独立 detached supervisor，helper 返回后仍持有子进程，不依赖 SSH 连接持续在线。首版不自动重启失败的 worker，避免旧 KV 状态被误用。

```yaml
# disaggregated；节点 ID 在集群中唯一。合部只需改 mode 和 workers。
schema_version: 1
name: qwen-lab
profile: ./qwen3-cuda-profile.yaml
mode: disaggregated
router:
  ip: 10.0.0.10
workers:
  - {id: p0, role: prefill, ip: 10.0.0.11, gpus: [0]}
  - {id: d0, role: decode, ip: 10.0.0.12, gpus: [0]}
```

样例文档（先保存其中的 YAML 代码块）：[公共 profile](../../examples/deploy.md#公共-profile)、[合部拓扑](../../examples/deploy.md#合部部署)、[分离拓扑](../../examples/deploy.md#pd-部署)。`plan` 不连接示例 IP；运行 `up` 会尝试连接所配置节点，必须先替换地址和版本占位。profile 配置模型路径、服务名、远程入口和 SSH 用户；worker 可覆盖 `model_path`，但模型内容指纹必须一致。helper/worker 必须从指定干净 checkout 导入，单独 wheel 暂无可接受的部署来源证明；安装和恢复命令见 [部署说明](../deployment.md)。

### 3.0.1 包名与源码对应的测试目录（已实现，本地验证已回填）

整改前 Python 包、安装元数据和进程入口使用 nanovllm/nano-vllm，与项目 Nano-XPU 名称不一致。本次已将源码包统一为 `nanoxpu`，发行包改为 `nano-xpu`，console script 仍叫 `nano-xpu`；同步 imports、动态模块名、`python -m`、SSH helper/supervisor、运行来源检查、CI、样例和文档。上游来源、作者与许可证保留原样。不提供旧导入别名，不改变推理算法、配置字段、HTTP 或 KV 协议。

反串讲：一次配置启动由 console script 调用 `nanoxpu.cli:main`，SSH 在目标 Python 环境执行 `nanoxpu.deployment.helper`，helper 派生同包下的 supervisor；preflight 核查实际 `nanoxpu/` 与提交来源。只移动文件夹会让这些字符串入口或包发现规则仍指向旧代码，故必须检查独立安装的 wheel 与真实子进程。现有用户须更新 imports，先用原版本停止已有服务、卸载旧发行包，再从相同新提交重新安装所有节点；不迁移活动进程或承诺旧 Python pickle 跨包兼容。回退时完整恢复旧提交及环境，不能混用新旧节点。

测试以源码职责为主：server/engine/deployment/kv 一一对应；根级 cli.py 与 runtime_identity.py 的测试位于 tests 根目录。共享测试工具归入 helpers，避免与生产 utils 混淆；真实 Linux 跨组件测试保持 integration。无测试的源码目录不创建空测试目录，也不为目录对称新增无意义测试。

| 编号 | 输入与方法 | 通过条件 | 实际结果 |
|---|---|---|---|
| N01 | 核对源码、脚本、CI 的模块引用，编译全部源码；比较迁移前后 pytest collection | 活跃代码无旧模块入口；285 个既有用例全部保留 | 285 个用例集合相同；47 个生产文件仅命名替换；编译通过 |
| N02 | 构建新 wheel，在源码之外全新安装，检查 METADATA、入口、包清单、模块 help 与两类 plan | 包名/入口一致；wheel 不含旧包；无 CUDA 依赖也能执行控制入口 | 新 wheel 源码外安装、包清单/元数据/旧包缺失与 8 项 CLI 检查通过 |
| N03 | 完整 CPU 回归，含 helper/supervisor 子进程和独立 checkout 来源检查 | 267 项既有基础用例零失败零跳过；18 项 Linux 集成另行执行 | 完整复测 267 通过；首轮 HTTP 退出间歇超时尚未定位；18 项 Linux 待运行 |
| N04 | 旧环境迁移说明、源码链接、CI 路径与设计稿检查 | 新用户命令一致，保留上游归属；不将 Linux/GPU 未执行写成通过 | 迁移文档、117 个本地链接与 CI 静态检查通过；设备和 Linux 未验证项保留 |

示例现在仅以 Markdown 保存，YAML 与脚本嵌入代码块；测试和 CI 提取到临时目录执行，同一份代码接受规划和 HTTP 验证。新增目录约束后基础套件 268 项通过；原有 285 项用例全部保留，总收集 286 项，其中 Linux 集成 18 项仍未运行。详见 [部署示例](../../examples/deploy.md) 与 [验证记录](../testing.md)。

### 3.1 参数、地址和卡号的明确含义

| 配置 | 默认或规则 | 失败行为 |
|---|---|---|
| `router.ip` / `worker.ip` | SSH、HTTP 和 KV 对端可达的 IPv4；首版不接受 hostname/IPv6 | 不能用 `0.0.0.0` 作为对端地址；网卡不一致须覆盖 `advertise_ip` |
| `gpus` | 当前主机**物理卡号**；写入 `CUDA_VISIBLE_DEVICES` | 列表为空/重复/卡不存在/同机分配重叠则拒绝 |
| TP | 默认 `len(gpus)`；进程中映射为逻辑 `0..TP-1` | 显式 TP 与卡数不一致拒绝；P/D 首版 TP 必须都为 1 |
| HTTP 端口 | router 8000；worker 按 workers 列表顺序，从 8100 起分配；plan 固定该顺序 | 规划结果固定，检测到冲突则报错，不偷偷换端口 |
| rendezvous / KV 端口 | worker 同一序号分别从 23000 / 24000 起 | 每台主机所有用途去重；合部不开放 KV 端口 |
| namespace | `cluster_id=name-config_hash前12位`，`instance_id=cluster_id-worker_id`；每次实际启动生成新的 generation | 共享内存名由 instance/generation 哈希生成；角色 ledger 按 name/role 存储，并校验 owner/generation；lease 含 cluster/request/generation |
| transport / API | 分离先固定 `nixl-pull`、非流式文本聊天 | 不支持的模式直接拒绝，不静默退回完整 P+D |
| 时间限制 | profile 默认 startup 600s、request 120s、lease 180s、drain 30s | schema 要求 lease 覆盖 request+drain；startup/request 已传 router，drain 用于停止；生产 P/D 的 lease/transfer 参数接线仍待后端实现 |
| 运行来源 / SSH | `helper_python` 指定绝对路径；可选 `known_hosts_file` 是控制机文件 | 实际导入路径、代码 clean、完整 revision 不符即拒绝；SSH 始终严格校验 host key |

同一主机 P 用 `[0]`、D 用 `[1]` 表达同机 PD；不同 IP 各用 `[0]` 表达跨机 PD。仅 router 角色不占卡。helper 已拒绝同一状态目录中其他部署的 GPU/端口预留冲突；它不是主机全局资源调度器，不能自动识别其他用户或第三方任务的 GPU 占用，目标设备的资源可用性仍须验收。

一键启动的前提：Linux 计算节点、SSH BatchMode 密钥认证与可信 known_hosts、同版本 nano-xpu、匹配 CUDA/依赖、模型权重和 router 已安装。MVP 不安装驱动、分发模型、提示输入密码或关闭 SSH host key 校验。控制机可以是 Windows。

## 4. 三层职责与启动流程

**CLI 管进程，router 管请求，nano-xpu 管计算与 KV。** 不把 KV 数据经过 CLI 或 router 中转。下图是目标 P/D 启动流程的解释性示意；controller 已实现相应编排，生产 P/D 后端和真实链路仍未验收。

```mermaid
sequenceDiagram
    participant CLI as Deployment CLI
    participant Host as SSH helper / role supervisor
    participant P as internal worker / P
    participant D as internal worker / D
    participant R as vllm-router
    CLI->>Host: preflight + resolved LaunchSpec
    Host->>P: start _worker --role prefill
    Host->>D: start _worker --role decode
    CLI->>P: /ready + /server_info
    CLI->>D: /ready + /server_info
    CLI->>CLI: match model / KV / connector fingerprint
    CLI->>Host: start router with P and D pools
    Host->>R: start pinned router process
    CLI->>R: readiness + worker registration check
    CLI-->>CLI: record identities; report endpoint
```

**启动前检查与启动后验收分开，不能把目标检查写成已经实现。** 当前 `helper.preflight()` 检查可执行文件、workdir、helper/worker 实际导入来源、模型文件摘要、端口和已有资源预留；router 核对源码提交与二进制哈希；GPU 路径只通过 `nvidia-smi` 检查卡号存在。所有节点通过这些检查后才启动任何 worker。

显存容量、连接器初始化、P↔D 数据通路和完整网络可达性尚未形成设备 preflight，保留为 T2-10/T2-16～18 的交付要求。启动后 controller 检查 worker 就绪与能力，再检查 router 的健康 worker 池；这些探测不能替代真实 KV 传输验收。生产 `NanoEngine.start()` 当前拒绝 P/D，不能用缺少检查的路径宣布可部署。

helper 通过有界 JSON stdin 接收请求，再创建每角色的独立 supervisor；进程以参数数组和环境字典启动。用户字段不拼进远端 shell。状态存于 `name/role_id` 目录，记录 PID、创建时间、用户、命令哈希及 owner；新 owner 在发出 start RPC 前写入控制机 ledger，回复丢失仍可回滚。helper 返回 RUNNING 只代表进程存活，`up` 还须验证 HTTP 就绪。

worker 启动后通过带内部凭据的 `/server_info` 比对实例/generation/cluster/role/model 和 model/tokenizer/template/RoPE 摘要；P/D 还检查 TP=1、transport/version、connector_ready 与 layout_digest 一致。dtype、层/头维度、stride 等在 KVDescriptor 的 ModelLayout 中校验，不作为已存在的 `/server_info` 独立字段。router 最后启动，controller 同时检查 `/health` 与 `/workers` 的完整 URL/角色/健康集合；router 自身没有 `/ready`。

**实际集群 ledger 的主路径是 `STARTING → RUNNING → STOPPING → STOPPED`。** 清理未完成保留 `CLEANUP_FAILED`；启动异常记录 failure 类型并进入同一清理流程。preflight 失败不会创建新的集群 ledger。`status` 根据 ledger、角色进程与 HTTP 聚合返回 RUNNING/DEGRADED/STOPPED/ABSENT；角色 supervisor 另有 FAILED 状态。规划、worker 就绪、router 就绪是执行阶段，不是额外的持久化状态值。相同配置且健康时重复 up 复用，同名不同配置或 degraded 时要求先 down。

### 4.1 内部 worker 与 router 命令

```bash
# 远程 worker 命令，由 CLI 生成环境与 namespace，不让用户手工抄写
CUDA_VISIBLE_DEVICES=0 nano-xpu _worker /models/Qwen3-0.6B \
  --role prefill --instance-id <cluster-id-p0> \
  --cluster-id <cluster-id> --engine-generation <new-generation> \
  --host 0.0.0.0 --advertise-ip 10.0.0.11 --port 8100 \
  --distributed-port 23000 --kv-port 24000 --kv-connector nixl-pull
```

这些 CLI 参数已实现。`--role` 默认 `colocated`；内部 worker 独立调试时自动生成 instance/generation，部署时由 controller 显式传入，D 另接收 P 的 instance=generation 列表。`Config/ModelRunner` 已使用实例独立共享内存名。参数可解析不代表生产 P/D 可运行。

```bash
# 合部：正常 HTTP reverse proxy；首版使用 round_robin
vllm-router --host 0.0.0.0 --port 8000 --policy round_robin \
  --worker-urls http://10.0.0.11:8100 http://10.0.0.12:8101 \
  --health-check-endpoint /ready --disable-retries
# 分离：上游原生两阶段路径；Python launcher 入口的参数合同需按固定提交验收
vllm-router --host 0.0.0.0 --port 8000 --vllm-pd-disaggregation \
  --kv-connector nixl --prefill http://10.0.0.11:8100 \
  --decode http://10.0.0.12:8101 --policy round_robin \
  --health-check-endpoint /ready --disable-retries
```

不启用 ZMQ discovery、DP 展开或 cache-aware policy，避免要求 nano-xpu 立即提供无关的 vLLM 统计接口。`--disable-retries` 是上游参数，**是否真正只尝试一次必须用实际 router 故障注入验证**，不能根据参数名保证无重放。上游启动时可能探测其他 metadata 路由，这些依赖同样属于集成 Gate。

## 5. 分离模式的请求和 KV 如何交接

### 5.1 HTTP 兼容不是 KV 兼容

**采用上游 NIXL pull 的 HTTP 协调合同，并为 nano-xpu 实现真实连接器。** 首版不修改 router 核心，也不假扮 Mooncake/NIXL 返回一个空描述符。NIXL API、内存注册和跨端握手的具体适配代码必须经过 M2 的 spike；无法满足合同则留在合部模式并回到此 RFC 更新，不能插入未声明的 PD gateway。

| 方向 | HTTP 语义 | nano-xpu 行为 |
|---|---|---|
| router → P | 相同聊天路由，`max_tokens=1, min_tokens=1, stream=false`；`do_remote_decode=true` | 计算 prompt KV，留存 lease，返回有效 `kv_transfer_params`，不在普通完成分支释放被导出的块 |
| P → router → D | pull 分支原样克隆 `kv_transfer_params`，D 请求仍保留原始 messages/max_tokens | D 从元数据定位 P，验证描述符并拉取 KV |
| D → client，经 router | 完整最终回答及 usage | prompt 只计一次，生成 token 预算不被 P 的内部一次采样扣减 |

P 返回的 `kv_transfer_params` 使用 `do_remote_prefill=true, do_remote_decode=false, remote_engine_id, remote_block_ids, remote_host, remote_port`；另带 `nano` 扩展，包含 `protocol_version=1, lease_id, generation, expires_at, cached_tokens, descriptor_digest`。`remote_host/port` 指向连接器控制通道，不能直接当 GPU 地址。完整 KVDescriptor 在受控握手中获取，最大 HTTP 元数据大小固定；不能把海量 block/tensor 数据塞进响应。

P/D 请求有独立严格 schema，并验证 router 产生的 `X-Request-Id`、阶段标志和同一部署凭据。colocated 保留拒绝 `kv_transfer_params` 的行为。`nano` 扩展字段是否被固定版本 router 完整透传，须以真实二进制测试确认。

router 给两阶段设置 Authorization 时会读取其 `OPENAI_API_KEY` 环境变量。本部署只向 router 注入新生成的**集群内部凭据**，workers 验证同一凭据；它不是用户的模型供应商 API key，不写入 plan、日志或状态文件。后端仅在可信部署网络可达。用户请求中的内部 KV 字段必须被拒绝或覆盖，不能据其读取任意地址。

### 5.2 首 token：D 重新计算最后一个 prompt token

**首个最终输出 token 由 D 采样，P 的内部输出不拼入答案。** 设模板输入长度为 L：P 可完成 L 个 token 的 Prefill，但导出的有效缓存前缀只标为 `max(L-1,0)`；D 导入该前缀，再对原始 prompt 的最后一个 token 做一次 forward，产生首个输出 token。后续正常 Decode。L=1 时导入前缀为 0，D 计算这一枚 prompt token。

这个选择避免跨进程交接 P 的随机采样状态，也不会对完整 prompt 重做 Prefill。P 的内部 `max_tokens=1` 结果仅满足上游调用形状，不作为最终文本；若固定版本 router 会拼接/依赖 P 的输出，则 M2 合同测试必须失败并修改方案，不能重复输出或漏 token。

D 保留完整 prompt token IDs、`num_prompt_tokens=L`，将 `num_cached_tokens=L-1` 和本地 block table 绑定到恢复序列；最后一枚 token 按现有 chunked-prefill 路径处理。不得使用 `Sequence.__setstate__` 直接恢复跨机序列。包含已缓存前缀的非整 block 尾部只写有效 token，未使用槽位不得参与 attention。

### 5.3 KVDescriptor 和 block 的所有权

| 字段组 | 必须携带或校验 |
|---|---|
| 身份 | cluster、lease、request、P engine generation、协议版本、原始 prompt token IDs 摘要 |
| 模型 | 权重/tokenizer/template 指纹、dtype、RoPE、层数、head 数与 head dim |
| 布局 | `[2, layers, blocks, block_size, kv_heads, head_dim]` 语义、有效 token 数、TP=1、tensor stride/字节数 |
| 传输 | 有界注册区域的 NIXL descriptors、远端逻辑 block 顺序、内容摘要/有效区间 |

源布局来自 [`ModelRunner.allocate_kv_cache`](../../nanoxpu/engine/model_runner.py)，不能套用其他运行时的 paged KV 格式。D 先分配自己的 block IDs，按**逻辑 token 位置**映射数据；P 的物理 block ID 不能直接成为 D 的 block table。首版不复用 D 的 prefix cache，也不把导入块发布到共享 prefix 索引，以减少恢复时的引用计数歧义。

当前 `LeaseTable` 保存的状态是 `EXPORTED / READING / ACKED / QUARANTINED`；释放后只留下身份墓碑，不保存 `RELEASED` 状态。P 计算和同步期间由原序列持有 block，随后增加 lease 引用、发布 descriptor，最后释放原序列引用；生产 NIXL 的注册步骤仍待实现。

**D 的顺序统一为：预留本地块 → pull → 等待计算可见 → 校验内容摘要 → ACK → 最后一个 prompt token 的 forward → Decode。** 这些是执行阶段，不是另一个已实现的状态枚举。ACK 证明 D 已持有独立且有效的数据副本，因此 P 无需等待 D 生成结束。

正常交接时，P 必须同时收到 ACK 和读取结束通知，才能释放对应源引用；两种通知可按任意顺序到达。同一 lease 的重复完成通知只释放一次。TTL 到期仅阻止新读取，**不是允许覆盖仍被 DMA 读取的内存**；进行中的传输必须完成或有可验证的取消/fence，无法确认时隔离该区域并将 worker 标为不可用，设备反注册与 DMA 停止仍需真实连接器验证。

每个 P 设置最大 lease 数与总 pinned-byte 预算；超限返回 429。D 仍只允许一个活动生成；busy 时返回 429，不静默换成在 D 上完整 Prefill。D 拒绝/断连且尚未 ACK 的 P lease 通过取消通知或到期流程回收，不能泄漏。控制事件由有界连接器控制线程处理，所有 Scheduler/KV 引用变化仍提交给引擎线程执行，不能只在新 HTTP 请求来时才清理。

## 6. 当前接口、代码落点与实施偏差

```text
nanoxpu/
  cli.py                     # 统一 serve 配置入口与内部 _worker 分发
  deployment/                # 已实现 CPU 可验证的控制路径
    schema.py                # profile + topology，拒绝未知字段
    planner.py               # 地址/卡/端口/版本解析，输出无副作用 plan
    ssh.py                   # 固定 helper + JSON stdin，受控远程调用
    controller.py            # 全节点 preflight、启动/探测/回滚与状态聚合
    helper.py                # 远端来源/资源检查与 supervisor 操作
    supervisor.py            # 每角色进程组、日志与有界停止
    state.py                 # 本地运行记录与所有权校验
  server/                    # 已实现角色 schema、server_info 与引擎线程调度
  engine/stage_engine.py     # 已实现阶段控制；目前由测试注入 CPU model/connector
  kv/descriptor.py           # 已实现版本协议与模型/layout 校验
  kv/connector.py            # 已定义 pull、完成、ACK/cancel/fence 合同
  kv/leases.py               # 已实现源块引用、预算、读取与到期清理
  kv/nixl.py                 # 尚不存在：待实现固定版本的真实设备后端
```

```python
# 当前 StageEngine.import_and_resume(self, seq, request) 的顺序摘要。
# 省略参数、校验细节与异常分支，不是可直接执行的示例。
descriptor = self.connector.describe(request.kv_transfer_params)
# 校验 peer/generation、cluster、request、layout、有效期与 prompt_digest
self._reserve(seq, len(seq) + request.max_tokens)
transfer = self.connector.pull(descriptor, self.model.storage, spans)
transfer.wait_compute_visible(self.transfer_timeout)
# 校验导入内容摘要；失败禁止 ACK 和 forward
self.connector.ack(descriptor)
seq.num_cached_tokens = descriptor.cached_tokens
self.scheduler.add(seq)
# Scheduler 驱动最后一个 prompt token 和后续 Decode；返回 Completion
```

实际方法是**同步所有者线程接口**，不是 `async def`；`EngineService.run()` 在专属 executor 中调用，HTTP 事件循环等待其结果。planner 的 `plan(config, profile)` 只生成 LaunchPlan，不调用 SSH。相关实现见 [planner](../../nanoxpu/deployment/planner.py)、[StageEngine](../../nanoxpu/engine/stage_engine.py) 和 [EngineService](../../nanoxpu/server/app.py)。

导入失败时，StageEngine 尝试取消源 lease，并分别证明传输和计算已停止；只有两者安全才释放 D 的块，否则保留 quarantine 并使实例不可服务。HTTP 断连不直接取消引擎所有者。当前 ACK 失败会终止本次生成并清理，**没有自动 ACK 重试**；重复通知幂等不等于已实现重试策略，也不能重新执行生成。

| 原描述或目标 | 当前实现与处置 | 剩余验收 |
|---|---|---|
| forward 后 ACK；另处写 ACK 后 forward | 统一为内容可见且校验成功后 ACK，再 forward；D 已有独立副本，避免无谓延长 P 的持有时间 | CPU 顺序/ACK 失败用例已覆盖；设备流可见性待验证 |
| async 阶段 API、独立 RouterLauncher | 改为真实同步 StageEngine 和 planner/controller/helper 的职责；避免文档虚构接口 | 真实 router 参数与故障合同待 Linux 执行 |
| cluster_uuid、额外部署/lease 状态 | 按实际 config hash、generation 和持久化状态记录；执行阶段与存储状态分开 | 设备启动和跨机恢复待验证 |
| preflight 验证显存、NIXL 与数据通路 | 当前只实现来源、文件、端口、卡号与部署预留检查；保留原设备要求 | T2-10、T2-16～18，不降低 Gate |
| down 成功删除状态 | 保留 STOPPED ledger，支持幂等查询和清理追溯；失败保留 CLEANUP_FAILED | 本地进程已测；真实 SSH/设备清理待验证 |

## 7. 故障、状态与停止

**首版禁止自动重放推理请求，停止只作用于本部署拥有的进程。** 输入错误、超容量、角色/版本不符在计算前拒绝；D 导入或 P 传输失败返回明确错误，不伪造成功、不静默转合部。重启后的 engine generation 必须变化，旧描述符一律失效。

| 情况 | 必须发生的行为 |
|---|---|
| 任一节点无法连接或未就绪 | up 非零退出，撤销本次新建进程；保留日志与失败原因 |
| router 存活但角色池不完整 | status 报 degraded；整体 ready=false，不把 /health 200 当成功 |
| P 已导出而 D 拒绝/崩溃 | 结束该 attempt；按取消/lease 到期及 fence 回收，不在另一个 D 盲目继续 |
| 生成响应丢失 | 不宣称 exactly-once；内部 attempt/lease 标识用于去重与清理，不缓存随机答案自动重放 |
| 控制机或 SSH 断开 | detach 集群继续运行；重连读取 supervisor 身份与状态，不重复创建 |
| down 遇到 IP 不通 | 能停的先停；本地记录仍保留不可达资源，非零返回，不能假报清理成功 |

down 顺序：停止 router 接收新生成 → 关闭 D（等待已有计算与清理）→ 关闭 P（完成传输/fence）→ 退出 supervisors。controller 已实现角色停止顺序，设备 drain/fence 仍是验收目标。超过 drain deadline 只对校验过身份的本集群进程组升级 TERM/KILL；禁止 `pkill python` 或仅凭复用的 PID 杀进程。清理失败保留可重试 CLEANUP_FAILED ledger，成功保留 STOPPED ledger。

## 8. 完整子任务清单、依赖与交付 Gate

**父任务同时包含 router 接入和 nano-xpu 真正的 P/D 分离，不能在 router 转发成功时提前关闭。** 下列 `T2-01`～`T2-19` 是本 RFC 的内部任务标识，尚未建立远程 Issue；验收栏定义目标，实际实施状态见第 9 节及 CPU 验证记录。每项可独立分配负责人、提交实现和收集验收证据，不预设人员或排期。

### 8.1 部署基础与 router 接入

| ID / 子任务 | 前置依赖 | 主要代码落点与交付 | 独立验收条件 |
|---|---|---|---|
| **T2-01：部署配置与无副作用 planner** | RFC-0001 CLI | `deployment/schema.py`、`planner.py`；统一 profile、两种拓扑、角色 IP/卡号、稳定端口和命令数组；增加 `serve plan` | **CPU / C01**：合法配置确定性输出；缺 router、重叠 GPU、端口冲突、未知字段均拒绝；不连接 SSH、不启动进程 |
| **T2-02：worker 实例与设备隔离** | T2-01 | `server/config.py`、`config.py`、`engine/model_runner.py`；新增 instance ID、generation、advertise IP、独立 rendezvous/共享内存命名；物理卡映射到逻辑 rank | **CPU + 设备**：两实例 namespace 不冲突；参数正确传递；设备资源释放归 T2-16 验收；不靠固定 `nanoxpu` 名称共存 |
| **T2-03：运行 profile、SSH executor 与 preflight** | T2-01 | `deployment/ssh.py`、profile 解析；固定 helper + JSON stdin；核对版本、权重、卡号、端口和可达性 | **CPU / C02**：stub SSH 可注入失败；不拼接用户 shell 字符串、不关闭 host-key 校验；真实主机 preflight 失败时零 worker 被启动 |
| **T2-04：supervisor、运行 ledger 与生命周期 CLI** | T2-01、T2-03 | `deployment/supervisor.py`、`state.py`、`cli.py`；`serve -f` 与 `serve status/logs/down`、detach、进程组、PID 身份与配置哈希 | **CPU / C02**：重复 up 不重复创建；配置变更拒绝；回滚只清理本次拥有的资源；SSH 断开仍能重新查询；无法清理时非零退出并保留 ledger |
| **T2-05：固定 vllm-router 依赖与启动适配** | T2-01 | router 版本/来源锁定、可执行文件检查、planner/controller/helper；生成 standard/PD 两套参数，不修改上游核心 | **真实 router / CPU**：固定二进制或 Python launcher 可启动；help/版本/参数合同有记录；支持正常关闭；不把同名包或 latest 当兼容证明 |
| **T2-06：角色、metadata、就绪与内部请求模型** | T2-02 | `server/schema.py`、`app.py`、`engine.py`；角色专用 schema、`/server_info`、`/ready`、内部凭据和能力协商 | **CPU**：colocated 拒绝 KV 字段；P/D 拒绝错误阶段/模型/generation；连接器未初始化不能 ready；metadata/凭据不泄露到普通响应或日志 |
| **T2-07：带 router 的合部编排** | T2-04、T2-05、T2-06 | planner + launcher + 状态聚合；完整 worker 池转发、统一 `router_url`；单 worker 也必须起 router | **真实 router + CPU workers / C03**：单 worker 和多 worker 都从统一入口访问；router 未就绪则 up 失败；不返回 worker 地址冒充服务入口；router 故障整体 degraded |

### 8.2 P/D HTTP 合同、KV 传输与阶段引擎

| ID / 子任务 | 前置依赖 | 主要代码落点与交付 | 独立验收条件 |
|---|---|---|---|
| **T2-08：锁定上游 P/D HTTP 协议** | T2-05、T2-06 | 合同测试与角色 schema；确认 P 请求改写、D 元数据透传、request ID、首 token 处理、错误/重试行为 | **真实 router + CPU P/D stubs / C03**：不是自制代理；KV 字段与原始 token 预算符合合同；P 内部输出不被重复拼入；错误不会意外重放生成 |
| **T2-09：KVDescriptor 与兼容性校验** | T2-06、T2-08 | `kv/descriptor.py`；版本、model/tokenizer/template 指纹、layout、有效 token 范围、logical block 映射与容量约束 | **CPU / C04**：序列化 round-trip；错误版本/形状/摘要/指纹拒绝；尾块、零缓存前缀和 L=1 不越界；不拿 P 的物理 block ID 直接索引 D |
| **T2-10：NIXL 后端 spike 与真实数据通路** | T2-02、T2-03、T2-09、T2-16 | `kv/nixl.py`、版本锁定与环境清单；注册、握手、pull、完成通知、取消/fence、受控反注册 | **GPU / G02 的传输子项**：独立进程间真实 KV tensor copy 及计算流可见性；有字节数/事件证据；故障中不提前反注册。输出可实现的 API 合同；不通过则更新 RFC，禁止假 NIXL |
| **T2-11：lease、引用计数与控制事件处理** | T2-09、T2-10 | `kv/connector.py`、block manager、引擎控制事件；pin 预算、ACK/cancel、重复通知、到期、隔离区域及 ledger | **CPU 故障注入 + GPU fence**：无新请求时也能回收；重复 ACK 不重复释放；TTL 不覆盖仍被 DMA 读取的数据；无法 fence 时隔离且不可继续服务 |
| **T2-12：Prefill-only 执行与 KV 导出** | T2-06、T2-11 | `engine/stage_engine.py`、scheduler、model runner；只完成 P 阶段，固定源块并返回真实 lease；适配 router 的 `max_tokens=1` | **GPU**：返回前写入/注册已完成；普通请求完成分支不会释放导出源；连续 Prefill 受 lease 预算限制；返回的 metadata 可定位到有效 KV |
| **T2-13：Decode 导入、序列恢复与首 token/usage** | T2-09、T2-10、T2-11、T2-12 | stage engine、Sequence、Scheduler；分配 D 本地块、导入前 L-1 个 token、计算最后一个 prompt token、继续 Decode，统一输出计数 | **GPU**：导入未完成前无 forward；不重做完整 prompt；P 内部采样不重复输出；L=1、max_tokens=1、EOS/length、尾块和模板一致；usage 与合部对齐 |
| **T2-14：P/D 集群编排与有序停止** | T2-04、T2-07、T2-08、T2-12、T2-13 | deploy controller、supervisor、ready/status 聚合；P/D 能力匹配、router 两阶段启动、失败回滚、生成/传输结束后停止 | **CPU 控制测试 + 设备**：缺 P/D、版本不符或 KV 通道不可用不能起分离集群；无静默退回合部；router 停流、D 结束、P fence/释放的顺序有证据 |

### 8.3 集成验证、设备验证与用户交付

| ID / 子任务 | 前置依赖 | 主要交付 | 独立验收条件 |
|---|---|---|---|
| **T2-15：CPU 集成与故障回归套件** | T2-01～T2-09、T2-11、T2-14 | 真实 router + 测试 workers/connector 的进程测试；覆盖 C01–C04，并保留 RFC-0001 的 43 项回归 | **CPU**：合部/分离控制流程、进程回滚、断连、429、重复 up/down、错误 generation 与内存所有权均通过；报告明确标注模拟 KV，不能据此宣布真实 PD 可用 |
| **T2-16：真实 CUDA 合部验收** | T2-07、RFC-0001 CUDA 单体 Gate | G01 的单卡、两副本记录；先验证单张 5090 + router，再验证不同卡/主机的副本隔离 | **GPU**：实际 `serve -f` 到 router 再到 Qwen；只有一张卡也经 router；停止一个副本不影响另一个；TP>1 仍需单独设备证据 |
| **T2-17：同机两卡真实 PD 验收** | T2-10～T2-16 | 同机 1P1D、两端 TP=1，真实 NIXL + Qwen；G02、G04 同机部分 | **GPU**：和合部数值/输出计数对照，证明 D 接续而非完整重算；传输中断、D 崩溃、到期、停止与资源释放通过 |
| **T2-18：跨机真实 PD 验收** | T2-17 | 两台计算节点 1P1D、控制机一键启动；G03、G04 跨机部分 | **两台 GPU 主机**：真实网络 KV 传输及首 token 正确；控制机/SSH 掉线和节点不可达时状态可信，清理 ledger 不丢；不承诺跨厂商/跨 TP |
| **T2-19：文档、配置与交付收口** | T2-15～T2-18 | 用户操作说明、单卡合部/同机 PD/跨机 PD 样例、版本锁定、测试报告与限制回填 | **发布 Gate**：两模式统一 serve 配置命令与 router 地址；所有实际兼容版本有证据；未通过项标未完成；不以启动脚本存在作为验收成功 |

“CPU”表示无需 GPU，不保证真实 router wheel 能直接在当前 Windows 环境运行；真实 router 合同测试需在 Linux CPU 节点或 CI 执行。无需设备的配置和控制测试先在本地完成；单元测试、实际进程测试和真实 router 集成分别记录，不能互相代替。

### 8.4 里程碑、依赖关系与停止条件

| 里程碑 | 子任务 | 能声明的交付 |
|---|---|---|
| M0：部署基础 | T2-01～T2-04 | 可规划、管理并隔离进程，不代表已支持 router 推理 |
| M1：router + 合部 | T2-05～T2-07、T2-16 | **一键合部且强制经过 router**；T2-16 通过后才声明真实 CUDA 合部已验收 |
| M2：P/D 协议和传输 spike | T2-08～T2-10 | 锁定控制/数据合同，具备真实 KV copy 证据，尚不能声明完整 PD 推理 |
| M3：阶段引擎与 KV 生命周期 | T2-11～T2-13 | P 导出与 D 接续路径具备可测试实现，完整部署仍待 M4 |
| M4：一键 PD 与交付 | T2-14、T2-15、T2-17～T2-19 | 所有 Gate 通过后，父任务才能关闭并声明一键合部/分离可用 |

上述依赖定义完整设备交付顺序；T2-16 不依赖完整 PD 回归，避免循环依赖。CPU 范围可先实现 T2-09、T2-11～13 的共享协议与状态控制，用明确标识的测试适配器验证，再回接 T2-10 的真实后端。这不算 T2-10 完成，也不解除 T2-17/18 的设备前置。当前采用此拆分，CPU 实现不是绕过 NIXL spike 的完整 M3 交付。

NIXL spike 不通过时停在 M2，允许保留已经验收的合部部署，但必须修改 RFC 后再继续 PD；不得用 mock connector 冒充 T2-10/T2-17。TP>1 的合部沿用独立设备 Gate；P/D 两端 TP>1 留到后续 RFC。

## 9. 验证计划与结果占位

| 编号 | 输入和执行 | 判定标准 | 实际结果/证据 |
|---|---|---|---|
| C01 | 两份拓扑、缺 router/计算角色、重叠卡、重复端口、坏 IP、TP 不匹配、未知字段 | plan 确定或明确拒绝；零远程副作用 | 42 项 planner 测试通过；见 [CPU 部署验证记录](../testing.md) |
| C02 | stub SSH 第二节点启动失败，重复 up/down，PID 被复用 | 仅回滚本次所有权内资源；不重复拉起/误杀 | controller 故障注入、实际 helper/supervisor/HTTP worker 通过；真实 SSH 尚未验收，未将整个 T2-03/04 标为完成 |
| C03 | **真实固定 router** + CPU test workers；单 worker 合部、多 worker 合部与 P/D 响应模拟 | 单 worker 合部也必须经过 router；router 未就绪整组失败；确认 worker /ready、KV 字段克隆、原始 token 预算、P 输出不拼接；断连与 429 不意外重试 | 角色 HTTP 单元测试通过；真实 router 尚未运行，本项未通过 |
| C04 | 纯 CPU tensor connector + 指纹/校验失败、尾块、L=1、M=1、重复 ACK、过期 generation | 不读越界、不提前释放、最终无活动 lease/reservation；不声称 NIXL 已验证 | 描述符、lease、周期维护及 StageEngine + CPU 数组传输测试通过；正向路径最终无活动资源，无法 fence 的故障路径明确保留隔离块，确认 fence 后才释放。真实设备连接器未验证 |
| G01 | 两个 CUDA 合部副本，经实际 CLI + router 发请求 | 都实际收到请求；停止一个不影响另一个；stdout/state 不泄露凭据 | 未执行，待设备 |
| G02 | 同型号两卡，先独立 NIXL KV tensor copy，再真实 Qwen 1P1D | 有实际传输字节/完成事件；D 没有重做完整 prompt；token/张量对照通过 | 未执行，待设备 |
| G03 | 跨机 1P1D，prompt 长度 1、block±1、多 block，输出预算 1/64 | 与合部采用相同模板；测试专用 argmax sampler 下 token IDs 对齐，生产随机采样另测分布/计数 | 未执行，待设备 |
| G04 | 导出后杀 D、传输中断、内存预算满、超时、控制机掉线、重复通知 | 无过早覆盖、无陈旧 lease 复用；清理 ledger 可解释；异常可重启 | 未执行，待设备 |

固定权重/tokenizer 快照、nano-xpu/router/NIXL/网络后端提交、GPU SKU、驱动、CUDA、dtype 和全部命令。记录 prompt/output token 数、端到端延迟 p50/p95（ms）、KV 字节数和传输耗时（ms）、活动/隔离 block 数、错误率；每项填实测数据与重复次数，当前均为待测。首版没有性能收益承诺，不以模拟曲线代替结果。

## 10. 反串讲、取舍与待决事项

**一个 100-token prompt 说明整条路径。** CLI 根据 p0/d0 的 IP 和卡号生成独立进程；router 选择 P，P 计算并固定至少前 99 个 token 的 KV。router 把 lease 元数据传给 D；D 分配自己的 block，拉取有效前缀，确认计算可见后 ACK，再计算第 100 个 prompt token、采样第一个输出。P 收到 ACK 且传输不再读源内存后释放；D 继续 Decode，最终由 router 返回一次完整回答。

为什么不能“仅接 router”？现有完整 generate 会释放 P 的 KV，也不会在 D 恢复序列；HTTP 两次调用只会得到两次独立推理。为什么先 TP=1？跨 TP 的 head 分片与 block 映射是另一个数据布局问题，不能靠多填几个卡号解决。代价转移到哪里？连接器、KV 固定内存、恢复状态与 supervisor；单请求策略仍可能让长 Decode 拒绝后续请求，这是当前明确的吞吐限制。

备选一是增加自建 PD gateway，可绕开上游合同，但会重复路由和错误处理，首版不选。备选二是只做 docker-compose，可覆盖单机，不能满足按远端 IP 一键起角色，首版不选。NIXL spike 若证明成本/可用性不合适，再单独评审 TCP Host Staging 与 router connector 扩展，不能以 `--kv-connector nixl` 名义运行假实现。

待决事项集中在 M2：锁定 NIXL/后端版本与内存注册 API、确认固定 router 启动依赖及异常重试、选定首批同型号 CUDA 设备并测定 lease 预算。它们阻止真实 PD 的交付声明；CPU 共享逻辑可按第 8.4 节先行验证，接入生产设备后端前必须回填 spike 结论。

当前用户已授权实施可在 CPU 验证的部分，并继续完成相应 CI；真实设备任务保持独立验收。当前工作区改动未提交、未推送；完成本地实现和验证后，按仓库发布规则提供当前版本预览，取得明确校核确认后再推送。

## 11. 评审与回填记录

2026-11-15 按用户要求复核模板和实施偏差：修正 ACK/forward 顺序、同步接口、真实状态与 namespace；区分历史基线、当前 CPU 实现和设备目标；移除已删除 HTML 的渲染记录及“全部验收未运行”的过期总述。C01/C04 与 C02 的本地部分已有结果，C03、真实 SSH 和 G01～G04 仍未验收；以 [CPU 验证报告](../testing.md) 的环境、版本和证据为准。

以上是 Agent 的实现一致性核查，不替代维护者正式接受方案，也不代表本工作区已经发布。若 NIXL spike 改变注册、ACK 或 fence 合同，须先更新本 RFC 和故障用例再接入设备后端。
