# 部署配置示例

同一个模型可能先在一张卡上验证，随后扩展为多个副本，或者尝试把 Prefill 与 Decode 放到不同节点。如果每次都在启动脚本中修改模型路径、IP、卡号和端口，很容易让运行环境与部署拓扑混在一起，难以看清这次究竟准备启动哪些进程。

Nano-XPU 用两份 YAML 描述一次部署：公共 profile 记录“模型和程序从哪里加载、运行时用哪些参数”，拓扑记录“router 和每个 worker 放在哪个节点、使用哪些卡”。切换合部与 P/D 时可以复用相同结构的 profile，再选择相应拓扑。它们最终都交给 `nano-xpu serve -f`，本页没有另一套 deploy 启动命令。

第一次使用时，先保存下面的 `qwen3-cuda-profile.yaml`，再选择一种拓扑保存到同一目录。拓扑的 `profile` 相对路径以拓扑 YAML 所在目录为准。代码块中的节点地址、程序路径和版本占位都是示例，需要按实际环境填写；服务的启动与请求步骤接着看 [server.md](server.md)。

目前本仓已验证配置规划和部分 CPU 控制路径；真实 Linux SSH/router、CUDA 模型服务和生产 NIXL/P/D 仍有未完成验收项。尤其是 P/D 配置，现在可用于规划和协议开发，不能直接视为已具备真实分离推理能力。

## 公共 profile

profile 把同一组 workers 应共享的运行条件放在一起。`model_path` 是节点上的模型目录，`served_model_name` 是客户端请求使用的名字。`runtime` 中的 Python、CLI 和 checkout 路径必须指向同一套安装；`expected_nano_xpu_revision` 填实际使用的完整 Git commit，启动时会核查实际导入的代码，而不只检查目录名是否相同。

先在各节点准备模型和运行环境，再填写这份配置。SSH 的私钥和 known_hosts 是控制机读取的文件，节点需要允许非交互登录，主机密钥也需预先核验。router 的可执行文件则位于 router 节点，须有与固定源码版本匹配的构建来源文件。具体安装和来源检查要求见[部署说明](../docs/deployment.md#主机与版本前提)。

将下面代码保存为 `qwen3-cuda-profile.yaml`：

```yaml filename=qwen3-cuda-profile.yaml
# Deployment profile example. Prepare runtime/model/SSH and pin real revisions before starting.
schema_version: 1
model_path: /models/Qwen3-0.6B
served_model_name: qwen3-0.6b
runtime:
  nano_xpu_bin: /opt/nanoxpu/bin/nano-xpu
  workdir: /opt/nanoxpu
  expected_nano_xpu_revision: "<implementation-commit-to-be-pinned>"
  helper_python: /opt/nanoxpu/bin/python
ssh:
  user: inference
  port: 22
  identity_file: ~/.ssh/id_ed25519
  # Optional controller-local file; strict host-key checking is always enabled.
  known_hosts_file: ~/.ssh/known_hosts
router:
  executable: /opt/nanoxpu/bin/vllm-router
  source_revision: 07c20da66c255ba65a97ab531f9fb5f60413a7d4
  policy: round_robin
  port: 8000
engine:
  max_model_len: 4096
  max_num_batched_tokens: 2048
  max_num_seqs: 1
  gpu_memory_utilization: 0.9
  enforce_eager: true
kv:
  connector: nixl-pull
  expected_version: "<M2-spike-must-pin-nixl-and-network-backend>"
  lease_seconds: 180
timeouts:
  startup_seconds: 600
  request_seconds: 120
  drain_seconds: 30
```

`engine` 设置每个 worker 的模型上下文、调度容量和显存使用参数；这些数值仍要结合实际设备和负载调整。`router` 设置公共入口端口和分发策略。`timeouts` 分别约束启动等待、请求和停止等待，不能用延长超时代替定位节点错误。

`kv` 描述 P/D 模式需要的传输连接与 lease 参数。示例保留了尚未完成设备验证的 NIXL 版本占位，供后续实验填写。合部规划不会要求解析这个 P/D 版本占位；P/D 规划会把它列为未解决项。即使填入一个字符串，也不代表相应后端已经实现或验收。

## 合部部署

合部意味着每个 worker 都负责一条请求的 Prefill 和 Decode。下面的例子有两个完整模型副本：`c0` 使用 `10.0.0.11` 的物理卡 0，`c1` 使用 `10.0.0.12` 的物理卡 0；router 在 `10.0.0.10` 接收请求并分发给 workers。两个节点各自的 `[0]` 是本机卡号，不是跨节点共享的一张卡。

将代码保存为 `cluster-colocated.yaml`，与公共 profile 放在同一目录：

```yaml filename=cluster-colocated.yaml
# serve plan can validate this example; replace profile placeholders before starting.
schema_version: 1
name: qwen-colocated
profile: ./qwen3-cuda-profile.yaml
mode: colocated
router:
  ip: 10.0.0.10
workers:
  - {id: c0, role: colocated, ip: 10.0.0.11, gpus: [0]}
  - {id: c1, role: colocated, ip: 10.0.0.12, gpus: [0]}
```

先运行计划检查。这个命令只读取和校验配置，输出待启动的命令与地址，不连接 SSH，也不创建服务进程，所以可以先在没有 GPU 的开发机上核对拓扑：

```bash
nano-xpu serve plan -f ./cluster-colocated.yaml
```

检查输出中的 `router_url`、workers 的节点和卡号，以及 `unresolved`。示例默认的 runtime commit 尚未填写，因此 plan 可以输出计划，但实际启动会拒绝未解决的配置。`unresolved` 为空后，还需要满足节点环境与来源检查；计划通过不保证远端环境已经可用。

第一次只验证一张卡时，可以删去 `c1`，保留 `c0` 和 router。两者也可以放在同一台机器上，只要地址、端口和运行环境符合配置；单 worker 仍使用 router 作为公开入口。如果把两个 workers 放到同一主机，须为它们分配不重叠的 GPU。更多卡并不自动表示多副本：同一个 worker 的 `gpus` 包含多张卡时，还会涉及该 worker 的张量并行与相应设备验收。

确认配置和环境后，按[服务启动说明](server.md#qwen-配置启动)执行 `nano-xpu serve -f ./cluster-colocated.yaml`。启动成功返回的 router 地址用于发送请求，后续查询与停止使用这里的部署名 `qwen-colocated`。

## PD 部署

P/D 分离把两个计算阶段交给不同的 workers。Prefill 先处理输入提示并产生 KV 状态，Decode 接收这些状态后继续逐 token 生成；因此除了能访问 HTTP 端口，还需要两端模型、tokenizer、布局和 KV 传输能力相互匹配。仅把角色名从 colocated 改成 prefill/decode，并不能让缺少的传输后端自动出现。

下面的拓扑把 P 放在 `10.0.0.11`，D 放在 `10.0.0.12`，客户端仍然访问 router。保存为 `cluster-pd.yaml`，继续引用同目录的公共 profile：

```yaml filename=cluster-pd.yaml
# serve plan validates this topology; production P/D backend remains unimplemented.
# Same host is also valid with distinct physical GPUs, e.g. P=[0], D=[1].
schema_version: 1
name: qwen-pd
profile: ./qwen3-cuda-profile.yaml
mode: disaggregated
router:
  ip: 10.0.0.10
workers:
  - {id: p0, role: prefill, ip: 10.0.0.11, gpus: [0]}
  - {id: d0, role: decode, ip: 10.0.0.12, gpus: [0]}
```

使用下面的命令查看角色计划，并核对 P、D 的命令、地址以及 `unresolved` 中提示的 runtime 和 KV 版本事项：

```bash
nano-xpu serve plan -f ./cluster-pd.yaml
```

同机实验也可以把 P、D 配置为相同 IP，并分别使用 `[0]` 和 `[1]` 等不重叠的卡。当前 P/D 配置约束为每个阶段 TP=1；跨机时还需要另外验证网络可达性、传输与资源回收，CPU 协议测试不覆盖这些设备条件。

当前生产引擎会拒绝尚未实现的 P/D 路径，因此本节的可执行检查止于配置规划。后续真实后端与环境完成后，仍使用相同的 `serve -f` 入口启动，并通过[请求示例](server.md#qwen-请求检查)检查服务，再补齐设备正确性与生命周期验收。这样配置表达的目标拓扑、代码当前支持的范围和实际验证结果可以分别追踪。
