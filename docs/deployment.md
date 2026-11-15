# 部署控制与 CPU 验证边界

状态：CPU 功能实现与集成准备中。配置规划、进程管理、角色协议和阶段控制已有本地测试；真实 Linux SSH/router 集成尚待 CI。**真实 CUDA 合部和 NIXL P/D 未验收，生产 P/D 引擎明确拒绝启动。** 本说明不将测试 worker 当作可部署的模型后端。

## 配置到启动

公开命令统一为 `nano-xpu serve -f cluster.yaml`。单机合部、多副本和 P/D 均从配置启动且经过 router。旧 `deploy ...` 和 `serve MODEL` 返回迁移提示；worker 入口 `_worker` 只供编排器调用。`up` 在下文仅指 Controller 的启动操作。

从 [合部拓扑](../examples/deploy.md#合部部署) 或 [P/D 拓扑](../examples/deploy.md#pd-部署) 与 [运行 profile](../examples/deploy.md#公共-profile) 开始，将文档中的 YAML 分别保存为 `cluster-colocated.yaml` / `cluster-pd.yaml` 和同目录的 `qwen3-cuda-profile.yaml`，修改实际节点、模型目录、可执行文件及版本。相对 profile 路径以拓扑 YAML 所在目录为基准。样例的版本占位允许 `plan` 展示，但 `up` 会拒绝。

```bash
nano-xpu serve plan -f ./cluster-colocated.yaml
nano-xpu serve -f ./cluster-colocated.yaml --detach
nano-xpu serve status --name qwen-colocated
nano-xpu serve logs --name qwen-colocated
nano-xpu serve down --name qwen-colocated
```

`--name` 必须与拓扑中的 `name` 一致。`--state-dir /path/to/state` 可用于 `up/status/logs/down`，默认控制机状态目录为 `$NANO_XPU_STATE_DIR` 或 `~/.local/state/nano-xpu`。同一部署的后续命令必须使用相同目录。未加 `--detach` 时，`up` 返回就绪信息后继续监测，Ctrl+C 按顺序停止本次部署。

`plan` 只检查配置并输出命令数组，不连接 SSH 或创建进程。`up` 先检查所有节点，再启动 workers，核对模型指纹、实例与 generation，最后启动 router 并核对完整健康 worker 池；成功返回的是 router 地址。单 worker 也经过 router。

## 主机与版本前提

- 节点使用 Linux，模型和运行环境预先安装；controller 不代替软件部署或模型分发。
- 在指定的干净 Git checkout 安装对应运行依赖及 `nano-xpu[server,deploy]`。部署 preflight 要求实际 worker 和 helper 导入同一 checkout，HEAD 等于 `expected_nano_xpu_revision`，`nanoxpu/` 与 `pyproject.toml` 没有未提交修改。单独安装的 wheel 当前没有可接受的构建来源信息，不能通过这一检查。
- `runtime.helper_python` 指向该环境的绝对 Python 路径，`runtime.nano_xpu_bin` 指向对应入口。仅复制源码到 workdir、实际却运行其他环境的同名包会被拒绝。
- router 使用 [固定上游源码](https://github.com/vllm-project/router/tree/07c20da66c255ba65a97ab531f9fb5f60413a7d4)构建；`.github/ci/build_router.py` 生成二进制及相邻的 `.source.json` 来源文件，部署 preflight 检查源码提交和二进制 SHA256。该脚本的 dev build 用于正确性验收，没有性能声明。
- SSH 必须能够非交互认证，且目标主机密钥已受信任。`ssh.identity_file` 和可选 `ssh.known_hosts_file` 是控制机上的路径；不配置后者时沿用 OpenSSH 默认 known_hosts。始终启用 `BatchMode=yes` 和 `StrictHostKeyChecking=yes`。

同一模型目录名不代表内容相同：preflight 对权重/config、tokenizer、模板与 RoPE 内容计算摘要，各 worker 必须一致；启动后再与 `/server_info` 报告比较。检查会读取模型文件，大模型应据实际耗时设置 `startup_seconds`。

## 重复操作、故障与恢复

同配置且健康的重复 `up` 复用已有进程；配置变化或记录显示部分失败时拒绝直接覆盖，先查询状态并执行 `down`。SSH helper 返回后，独立 supervisor 继续持有 worker；新的 CLI 进程可以读取 ledger 查询、取日志或停止。

启动失败只回滚记录中属于本次部署的进程。停止顺序是 router、D/合部 worker、P；信号发送前核对 PID、创建时间、用户、命令摘要和 owner，避免误杀复用的 PID。节点不可达或未能完成清理时保留记录并非零退出，恢复连通后重试 `down`，不要删除 ledger 来掩盖残留资源。

内部凭据经 SSH stdin 传入，不写进公开计划或 ledger；日志读取有上限并脱敏。`status` 的 `ready` 同时取决于远端进程身份、worker 就绪和 router 健康池；进程存在不等于集群可用。

## CPU 如何验收

按照 [CPU CI 指南](ci.md)运行两套测试：`tests/` 验证共享生产控制代码；`tests/integration/` 使用真实 SSH、router、HTTP 和独立进程，worker 明确替换为测试专用实现。不会向生产 `serve` 增加模拟开关。

CPU P/D 阶段测试使用真实 Scheduler/BlockManager 和 NumPy KV 数组，检查源块保留、目标块独立分配、复制可见后才计算、首 token/usage、错误与回收。它不能证明设备 DMA 已停止、CUDA stream 可见性、Qwen 数值正确或 NIXL 可用。完整 RFC 的设备 Gate 仍保留，详见 [RFC-0002](rfcs/0002-router-oneclick-pd.md) 和 [当前验证记录](testing.md)。

## 从旧包名迁移

当前源码包为 `nanoxpu`，发行包为 `nano-xpu`，CLI 仍为 `nano-xpu`。Python 用户将 `from nanovllm ...` 更新为 `from nanoxpu ...`，模块命令改成 `python -m nanoxpu ...`；不提供旧导入别名。

升级前用旧版本的停止命令关闭已有服务。然后在每个受控 Python 环境中卸载旧包（`python -m pip uninstall nano-vllm`），进入新提交的 checkout，执行 `python -m pip install -e '.[server,deploy]'`，并更新 profile 的预期提交。用 `python -m nanoxpu runtime-info` 核对每个节点的实际包路径和提交，再 plan/启动。不要混用新旧节点、从旧 wheel 或旧 build 目录继续启动；升级回退也必须整组停止并统一提交。旧 Python pickle 中的模块名不保证兼容，活动进程和旧部署 ledger 不做原地迁移。

发行包改名不表示已发布到 PyPI；本仓安装仍使用 checkout 或明确构建的 wheel。作者、许可证与上游来源不变。
