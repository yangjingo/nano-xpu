# 配置驱动的服务入口

当前公开入口为 `nano-xpu serve -f cluster.yaml`，内部 `_worker` 只供编排调用。旧 deploy 命令和直接模型参数明确拒绝。

将[配置示例](../examples/deploy.md)的拓扑与 profile 代码块保存到同一目录，填写实际主机、模型路径、程序路径及版本。先用 `nano-xpu serve plan -f cluster-colocated.yaml` 检查；plan 不连接 SSH，也不启动进程。安装依赖使用 `.[server,deploy]`。

实际启动需要 Linux 节点、预先安装的模型环境、受信任的 SSH 主机密钥以及固定版本 router。helper 与 worker 都核查实际导入代码和干净 checkout 的提交。先检查所有节点，再启动 workers，最后检查 router；成功返回 router URL。前台 Ctrl+C 停止集群；加 --detach 则在 ready 后返回，用 serve status/logs/down --name 管理同一 ledger。

[服务代码示例](../examples/server.md)包含 CPU harness 与请求检查。CPU 输出和模拟传输不表示真实模型、CUDA/NCCL 或生产 NIXL/P/D 通过。基础测试执行 `python -m pytest -q tests --ignore=tests/integration --strict-markers`；真实 router/SSH 由 Linux CI 单独执行 tests/integration，尚无本次远程结果。
