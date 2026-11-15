# 测试组织

测试按 nanoxpu 的 server/engine/deployment/kv 职责分组；根级模块测试位于 tests 根目录，共享工具放 helpers。配置和脚本以 examples 的两份 Markdown 为来源，提取到临时目录后执行。

运行 `python -m pytest -q tests --strict-markers`，包含基础控制层、真实本地子进程、HTTP、Scheduler、KV 协议和阶段测试；CPU model/connector 不是生产后端。Linux router/SSH 测试与 CI 在下一 PR 加入，设备验收未完成。
