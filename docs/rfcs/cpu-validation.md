# S0 验证记录：服务入口、切分参数与本地 CPU 测试

- 日期：2026-11-13
- 基线：`40a001f99181321d8b2ab30a0b8a1cfcdba0c9d5`，验证对象为本次未提交的工作区实现。
- 环境：Windows x86_64，CPython 3.12.13；依赖安装在仓库 `.cpu-deps`，未安装 torch、transformers、triton、flash-attn。
- 依赖：FastAPI 0.136.3、Uvicorn 0.48.0、Pydantic 2.13.4、HTTPX 0.28.1、pytest 9.0.3、NumPy 2.4.6、xxhash 3.6.0；运行版本固定在 [requirements-cpu.txt](../../tests/requirements-cpu.txt)。
- 安装：`uv pip install --target .cpu-deps -r tests/requirements-cpu.txt` 与 `uv pip install --target .cpu-deps --no-deps -e .`，两者均显式使用上述 Python 3.12 解释器。
- 执行：`python -m pytest -q tests --junitxml=docs/rfcs/cpu-test-results.xml`，使用上述解释器和指向 `.cpu-deps` 的 PYTHONPATH。
- 结果：**43 passed in 6.39s**，无失败、无跳过；[JUnit 原始报告](cpu-test-results.xml)。耗时仅为本次测试运行时间，不是推理性能指标。

| 范围 | 实际观察 | 证据 |
|---|---|---|
| CPU-01 / CLI | 服务模块与 help 不加载推理依赖；安装入口可加载到 main | `test_imports_and_help_do_not_load_cuda`、`test_console_entry_point` |
| CPU-02 / 切分配置 | TP=2、token 预算、序列数、通信端口传到适配器；非法参数拒绝 | `test_cli_rejects_invalid_configuration`、`test_tp_chunk_budget_and_template_reach_engine` |
| CPU-03 / 路由 | 四个接口工作；连续 100 次合成生成成功，usage 正确 | `test_routes_and_lifecycle` |
| CPU-04 / 错误 | 校验 400、错模型 404、忙碌 429、故障 500/503；输入错误后可恢复 | `test_request_validation`、`test_fatal_generation_marks_unhealthy` |
| CPU-05 / 断连 | 取消等待请求后引擎任务继续；结束前保持忙碌，结束后接受新请求 | `test_busy_and_cancelled_connection_keep_slot_until_engine_finishes` |
| CPU-06 / 生命周期 | 初始化、生成、关闭在同一专属线程；启动失败也关闭；watchdog 限时非零退出 | `test_start_failure_is_closed`、`test_shutdown_watchdog_bounds_stuck_process` |
| CPU-07 / 真实 HTTP | 独立子进程监听本地端口，实际执行 PowerShell 启动/调用脚本及 Bash 调用示例的 Python 部分，正常退出后同端口再拉起成功 | `test_real_http_subprocess_and_same_port_restart` |
| CPU-08 / 原调度器 | Prefill 分块为 4、4、2，随后 Decode；结束后活动 block 引用归零 | `test_chunked_prefill_and_block_release` |

测试文件：[服务层](../../tests/test_server.py)、[适配器边界](../../tests/test_engine_adapter.py)、[原 Scheduler](../../tests/test_scheduler_cpu.py)。测试引擎输出明确标记 `[CPU TEST ENGINE: no Qwen model loaded]`，其 token 计数是合成值，不是 Qwen tokenizer 结果。

Windows Application Control 拒绝启动新建虚拟环境中的 python launcher 和安装生成的 `nano-xpu.exe`。已用原 CPython 3.12 解释器执行等价 CLI 路径，并验证 console entry point 安装元数据；**没有声称原生 `.exe` launcher 运行通过**。该结果不阻塞服务层 CPU 测试，但仍需在目标 Linux/CUDA 环境验证实际 `nano-xpu serve` 命令。

尚未验证：真实 Qwen 模型加载/生成、数值对齐、CUDA Graph、NCCL 端口行为、TP=2 计算与子进程清理、真实 KV 容量与设备 OOM、SIGTERM 后 GPU 资源释放、吞吐/延迟。CPU 测试通过只完成 RFC 的 S0，不关闭整个服务化 Issue。

设计 HTML 已生成并通过静态锚点、链接及内嵌脚本检查。视觉检查未完成：Playwright 禁止 file 协议访问，启动本地预览 HTTP 服务又被自动审批策略拒绝，返回 blocked by policy。该限制不涉及上述已运行的服务层功能测试。
