# 测试组织与运行

测试统一放在 `tests/`，按受测职责组织；真实跨组件验证另设 `integration/`，运行环境通过前置检查说明，不在仓库根目录维护另一套测试树。

| 源码 | 测试 | 验证范围 |
|---|---|---|
| `nanoxpu/cli.py` | `tests/test_cli.py` | 公开配置入口与生命周期分发 |
| `nanoxpu/runtime_identity.py` | `tests/test_runtime_identity.py` | 实际导入路径、提交和代码清洁状态 |
| `nanoxpu/server/` | `tests/server/` | HTTP、请求校验、引擎适配、角色协议与服务维护 |
| `nanoxpu/engine/` | `tests/engine/` | Scheduler、阶段执行、实例与子进程生命周期 |
| `nanoxpu/deployment/` | `tests/deployment/` | 配置规划、Controller、helper/supervisor、ledger |
| `nanoxpu/kv/` | `tests/kv/` | 描述符、lease、预算与回收协议 |
| 多个模块协作 | `tests/integration/` | 真实 Linux SSH、固定源码 router、HTTP StageEngine |
| 无生产对应模块 | `tests/helpers/` | 共享测试数据、CPU model/connector、HTTP/进程夹具与测试 worker |

测试优先镜像源码模块，再按可观察行为命名；例如 `server/engine.py` 对应 `server/test_engine.py`，`server/app.py` 对应 `server/test_app.py`。一个行为跨多个文件时可以用专题名，不要求每个源码文件都有一个测试文件。暂未有独立测试的 `layers/`、`models/`、生产 `utils/` 不创建空目录。共享夹具用 helpers，避免被误认为生产 utils 的测试。

共享能力从 `tests.helpers` 导入，不从另一个 `test_*.py` 导入。只被单个文件使用的夹具留在该文件；会启动子进程的测试保留实际进程和网络边界，不以目录整理改变断言或验证范围。

在仓库根目录，按 [CI 指南](../docs/ci.md) 安装 CPU 依赖后运行：

```bash
# 基础 CPU：无需真实 router/sshd。
python -m pytest -q tests --ignore=tests/integration --strict-markers
# Linux 集成：先准备固定版本 router、sshd、sudo 和环境变量。
python -m pytest -q tests/integration --strict-markers
```

CI 的 `cpu` 和 `cpu-integration` 分别完整执行这两个互不重叠的范围，`ci-gate` 要求两者及 static 全部成功。缺少 Linux/router/SSH 前置时明确失败，不用 skip 得到绿色结果。直接 `pytest tests` 会包含所有集成用例，因此普通 CPU 开发环境应使用上面的第一条命令。

## 借鉴依据与边界

本轮实际对照 `code/vllm`，提交 **`80e0cd68f812bfef5e6e7cffef3350daa01ff222`**，核查时工作区干净；Nano-XPU 基线为 **`2d61a9c8aa6eaf7eb6a928b1331f67418b5bd79b`** 加 `feat/rfc-cpu-deployment` 未提交改动。没有假设其他 vLLM 副本版本相同。

- [vLLM tests](https://github.com/vllm-project/vllm/tree/80e0cd68f812bfef5e6e7cffef3350daa01ff222/tests) 按 engine、entrypoints、distributed 等模块分组，并有共享工具。本项目借鉴模块组织和工具复用；目录名按自己的 nanoxpu 源码对应，使用 server 而不照搬 entrypoints，另保留实际需要的 deployment/kv/integration 范围。
- [entrypoints CI](https://github.com/vllm-project/vllm/blob/80e0cd68f812bfef5e6e7cffef3350daa01ff222/.buildkite/test_areas/entrypoints.yaml) 与 [engine CI](https://github.com/vllm-project/vllm/blob/80e0cd68f812bfef5e6e7cffef3350daa01ff222/.buildkite/test_areas/engine.yaml) 显式选择测试路径；本项目使用 GitHub Actions 的两套完整路径选择，不引入 Buildkite 分片。
- vLLM 的 `pyproject.toml::tool.pytest.ini_options` 还定义了 cpu_test 等标记。本项目当前用目录即可明确覆盖范围，暂不再叠加一套重复分类标记。

整理验收：迁移前后用例集合相同，基础与 Linux 集成无交集且合并后覆盖全部用例；基础回归、脚本启动、独立 checkout 来源检查和 CI 路径都通过。真实 Linux 集成仍须实际执行，collect-only 只证明可收集。实际结果集中于 [验证记录](../docs/testing.md)。

示例以 `examples/` 的 Markdown 为唯一来源。共享 helper 提取带 filename 的代码块到 pytest/CI 临时目录；配置解析、源码外安装包 plan 和 HTTP 请求均使用提取结果。examples 不保存可执行副本，不向源码目录生成脚本。
