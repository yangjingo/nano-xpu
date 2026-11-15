# CPU 基础 CI

状态：2026-11-15 已完成本地实现与适用检查；GitHub 托管运行和分支保护尚未验收。

## 目标与执行路径

每个 PR 都应得到一份明确的基础功能结论。第一阶段在 Ubuntu 24.04、Python 3.12 上运行全部 CPU 测试，不按路径跳过文档 PR，也不依赖 GPU、模型下载或私有 Secret。当前使用 `tests/requirements-cpu.txt`，仅用 `--no-deps` 安装本项目 wheel；这不是生产推理环境。

工作流 `.github/workflows/ci.yml` 响应 PR、main push 和手动运行。`static` 执行 Ruff 的错误检查、Python 编译检查、工作流语法检查及 CI 校验器测试；`cpu` 构建并安装 wheel、检查安装后的 CLI、运行 `tests/`，生成 JUnit 报告。两个任务独立执行，`ci-gate` 等待两者，只接受两者均为 `success`。失败、取消或任务被跳过都不能得到成功汇总结论。

CPU 测试必须收集到用例，且没有失败、错误或跳过。JUnit 校验不会替代 pytest 的退出码。报告和子进程日志随本次运行上传，保留 7 天；安装或收集阶段失败可能没有完整报告，但仍必须使任务失败。只缓存 pip 下载，不缓存测试结果。新增 CPU 用例放进 `tests/test_*.py`；未来设备测试应另建套件，不能通过在本套件中大量 skip 来接入。

当前范围覆盖 CLI、HTTP 请求校验、服务生命周期、错误恢复与拒绝、真实 HTTP 子进程关闭及重启、适配器参数传递，以及 Scheduler 的分块 Prefill 和 block 引用释放。模型计算由测试引擎或桩替代，不证明 Qwen 数值正确、CUDA Graph、TP/NCCL、跨机 P/D 或吞吐收益。

## 参考与取舍

对照基线：Nano-XPU `53e77bb993291ccc28d8e3c943ab618687aa32e2`，本次 CI、模板、RFC 文档为未提交工作区修改；另有用户已有 roadmap、RFC-0002 与示例修改，未据此声称相关能力已经实现。

- vLLM `730194d2362716c47648d06e172c91bcdace3b32`：[pre-commit.yml](https://github.com/vllm-project/vllm/blob/730194d2362716c47648d06e172c91bcdace3b32/.github/workflows/pre-commit.yml) 将便宜的静态检查独立运行，[engine.yaml](https://github.com/vllm-project/vllm/blob/730194d2362716c47648d06e172c91bcdace3b32/.buildkite/test_areas/engine.yaml) 声明领域测试与依赖。本仓先保留独立静态任务、可本地执行的原生测试命令，不引入 Buildkite 或路径选测。
- SGLang `40723a52ded9229e221a437254f34bc8a3327251`：[pr-test.yml 的 pr-test-finish](https://github.com/sgl-project/sglang/blob/40723a52ded9229e221a437254f34bc8a3327251/.github/workflows/pr-test.yml) 在任务结束后集中检查结果。本仓借鉴固定汇总任务，但所有 CPU 基础任务必跑，因此采用更严格的“全部 success”，不允许 skipped。

不复制上游的设备矩阵、机器人授权、分片和夜间任务。当前 43 项基础功能测试规模很小，选测系统会增加维护成本。依赖的直接版本固定，但传递依赖尚未做完整锁定；CI 输出实际安装列表，后续若遇到依赖漂移再引入统一 lock。

## Agent 反串讲

例如 PR 修改了请求取消逻辑：GitHub 检出该次待测提交，CPU job 安装其 wheel，pytest 发起一次请求并取消等待，测试引擎继续运行；用例确认后台完成前不会放出槽位。pytest 把断言结果写入 JUnit，校验器拒绝空跑、失败或跳过，job 结果再交给 `ci-gate`。报告只属于当前 run/attempt，不从缓存读取旧的通过结果。

为什么有效？它用可重复的请求与状态变化检查基础行为，并把错误沿退出码传到汇总。哪里无效？桩不执行模型或设备操作。代价在哪里？所有 PR 都运行便宜测试，换来较少的选测规则。怎样证伪？注入断言失败、空报告、skip 和取消状态，任何一种还能得到通过就是 CI 缺陷。剩余关键环境差异是 Windows 本地验证与 Ubuntu 托管运行，需要上线后记录真实运行链接。

## 验收设计与结果

| 用例 | 输入与步骤 | 通过标准 | 实际结果 |
|---|---|---|---|
| CPU-01 | 安装 CPU 依赖，构建并安装 wheel；运行全部 `tests/` | 用例非空、零失败、零错误、零跳过 | Windows / CPython 3.12.13；全新依赖目录安装 wheel，43 passed，4.22s；JUnit 校验通过 |
| CPU-02 | 在源码目录外执行安装后的 CLI help | 入口可运行，不导入 CUDA 核心依赖 | wheel 来源已确认；模块入口通过，四个 CUDA 核心依赖均未安装；Windows console launcher 被 Application Control 阻止，Linux launcher 待托管验证 |
| CPU-03 | Ruff、compileall、actionlint | Python 与工作流的基础错误检查通过 | Ruff 0.14.4、compileall、actionlint 1.7.12 通过；本地无 ShellCheck/Pyflakes，actionlint 的对应外部检查未运行 |
| CPU-04 | 校验成功、失败、错误、空、skip、缺失和损坏 JUnit | 仅非空且全成功的报告通过 | 6 个 unittest 通过，包含 failure/error/skipped 子用例及部分 skip；真实 43 项报告接受 |
| CPU-05 | 将汇总的两个依赖分别设为 success/failure/cancelled/skipped | 仅 success + success 返回 0 | 从最终 YAML 提取原始 Bash，16 种组合均符合预期；非 GitHub 调度实测 |
| CPU-06 | GitHub 上普通 PR、文档 PR、main push；检查报告下载和 required check | 均有结论，失败阻止合入，结果对应当前提交 | 未执行；本地准备后需用户确认发布并配置仓库规则 |

本地干净安装首次复测遗漏创建 `artifacts/`，出现 5 passed / 38 errors；原因是 `--basetemp` 的父目录不存在。工作流已有 `mkdir -p artifacts`，已修正文档的复现命令并重跑得到上表结果。工作区证据保存于 `../../output/ci-workflow-audit-20261115/`（不随仓库发布）：`cpu-tests.xml`、初次失败报告、`ci-validation.json`。3 份 Issue 模板元数据、PR 模板、36 个本地文档链接和 `git diff --check` 均已检查。耗时只是单次本地正确性运行记录，不是性能结论。

## 上线与维护

标准 Linux / Python 3.12 环境中，在仓库根目录复现主要命令：

```bash
python3.12 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install -r ci/requirements.txt -r tests/requirements-cpu.txt
python -m ruff check .
python -m compileall -q nanoxpu tests ci
python -m unittest discover -s ci -p 'test_*.py' -v
python -m build --wheel --no-isolation
python -m pip install --no-deps dist/*.whl
(cd /tmp && nano-xpu serve --help && python -m nanoxpu serve --help)
mkdir -p artifacts
python -m pytest -q tests --strict-markers --junitxml=artifacts/cpu-tests.xml --basetemp=artifacts/pytest
python ci/check_junit.py artifacts/cpu-tests.xml
```

工作流语法另用 actionlint 1.7.12 检查，工作流内包含下载与 SHA256 校验命令。本机 Windows 的解释器限制和可用命令见 [CPU 本地验证](rfcs/README-cpu.md)。Ruff 当前只启用语法和确定性错误规则，不强制整仓格式化。新建 job 时必须同步 `ci-gate.needs` 和成功判定，避免遗漏。

本地可审阅的工作流与模板准备完成后，按仓库发布规范校核并推送。在 GitHub 完成一次成功运行后，把 job 名 `ci-gate` 配为 main 的 required status check，并确认来源为 GitHub Actions；YAML 自身不会设置分支保护。Issue 模板进入默认分支后才出现在新建选择页。当前不配置自动合入、远程发布、标签或人员指派。

PR 只取消同一 PR 的旧运行；push/手动运行使用各自 run ID。job 有超时限制。任务使用 GitHub 托管 runner 和只读仓库权限，checkout 不持久化凭据。Actions 固定到完整提交；更改固定版本时重新验证。若以后启用 merge queue，需补充 `merge_group` 事件及其验收；本版没有宣称支持队列合入。
