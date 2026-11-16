# CPU 功能与集成 CI

状态：2026-11-16。原基础工作流在 [main 的运行](https://github.com/yangjingo/nano-xpu/actions/runs/43700398954)中已通过。当前分支新增部署与阶段控制测试、真实 router/SSH 集成 job；本地基础测试通过，新增 Linux 集成尚未运行，分支保护设置未核验。

## 目标与执行路径

工作流位于 `.github/workflows/`，构建 router、检查 JUnit 的脚本及检查依赖集中在 `.github/ci/`；根目录不再单列 `ci/`。功能测试统一位于 `tests/`，按受测模块分类；真实跨组件验证放在 `tests/integration/`。组织方式与对照版本见 [测试目录指南](../tests/README.md)。

每个 PR 都应得到一份明确的基础功能结论。第一阶段在 Ubuntu 24.04、Python 3.12 上运行全部 CPU 测试，不按路径跳过文档 PR，也不依赖 GPU、模型下载或私有 Secret。当前使用 `tests/requirements-cpu.txt`，仅用 `--no-deps` 安装本项目 wheel；这不是生产推理环境。

工作流 `.github/workflows/ci.yml` 响应 PR、main push 和手动运行，三个任务独立执行：

- `static`：Ruff、Python 编译、actionlint 和 CI 报告校验器测试。
- `cpu`：构建并安装 wheel，从 Markdown 提取示例到临时目录，检查源码目录外的 CLI/plan，执行 `tests/` 中除 `integration/` 外的全部基础用例。
- `cpu-integration`：从固定源码构建真实 vllm-router，在 Linux 上启动独立 sshd，执行 `tests/integration/` 的真实进程、HTTP、SSH 和 controller 测试。

`ci-gate` 等待三个任务，只接受全部为 `success`。失败、取消或任务被跳过都不能得到成功汇总结论。集成构建最多使用 90 分钟，测试步骤另限 15 分钟；当前没有缓存 router 二进制或 Rust 构建结果，首次托管运行后再依据实测决定是否优化。

两套 CPU 测试都必须收集到用例，且没有失败、错误或跳过。JUnit 校验不会替代 pytest 的退出码。报告和子进程日志随本次运行上传，保留 7 天；安装或收集阶段失败可能没有完整报告，但仍必须使任务失败。只缓存 pip 下载，不缓存测试结果。一般 CPU 用例按职责放进 `tests/server/`、`tests/engine/`、`tests/deployment/`、`tests/kv/`，依赖 Linux/router/sshd 的用例放进 `tests/integration/`；缺失先决条件直接失败，不能用 skip 绕过。

当前范围还覆盖部署配置、进程所有权、运行包来源、失败回滚、角色 HTTP、KV 描述符、lease 回收与 CPU 阶段控制。模型计算和 KV 传输使用明确的测试实现，不证明 Qwen 数值正确、CUDA Graph、TP/NCCL、NIXL、跨机真实 P/D 或吞吐收益。详细边界见 [CPU 部署验证记录](testing.md)。

## 参考与取舍

初版对照基线：Nano-XPU `53e77bb993291ccc28d8e3c943ab618687aa32e2`。当前扩展以已合入的 `2d61a9c8aa6eaf7eb6a928b1331f67418b5bd79b` 加 `feat/rfc-cpu-deployment` 工作区修改为对象；下面 vLLM/SGLang 的参考文件与固定提交不变。

- vLLM `730194d2362716c47648d06e172c91bcdace3b32`：[pre-commit.yml](https://github.com/vllm-project/vllm/blob/730194d2362716c47648d06e172c91bcdace3b32/.github/workflows/pre-commit.yml) 将便宜的静态检查独立运行，[engine.yaml](https://github.com/vllm-project/vllm/blob/730194d2362716c47648d06e172c91bcdace3b32/.buildkite/test_areas/engine.yaml) 声明领域测试与依赖。本仓先保留独立静态任务、可本地执行的原生测试命令，不引入 Buildkite 或路径选测。
- SGLang `40723a52ded9229e221a437254f34bc8a3327251`：[pr-test.yml 的 pr-test-finish](https://github.com/sgl-project/sglang/blob/40723a52ded9229e221a437254f34bc8a3327251/.github/workflows/pr-test.yml) 在任务结束后集中检查结果。本仓借鉴固定汇总任务，但所有 CPU 基础任务必跑，因此采用更严格的“全部 success”，不允许 skipped。

不复制上游的设备矩阵、机器人授权、分片和夜间任务。当前基础套件规模适合完整运行；真实 router 编译单独占用一个 job，避免阻塞便宜检查的反馈。Python 直接依赖版本固定，传递依赖尚未完整锁定；CI 输出实际安装列表。router 固定源码提交、Cargo.lock 和上游使用的 Rust 1.95.0，显式启用 `serde_json/float_roundtrip`，使 KV 元数据的浮点期限经 JSON 转发后保持原值；构建收据记录 feature 与二进制 SHA256，不以相同包版本号代替源码证明。首次托管 CI 的失败、修正与待复测项见 [验证记录](testing.md#首次托管-ci-与修正)。

## Agent 反串讲

例如 PR 修改了请求取消逻辑：GitHub 检出该次待测提交，CPU job 安装其 wheel，pytest 发起一次请求并取消等待，测试引擎继续运行；用例确认后台完成前不会放出槽位。pytest 把断言结果写入 JUnit，校验器拒绝空跑、失败或跳过，job 结果再交给 `ci-gate`。报告只属于当前 run/attempt，不从缓存读取旧的通过结果。

为什么有效？它用可重复的请求与状态变化检查基础行为，并把错误沿退出码传到汇总。哪里无效？桩不执行模型或设备操作。代价在哪里？所有 PR 都运行便宜测试，换来较少的选测规则。怎样证伪？注入断言失败、空报告、skip 和取消状态，任何一种还能得到通过就是 CI 缺陷。剩余关键环境差异是 Windows 本地验证与 Ubuntu 托管运行，需要上线后记录真实运行链接。

## 验收设计与结果

| 用例 | 输入与步骤 | 通过标准 | 实际结果 |
|---|---|---|---|
| CPU-01 | 安装 CPU 依赖，构建并安装 wheel；运行除 `tests/integration/` 外的基础测试 | 用例非空、零失败、零错误、零跳过 | 首次托管基础套件 266 通过、2 失败；修正后的 Windows / CPython 3.12.13 基础套件 272 通过（72.44s），Linux 修正结果待复测；详见 testing.md |
| CPU-02 | 在源码目录外执行安装后的 CLI help | 入口可运行，不导入 CUDA 核心依赖 | 当前 nano-xpu wheel 的元数据、nanoxpu 入口与旧包缺失检查通过；serve/_worker help 与两类 serve plan 和 runtime-info 模块入口通过，确认导入安装目录且无四个 CUDA 核心依赖；首轮新 CI 的 Linux console 与源码外 wheel smoke 已通过，后续测试失败仍阻止 job 通过 |
| CPU-03 | Ruff、compileall、actionlint | Python 与工作流的基础错误检查通过 | Ruff 0.14.4、compileall、actionlint 1.7.12 通过；本地无 ShellCheck/Pyflakes，actionlint 的对应外部检查未运行 |
| CPU-04 | 校验成功、失败、错误、空、skip、缺失和损坏 JUnit | 仅非空且全成功的报告通过 | 6 个 unittest 通过，包含 failure/error/skipped 子用例及部分 skip；最新 272 项报告接受；首轮托管失败报告被正确拒绝 |
| CPU-05 | 将汇总的三个依赖分别设为 success/failure/cancelled/skipped | 仅三个 success 返回 0 | 从当前 YAML 提取实际 Bash，Git Bash 执行全部 64 种组合通过；不代表已验证 GitHub 调度 |
| CPU-06 | GitHub 上 PR/main push；检查报告下载和 required check | 均有结论，结果对应当前提交；配置保护后失败阻止合入 | #2/#3 检查通过并合入；#4–#6 的失败使 ci-gate 失败，尚未合入；不能据此声称仓库已配置强制保护 |
| CPU-07 | 固定源码 router，单/双 worker、P/D 预算和元数据、失败/断连 | 真实转发正确、无意外重放；内部 P 输出不泄漏 | 首轮已运行；router 元数据出现浮点末位变化，SSH 被 StrictModes 拒绝；对应修正待 Linux 复测，详见 testing.md |
| CPU-08 | 临时 sshd、严格 host key、controller 与恢复 CLI | 未信任 host 拒绝；重复 up 不增进程；启动失败回滚；down 无持有进程残留 | 首轮已运行；router 元数据出现浮点末位变化，SSH 被 StrictModes 拒绝；对应修正待 Linux 复测，详见 testing.md |
| CPU-09 | 真实 router → HTTP P/D StageEngine → CPU 数组复制 | 输出/usage 与完整 CPU Prefill 基线一致，D 只恢复最后 prompt token，结束后无占用块 | 首轮真实 router 3 项中 1 通过、2 项 descriptor 不兼容；文档分支同轮 3 项通过，浮点往返的偶现问题仍需修正后复测；KV 为进程内数组，不是跨进程传输 |

本地干净安装首次复测遗漏创建 `artifacts/`，出现 5 passed / 38 errors；原因是 `--basetemp` 的父目录不存在。工作流已有 `mkdir -p artifacts`，已修正文档的复现命令并重跑得到上表结果。工作区证据保存于 `../../output/ci-workflow-audit-20261115/`（不随仓库发布）：`cpu-tests.xml`、初次失败报告、`ci-validation.json`。3 份 Issue 模板元数据、PR 模板、36 个本地文档链接和 `git diff --check` 均已检查。耗时只是单次本地正确性运行记录，不是性能结论。

## 上线与维护

标准 Linux / Python 3.12 环境中，在仓库根目录复现主要命令：

```bash
python3.12 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install -r .github/ci/requirements.txt -r tests/requirements-cpu.txt
python -m ruff check .
python -m compileall -q nanoxpu tests .github/ci
python -m unittest discover -s .github/ci -p 'test_*.py' -v
python -m build --wheel --no-isolation
python -m pip install --no-deps dist/*.whl
(cd /tmp && nano-xpu serve --help && python -m nanoxpu serve --help)
python tests/helpers/examples.py artifacts/documented-examples
nano-xpu serve plan -f artifacts/documented-examples/deployment/cluster-colocated.yaml
nano-xpu serve plan -f artifacts/documented-examples/deployment/cluster-pd.yaml
mkdir -p artifacts
python -m pytest -q tests --ignore=tests/integration --strict-markers --junitxml=artifacts/cpu-tests.xml --basetemp=artifacts/pytest
python .github/ci/check_junit.py artifacts/cpu-tests.xml
```

工作流语法另用 actionlint 1.7.12 检查，工作流内包含下载与 SHA256 校验命令。本机 Windows 的解释器限制和可用命令见 [CPU 本地验证](testing.md)。Ruff 当前只启用语法和确定性错误规则，不强制整仓格式化。新建 job 时必须同步 `ci-gate.needs` 和成功判定，避免遗漏。

Linux 集成需已安装 `rustup`，以及工作流列出的 Rust 原生构建依赖和 `openssh-server`。安装 Python 依赖后执行：

```bash
python -m pip install --no-deps --no-build-isolation -e .
build_root=$(mktemp -d)
export NANO_XPU_ROUTER_BIN="$build_root/bin/vllm-router"
python .github/ci/build_router.py --source "$build_root/source" --output "$NANO_XPU_ROUTER_BIN"
mkdir -p artifacts
python -m pytest -q tests/integration --strict-markers --junitxml=artifacts/cpu-integration.xml --basetemp=artifacts/integration-pytest
python .github/ci/check_junit.py artifacts/cpu-integration.xml
```

SSH 测试通过 `sudo -n` 启动仅监听 localhost 随机高位端口的独立 sshd，不修改系统 SSH 配置。测试临时生成 host/client key，以专用 known_hosts 验证主机身份；关闭时只停止核对过身份的测试 daemon。密钥和运行状态保存在当前用户 home 下的独立私有临时目录，使授权文件的祖先路径满足 `StrictModes yes`，不用可被其他用户写入的 `/tmp`。artifact 只上传列出的日志、JUnit、安装清单、router 构建来源和 help；不上传临时目录。controller 测试使用独立、已提交的临时源码副本检查实际 helper/worker 的导入来源，生产 CLI 不增加 CPU 模拟开关。

本地可审阅的工作流与模板准备完成后，按仓库发布规范校核并推送。在 GitHub 完成一次成功运行后，把 job 名 `ci-gate` 配为 main 的 required status check，并确认来源为 GitHub Actions；YAML 自身不会设置分支保护。Issue 模板进入默认分支后才出现在新建选择页。当前不配置自动合入、远程发布、标签或人员指派。

PR 只取消同一 PR 的旧运行；push/手动运行使用各自 run ID。job 有超时限制。任务使用 GitHub 托管 runner 和只读仓库权限，checkout 不持久化凭据。Actions 固定到完整提交；更改固定版本时重新验证。若以后启用 merge queue，需补充 `merge_group` 事件及其验收；本版没有宣称支持队列合入。
