# 参与 Nano-XPU

先阅读 [README](README.md)、相关 [RFC](docs/rfcs/README.md)、目标模块和现有测试，说明本次希望改变的行为与验收条件。

- 缺陷用 [Bug 模板](.github/ISSUE_TEMPLATE/bug_report.md)，提供最小复现与环境。
- 新能力或可独立完成的任务用 [Feature 模板](.github/ISSUE_TEMPLATE/feature_request.md)，说明范围、依赖与完成标准。
- 接口、架构或重要行为调整先用 [RFC 模板](.github/ISSUE_TEMPLATE/rfc.md) 讨论机制与取舍；定稿保存在 `docs/rfcs/`。方案讨论通过不代表实现或测试已经完成。
- PR 使用 [PR 模板](.github/pull_request_template.md)，一个 PR 聚焦一个问题，列出实际验证和未完成项；只在完整解决 Issue 时使用 `Closes`。

基础检查按 [CPU CI 指南](docs/ci.md) 执行。新增 CPU 回归测试放进 `tests/`，CI 自动运行整个目录；测试应观察行为，而非重复实现。设备测试需要另行记录模型、设备、版本与输入，不能用 CPU 桩结果代替。

改动后同步相关文档。工作流的 `ci-gate` 是基础验证结论；是否允许合入仍由仓库规则与评审决定，模板本身不会创建规则、标签或自动指派人员。
