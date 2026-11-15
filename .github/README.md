# Issue 与 PR 模板

贡献入口见 [CONTRIBUTING.md](../CONTRIBUTING.md)，本地检查和 required check 配置见 [CPU CI 指南](../docs/ci.md)。

| 用途 | 模板 | 需要说明什么 |
|---|---|---|
| 报告缺陷 | [Bug](ISSUE_TEMPLATE/bug_report.md) | 最小复现、预期与实际行为、环境、日志和修复验收条件 |
| 提出需求或拆分功能任务 | [Feature](ISSUE_TEMPLATE/feature_request.md) | 场景、期望能力、范围、备选方式与完成条件 |
| 讨论技术方案 | [RFC](ISSUE_TEMPLATE/rfc.md) | 机制、接口、状态与失败处理、取舍、实施依赖及验收设计 |
| 提交具体文件改动 | [PR](pull_request_template.md) | 改动前后行为、关联设计、范围、实际验证、风险与剩余工作 |

Issue 用于描述问题、需求与讨论；PR 用于审阅具体文件改动。重大设计先写 RFC，Feature 不必重复整份设计；行为明确的小修复可引用已有方案。RFC 的详细说明见 [RFC 写作指南](../docs/rfcs/README.md)。

四份模板都使用 Markdown。Issue 模板含 GitHub 元数据，PR 模板直接填写正文；按任务裁剪不适用子节并说明原因。它们是填写提示，不提供必填字段校验、自动指派、标签创建或审批门禁。

按照 [GitHub 官方说明](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/about-issue-and-pull-request-templates)，进入远程默认分支后，Issue 模板出现在新建 Issue 选择页，PR 模板用于预填新建 PR 的正文。当前本地准备不代表线上已启用。

## 参考与适配

- [vLLM Bug Report](https://github.com/vllm-project/vllm/blob/730194d2362716c47648d06e172c91bcdace3b32/.github/ISSUE_TEMPLATE/400-bug-report.yml) 和 [SGLang Bug Report](https://github.com/sgl-project/sglang/blob/40723a52ded9229e221a437254f34bc8a3327251/.github/ISSUE_TEMPLATE/1-bug-report.yml)：借鉴环境、最小复现与实际输出的组织。Nano-XPU 按实际依赖填环境，不调用上游专用的环境采集命令。
- [vLLM Feature Request](https://github.com/vllm-project/vllm/blob/730194d2362716c47648d06e172c91bcdace3b32/.github/ISSUE_TEMPLATE/500-feature-request.yml)：借鉴具体场景、功能动机与备选方式。
- [SGLang PR Template](https://github.com/sgl-project/sglang/blob/40723a52ded9229e221a437254f34bc8a3327251/.github/pull_request_template.md)：借鉴动机、改动、正确性测试与性能测试分开呈现；本项目补充验证状态和设备能力边界，不移植上游审批角色与机器人指令。

上述参考固定到与 [RFC 指南](../docs/rfcs/README.md#借鉴来源与本仓库适配) 相同的完整提交，属于模板与协作文档对照，不涉及运行时实现对比。
