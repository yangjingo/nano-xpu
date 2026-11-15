# RFC 写作与使用说明

RFC（Request for Comments，征求意见稿）用于在实现前讨论问题、方案和验收标准。Nano-XPU 的模板固定在 [`.github/ISSUE_TEMPLATE/rfc.md`](../../.github/ISSUE_TEMPLATE/rfc.md)，正文采用 Markdown，便于 GitHub 讨论和仓库内阅读。

## 如何使用

1. 先阅读 [Roadmap](../roadmap.md) 和已有 RFC。新增能力、架构、接口或性能机制变化应先说明方案；错字、链接和行为明确的小修复可复用已有设计，补充原因与验证记录。
2. 在 GitHub 创建 Issue 时选择“RFC 技术方案”。本地编写时复制同一模板，删除开头的 GitHub YAML frontmatter，替换占位，使用 `RFC-NNNN：标题`。Issue 标题使用 `[RFC] 标题`，不把 Issue 编号当作 RFC 编号。
3. 需要保存到仓库时，使用 `docs/rfcs/NNNN-short-title.md`，接续现有四位编号并检查冲突；在同一文件中更新当前方案。按实际需要保留子节，不适用项说明原因，不为了填满模板新增设计。
4. 写清基线、调用流程、接口、失败处理和验收方法，主动完成反串讲。影响实现的关键疑问解决后再实施；讨论结论写回正文，保留真实评审链接。方案已接受、代码已实现、验收已通过分别记录。
5. 实施后将实际结果和证据回填用例与结果表。失败、未执行、环境阻塞同样保留，写明原因和下一步。兼容性或方案有变化时同步正文。

已有 RFC 无需仅为套用模板重写。后续实质更新时补齐适用的缺项即可；本模板不要求额外保存 HTML。

模板使用 Markdown Issue Template，支持整篇编辑和本地复用；它没有表单级必填校验或自动审批。按照 [GitHub 官方说明](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/about-issue-and-pull-request-templates)，模板进入远程仓库的默认分支后，才会出现在新建 Issue 的选择页。本地文件存在不表示线上已启用。

## 写到什么程度

- **先说具体行为。** 写清谁做什么、何时执行、满足什么条件才能继续。以一个真实请求串起调用、数据和状态流，避免只列模块与术语。
- **区分证据与方案。** 现状附代码路径、符号和完整 commit；外部主张附原始链接；推断与拟新增能力明确标注。跨仓对比记录双方版本及相关工作区修改，不能把设计当实现。
- **接口能被审阅。** 输入、输出、调用方、时机、前后置条件要明确。异步与设备操作要说明状态、数据和资源的归属，以及失败后能否安全继续。
- **验收可以执行。** 提前写好输入、步骤、预期和判定标准，只把实际结果留待回填。CPU 桩验证不能代替真实模型、CUDA/Ascend、多卡或跨机验证。
- **性能结论有边界。** 记录同条件基线、环境、负载、预热、重复次数与统计口径，并同时检查输出正确性。没有设备就注明未验证，不写预计收益为实测结果。
- **复杂工作可拆分。** 每项任务有独立产物、依赖和完成条件，明确哪些问题需要维护者决策；不编造认领、排期或远程编号。

## 借鉴来源与本仓库适配

以下于 2026-11-15 核查，链接固定到完整提交。只借鉴与 RFC 写作有关的结构，不引入上游的角色体系、机器人命令或审批门槛。

| 来源与检查路径 | 借鉴内容 | Nano-XPU 的适配 |
|---|---|---|
| [vLLM RFC Issue 模板](https://github.com/vllm-project/vllm/blob/730194d2362716c47648d06e172c91bcdace3b32/.github/ISSUE_TEMPLATE/750-RFC.yml) | 以动机和拟议改动为核心，提供反馈周期和反馈对象 | 保留简短摘要与反馈问题，周期按需填写，不设固定等待期限 |
| [vLLM Collaboration Policy：Adding New Major Features](https://github.com/vllm-project/vllm/blob/730194d2362716c47648d06e172c91bcdace3b32/docs/governance/collaboration.md#adding-new-major-features) | 重大功能先讨论 RFC，说明问题、备选方案和拟议改变 | 显式比较最简单可行方案，补充依赖与待决事项 |
| [SGLang Feature Request](https://github.com/sgl-project/sglang/blob/40723a52ded9229e221a437254f34bc8a3327251/.github/ISSUE_TEMPLATE/2-feature-request.yml) | 简洁说明动机并提供相关资源 | 现状就近附代码和原始来源；不将该模板称为独立 RFC 模板 |
| [SGLang PR Template](https://github.com/sgl-project/sglang/blob/40723a52ded9229e221a437254f34bc8a3327251/.github/pull_request_template.md) | 分开呈现改动、正确性测试与速度测试及 profiling | RFC 阶段先设计适用验证，实施后回填结果；不把 PR 结果要求写成草案已通过 |

版本：vLLM `730194d2362716c47648d06e172c91bcdace3b32`；SGLang `40723a52ded9229e221a437254f34bc8a3327251`。这里只检查远程模板与协作文档，未使用上游工作区修改，也不涉及运行时实现对比。

本仓库基线为 `53e77bb993291ccc28d8e3c943ab618687aa32e2`。核查时存在未提交的 RFC HTML 清理、RFC-0001 链接修正、Roadmap 补充、RFC-0002 与配置样例；这些内容作为当前文档上下文，不代表已合入或完成设备验收。接口与资源语义、反串讲、任务拆分、验证占位及设备证据边界是结合本项目需要增加的内容，不声称是上游模板原有要求。

## 现有 RFC 与验证资料

- [RFC-0001：nano-xpu serve 服务化与切分支持](0001-serve-qwen-colocated.md)
- [CPU 验证指南](README-cpu.md)
- [CPU 验证记录](cpu-validation.md)
