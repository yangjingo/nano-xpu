# 测试组织

测试按源码职责组织：`nanoxpu/server/` 对应 `tests/server/`，`nanoxpu/engine/` 对应 `tests/engine/`。公共测试引擎、HTTP 请求和客户端夹具放在 `tests/helpers/`，不从另一个测试文件导入。

运行 `python -m pytest -q tests --strict-markers`。本次整理保留已有 43 项 CPU 用例，生产推理逻辑不变；GPU 数值与设备性能仍需独立验证。文档中的 CPU 启动脚本同步新的测试模块路径。
