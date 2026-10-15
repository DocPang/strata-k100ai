# 贡献与结果报告

提交修改前运行 `./scripts/check.sh`（需要维护者环境的 Python 3）；脚本执行 Shell 语法、参数与安装契约检查，并在有 Compose 插件时检查可选配置。没有本地 Release 运行包时，离线成功路径测试会跳过；发布维护者必须用实际包执行并在目标环境构建验证。

性能报告记录设备、驱动/DTK、引擎 SHA、权重与 pack 哈希、输入文本/长度、冷暖缓存、MTP 接受率、prefill/TTFT/decode、显存和并发。相同设备、相同负载交错 A/B/A；不把 profiler、组件或不同接受率数字当作端到端收益。

新模型与 TP2 按 docs/ARCHITECTURE.md 新增独立配置和验收记录。不要上传 API key、访问令牌、私有服务配置、完整原始对话、模型权重或大日志。
