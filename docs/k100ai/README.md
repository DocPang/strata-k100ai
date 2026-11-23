# Strata for K100AI 文档

[项目首页](../../README.md) · [下载 v0.3.2](https://github.com/DocPang/strata-k100ai/releases/tag/v0.3.2)

## 安装与运行

- [安装指南](INSTALL.md) — 校验发布包、准备模型文件、启动服务。
- [配置指南](../../configs/k100ai/README.md) — 按卡数选配置，查看对应 JSON。
- [运行指南](RUNTIME.md) — 切换引擎、设置资产路径、调用和验收 API。

## 模型与实测

- [Qwen3.8-Flash-Next IQ3_S](QWEN31.md) — v0.3.1 单卡配置、Prefill 和 Decode 实测、输出校验。
- [DeepSeek-V4.1-Flash](DEEPSEEK41.md) — v0.3.0 双卡和四卡配置、首 token 时间、Decode 和 PPL。
- [验证记录](VALIDATION.md) — 精度、性能、长输入、API 和发布包测试。

## 构建与复现

- [复现指南](REPRODUCIBILITY.md) — 构建步骤、测试输入和验收门槛。
- [来源与校验](PROVENANCE.md) — 二进制、源码和模型资产的来源与哈希。
- [优化说明](OPTIMIZATION.md) — gfx928 内核、显存、MMAC、MTP 和 DSpark。

## 版本

[版本记录](release-notes/README.md) · [v0.3.2 发布说明](release-notes/v0.3.2.md) · [v0.2.3 归档](archive/v0.2.3/README.md)
