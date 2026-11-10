# K100AI 文档

[返回首页](../../README.md) · [配置目录](../../configs/k100ai/README.md)

本目录面向 Hygon K100AI / gfx928。安装、运行、验证、来源和版本记录分别归档；文件名不带版本号，当前版本以 README 和发行记录为准。

## 从这里开始

| 文档 | 内容 |
| --- | --- |
| [安装指南](INSTALL.md) | 运行环境、模型资产、下载校验和启动 |
| [运行指南](RUNTIME.md) | 引擎切换、模型参数、API 验收和运行边界 |
| [配置目录](../../configs/k100ai/README.md) | 三套 DeepSeek 配置与 Qwen 模板 |

## 模型

| 模型 | 文档 |
| --- | --- |
| DeepSeek-V4.1-Flash | [三套生产配置、性能与 R001–R179](DEEPSEEK41.md) |
| Qwen3.8-Flash-Next | [运行参数与引擎说明](RUNTIME.md) |

## 验证与复现

| 文档 | 内容 |
| --- | --- |
| [验证记录](VALIDATION.md) | Qwen 精度、性能、DeepSeek 长输入、API 和发布包测试 |
| [优化说明](OPTIMIZATION.md) | gfx928 内核、显存、MMAC、MTP 与 DSpark 路径 |
| [复现指南](REPRODUCIBILITY.md) | 固定输入、构建步骤和验收门槛 |
| [来源与校验](PROVENANCE.md) | 生产二进制、源码、模型和运行资产 |

## 版本记录

| 版本 | 发行记录 |
| --- | --- |
| [全部版本记录](release-notes/README.md) | 按版本查看功能和发布资产 |
| v0.3.0 | [DeepSeek-V4.1 与 Qwen3.8-Flash-Next](release-notes/v0.3.0.md) |

## 历史资料

早期文档保留在 [v0.2.3 归档](archive/v0.2.3/README.md)，仅用于追溯当时的手动安装和来源记录。
