# K100AI 复现指南

[返回首页](../../README.md) · [文档索引](README.md) · [来源与校验](PROVENANCE.md)

本页说明如何从 v0.3.1 发布包验证二进制、准备模型输入、启动服务并重复
质量与 API 验收。固定构建环境为 **K100AI/gfx928、Linux x86_64、
DTK 2604**。

## 发布输入

| 输入 | 位置 |
| --- | --- |
| 运行镜像 | [安装指南](INSTALL.md#安装) |
| DeepSeek 三套配置 | `configs/k100ai/*-prod.json` |
| Qwen v0.3.1 生产模板 | `configs/k100ai/qwen-iq3s-r112.template.json` |
| Qwen 旧引擎模板 | `configs/k100ai/qwen-server.template.json` |
| 模型与资产来源 | [已固定的 revision 和 SHA-256](PROVENANCE.md) |
| Qwen v0.3.1 完整方法 | [Qwen v0.3.1 指南](QWEN31.md) |
| DeepSeek 最终参数 | [DeepSeek-V4.1 指南](DEEPSEEK41.md#启用参数) |

模型权重、tokenizer 和 MTP runtime 不进入发布归档。expert profile 与
IQ3_S draft vocabulary 随发布包提供。下载后按 `BUILD_INFO.json` 和
`configs/k100ai/release-inputs.json` 核对 revision、文件大小与 SHA-256。

## 校验发布包

```bash
sha256sum -c SHA256SUMS
```

校验四个生产二进制：

```bash
sha256sum \
  bin/strata-qwen-iq3s-r112 \
  bin/strata-0.1.47 \
  bin/strata-0.1.55 \
  bin/strata-dsv41
```

预期值：

```text
102b4b412068a68588883c13a28103600d8b078785da9dcbbda34c440e9a5074  bin/strata-qwen-iq3s-r112
bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9  bin/strata-0.1.47
25d4ea83b883443d0a8bc68686c72f613cd0923a2897c2f7e9063939025e904e  bin/strata-0.1.55
4fd967c1fa6769e60298357d55683c18e5522f6689441b12fe5a97489b3a35d6  bin/strata-dsv41
```

DeepSeek 二进制的生产来源、源码对账和 clean build 记录见 [PROVENANCE.md](PROVENANCE.md)。发布机没有重新编译出相同 SHA-256，复现验证以已记录的生产构建和逐文件源代码对账为准。

## 启动与 API 验收

先准备 Qwen IQ3_S 或 DeepSeek 模型资产，再按 [安装指南](INSTALL.md)
启动服务。Qwen 资产恢复脚本为：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
  bash scripts/k100ai/fetch-qwen-iq3s.sh \
  /data/k100ai-assets/qwen3.8-flash-next-iq3-s
```

Qwen 下载脚本会在获取大模型分片前准备固定提交 `3cf0325` 的 `gguf-py`。
发布包应自带该目录；源码检出缺失时由脚本下载并校验固定 ZIP，打包器也
强制要求 `--ggml-dir`，防止生成缺少该依赖的包。

每次 GPU 工作前执行：

```bash
bash scripts/k100ai/preflight.sh
```

API 验收命令见[运行指南](RUNTIME.md#api-验收)。固定四项测试覆盖健康检查、模型列表、OpenAI 消息和 Anthropic 消息。

## 性能与质量

- Qwen v0.3.1 IQ3_S 使用 input exact、output exact、retrieval、
  Native / MTP 输出哈希和确定性 draft 验证。
- 旧 Qwen 快速档使用 top-1、余弦相似度和 greedy 前缀验收，不要求逐位
  一致。
- DeepSeek 三套生产配置要求 PPL 相对基线不超过 `+2%`，并执行固定 API 测试。
- 原始时间、配置和限制分别记录在 [DEEPSEEK41.md](DEEPSEEK41.md) 与 [VALIDATION.md](VALIDATION.md)。

## 对照运行

比较两个候选时，保持硬件、模型 pack、输入、采样参数、并发数和输出上限一致，并交替运行候选。至少保存：

- 完整启动参数和配置快照
- 实际 prompt / output token 数
- 首 token 时间和输出 tok/s
- MTP / DSpark offered / accepted 数
- 输出哈希或质量检查结果
- 每次运行的原始 `/metrics` 和引擎日志摘要

单次有利样本只作为观察，不写成稳定加速结论。
