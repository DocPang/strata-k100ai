# K100AI 生产配置

[返回首页](../../README.md) · [安装指南](../../docs/k100ai/INSTALL.md) · [运行指南](../../docs/k100ai/RUNTIME.md)

本目录收纳 v0.3.0 的 DeepSeek-V4.1-Flash 生产配置，以及 Qwen3.8-Flash-Next 的运行模板。三套 DeepSeek 配置共用同一个生产二进制：

```text
strata-dsv41
SHA-256 4fd967c1fa6769e60298357d55683c18e5522f6689441b12fe5a97489b3a35d6
```

## 生产配置

| 文件 | 模型格式 | 设备 | 用途 |
| --- | --- | --- | --- |
| `exl3-tp1-prod.json` | EXL3 2.9 bpw | 四卡 TP1 | 较低 4K 首 token 延迟；DSpark 关闭 |
| `exl3-tp4-prod.json` | EXL3 2.9 bpw | 四卡 TP4 | DSpark 投机解码 |
| `gguf-tp2-prod.json` | GGUF 2.00 bpw | 双卡 TP2 | DSpark 与自适应 Prefill |

配置中的 `description_zh`、`notes_zh`、`measured_zh`、`quality_zh`、`binary_sha256` 和 `provenance_zh` 是发布元数据。服务端只消费运行参数，不会把未知元数据传给引擎。

## Qwen 配置

`qwen-server.template.json` 是可提交的模板，`qwen-server.json` 是生成后的本机配置。启动器通过 `scripts/k100ai/render-config.py` 渲染模板：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

## 启动前

1. 确认 `exe` 指向发布包内的 `bin/strata-dsv41`，`cwd` 位于发布根目录。
2. 按实际机器修改 `tokenizer`、`log`、`--pack` 和 `--expert-profile`。
3. 确认模型资产和冻结 profile 存在。
4. EXL3 只使用 HCU0–3；GGUF TP2 使用 HCU0–1。
5. 校验二进制：

```bash
printf '%s  %s\n' \
  4fd967c1fa6769e60298357d55683c18e5522f6689441b12fe5a97489b3a35d6 \
  bin/strata-dsv41 | sha256sum -c -
```

## 启动

```bash
bash scripts/k100ai/start-dsv41.sh configs/k100ai/exl3-tp4-prod.json
```

同一时间只启动一套配置。切换前停止当前服务并执行 `bash scripts/k100ai/preflight.sh`。

## 实测摘要

输入速度单位为 tok/s。DeepSeek 的 4K/32K 数字按标称输入长度除以记录的 TTFT 换算，是端到端等效吞吐约数；原始记录缺少精确 prompt token 数和引擎内部 Prefill 用时，不能当作精确引擎吞吐。

| 配置 | 4K 输入速度约数 | 32K 输入速度约数 |
| --- | ---: | ---: |
| `exl3-tp1-prod.json` | 约 1,374 | 约 1,825 |
| `exl3-tp4-prod.json` | 约 1,013 | 约 1,121 |
| `gguf-tp2-prod.json` | 约 505 | 约 625 |

投机解码开启时，EXL3 TP4 的 DSpark 为 **43.9 tok/s**，GGUF TP2 的 DSpark 四组样本中位数为 **42.05 tok/s**。TP1 关闭 DSpark，原生解码记录不与上述数值混列。

原始时间、质量门和限制见 [DeepSeek-V4.1 指南](../../docs/k100ai/DEEPSEEK41.md)。
