# K100AI 运行指南

[返回首页](../../README.md) · [文档索引](README.md) · [安装指南](INSTALL.md) · [配置目录](../../configs/k100ai/README.md)

适用版本：**v0.3.0**。同一时间只运行一套模型服务。

## 支持的模型

| 模型 | 服务名 | 最少设备 | 二进制 |
| --- | --- | ---: | --- |
| Qwen3.8-Flash-Next | `qwen3.8-flash-next` | 1× K100AI 64 GiB | `bin/strata-0.1.47` / `bin/strata-0.1.55` |
| DeepSeek-V4.1-Flash | `deepseek-v4.1-flash` | 2× 或 4× K100AI 64 GiB | `bin/strata-dsv41` |

模型权重、tokenizer、MTP runtime 和 DeepSeek expert profile 不随发布包提供。

## 设备

本版使用 HCU0–3，对应：

```text
/dev/kfd
/dev/dri/renderD128
/dev/dri/renderD129
/dev/dri/renderD130
/dev/dri/renderD131
```

Qwen 使用 HCU0；DeepSeek EXL3 使用 HCU0–3；GGUF TP2 使用 HCU0–1。启动前执行：

```bash
bash scripts/k100ai/preflight.sh
```

## Qwen3.8-Flash-Next

Qwen 默认使用 0.1.47 快速档。默认资产目录：

```text
$K100AI_ASSETS_DIR/qwen3.8-flash-next-iq3-xxs
```

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

切换到 0.1.55 对照档：

```bash
K100AI_QWEN_ENGINE=bin/strata-0.1.55 \
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

可单独覆盖以下资产路径：

```text
K100AI_QWEN_PACK
K100AI_QWEN_NATIVE
K100AI_QWEN_PLE
K100AI_QWEN_TOKENIZER
K100AI_QWEN_MTP_RT
```

默认快速档采用 IQ3_XXS、INT8 KV、24,576 expert slots、`--spec 6` 和 MTP。它启用 gfx928 prompt MMAC、fused expert 和 HC upmix，允许数值漂移，不保证逐位一致。精确结果和对照数据见[验证记录](VALIDATION.md#qwen-精度和快速档验收)。

## DeepSeek-V4.1-Flash

三套配置全部位于 `configs/k100ai/`：

| 配置 | 设备 | 格式 | Decode |
| --- | ---: | --- | --- |
| `exl3-tp1-prod.json` | 4 卡 TP1 | EXL3 2.9 bpw | Native |
| `exl3-tp4-prod.json` | 4 卡 TP4 | EXL3 2.9 bpw | DSpark |
| `gguf-tp2-prod.json` | 2 卡 TP2 | GGUF 2.00 bpw | DSpark |

启动前按机器修改：

```text
tokenizer
args 中的 --pack
GGUF TP2 的 --expert-profile
```

```bash
bash scripts/k100ai/start-dsv41.sh configs/k100ai/exl3-tp4-prod.json
```

DeepSeek 的配置说明、原始计时、PPL 和限制见 [DeepSeek-V4.1 指南](DEEPSEEK41.md)。

## API

服务提供：

```text
POST /v1/chat/completions    OpenAI 兼容
POST /v1/messages            Anthropic 兼容
GET  /v1/models
GET  /health
GET  /metrics
```

Qwen：

```bash
curl --fail http://127.0.0.1:18096/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"你好"}],"max_tokens":32,"temperature":0}'
```

DeepSeek：

```bash
curl --fail http://127.0.0.1:18097/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"你好"}],"max_tokens":32,"temperature":0}'
```

默认监听 `127.0.0.1`，配置不启用 API key。不要直接暴露到不受信任的网络。

## API 验收

Qwen：

```bash
BASE_URL=http://127.0.0.1:18096 \
MODEL=qwen3.8-flash-next \
EXPECT_DRAFT_METRICS=1 \
OUT=api-qwen \
  bash scripts/k100ai/validate-api.sh
```

DeepSeek：

```bash
BASE_URL=http://127.0.0.1:18097 \
MODEL=deepseek-v4.1-flash \
OUT=api-dsv41 \
  bash scripts/k100ai/validate-api.sh
```

## 运行边界

- 只验证 K100AI/gfx928、Linux x86_64 和 DTK 2604。
- 不自动停止其他服务或研究容器；切换模型前先停止当前服务。
- DeepSeek DSpark 的输出速度会随内容、draft 命中率和负载波动。
- 完整 1M context 尚未完成质量评测。
