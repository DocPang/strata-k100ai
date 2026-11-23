# K100AI 运行指南

[返回首页](../../README.md) · [文档索引](README.md) · [安装指南](INSTALL.md) · [配置目录](../../configs/k100ai/README.md)

适用版本：**v0.3.2**。同一时间只运行一套模型服务。

## 支持的模型

| 模型 | 服务名 | 最少设备 | 二进制 |
| --- | --- | ---: | --- |
| Qwen3.8-Flash-Next IQ3_S 生产档 | `qwen3.8-flash-next` | 1× K100AI 64 GiB | `bin/strata-qwen-iq3s-r112` |
| Qwen3.8-Flash-Next IQ3_XXS 快速兼容档 | `qwen3.8-flash-next` | 1× K100AI 64 GiB | `bin/strata-0.1.47` |
| Qwen3.8-Flash-Next 0.1.55 对照档 | `qwen3.8-flash-next` | 1× K100AI 64 GiB | `bin/strata-0.1.55` |
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

v0.3.2 的 `qwen` 入口默认启动 IQ3_S 生产引擎。它使用完整路由 top-10、
QSA 精确 top-k 和目标模型验证的 MTP，关闭 prompt-lookup 后缀投机，
不启用 Top-k 截断或其他近似 Prefill 算法：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

默认资产目录为
`$K100AI_ASSETS_DIR/qwen3.8-flash-next-iq3-s`。`K100AI_QWEN_MTP_VOCAB`
默认指向发布包内
`data/k100ai-qwen-iq3s-draft-vocab-zh60k.bin`。

生产模板参数：

```text
--expert-cache 24576
--vram-reserve-mib 2048
--prefill 20480
--spec 3
--spec-min-p 0.5
--max-context 262144
--kv int8
--kv-resident 65536
--mtp-window 4096
--mtp-q4 all
--no-publish-kernel
--suffix-draft 0
STRATA_PREFILL_TOPK=10
```

`qwen-iq3s` 是同一引擎的显式别名。

安装和资产准备见[安装指南](INSTALL.md#qwen38-flash-next)，完整成绩见
[Qwen v0.3.1 指南](QWEN31.md)。

### 旧 IQ3_XXS 快速档

旧快速档使用 `qwen-iq3xxs` 入口，资产目录为
`$K100AI_ASSETS_DIR/qwen3.8-flash-next-iq3-xxs`：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen-iq3xxs
```

它使用 IQ3_XXS、INT8 KV、24,576 expert slots、`--spec 6` 和 MTP。
该路径允许数值漂移，不保证逐位一致，不属于 v0.3.1 IQ3_S 精确路径。

### 旧 0.1.55 对照档

显式切换引擎并继续使用 IQ3_XXS 资产：

```bash
K100AI_QWEN_ENGINE=bin/strata-0.1.55 \
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen-iq3xxs
```

可单独覆盖以下资产路径：

```text
K100AI_QWEN_PACK
K100AI_QWEN_NATIVE
K100AI_QWEN_PLE
K100AI_QWEN_TOKENIZER
K100AI_QWEN_MTP_RT
K100AI_QWEN_MTP_VOCAB
```

旧快速档与 0.1.55 的历史精度和对照数据见
[验证记录的历史章节](VALIDATION.md#历史-qwen-023-快速档与对照档)。

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
- Qwen IQ3_S 默认上下文 262,144 tokens，65,536 个 KV 位置驻留显存。
  更长请求按需从主机内存读取 KV。发布验收覆盖 261,632 输入与 256 输出。
- 完整 1M context 尚未完成质量评测。
