# v0.3.0 安装与启动

v0.3.0 同时提供：

- Qwen3.8-Flash-Next 0.1.47 默认引擎
- Qwen3.8-Flash-Next 0.1.55 对照引擎
- DeepSeek-V4.1-Flash `strata-dsv41`

模型权重、tokenizer 和 MTP runtime 不随发布包提供。

## 硬件

| 模型 | 最少设备 |
| --- | --- |
| Qwen3.8-Flash-Next | 1 张 K100AI |
| DeepSeek EXL3 TP1 / TP4 | 4 张 K100AI |
| DeepSeek GGUF TP2 | 2 张 K100AI |

本版只映射 HCU0-3：

```text
/dev/kfd
/dev/dri/renderD128
/dev/dri/renderD129
/dev/dri/renderD130
/dev/dri/renderD131
```

## 安装

```bash
BASE=https://github.com/DocPang/strata-k100ai/releases/download/v0.3.0
curl -fLO "$BASE/strata-k100ai-v0.3.0.tar.gz"
curl -fLO "$BASE/strata-k100ai-v0.3.0.tar.gz.sha256"
sha256sum -c strata-k100ai-v0.3.0.tar.gz.sha256
tar -xzf strata-k100ai-v0.3.0.tar.gz
cd strata-k100ai-v0.3.0
sha256sum -c SHA256SUMS
bash scripts/k100ai/setup-runtime.sh
bash scripts/k100ai/preflight.sh
```

固定镜像：

```text
harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
```

## Qwen3.8-Flash-Next

默认资产目录：

```text
$K100AI_ASSETS_DIR/qwen3.8-flash-next-iq3-xxs
```

启动 0.1.47：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

启动 0.1.55：

```bash
K100AI_QWEN_ENGINE=bin/strata-0.1.55 \
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

可单独覆盖：

```text
K100AI_QWEN_PACK
K100AI_QWEN_NATIVE
K100AI_QWEN_PLE
K100AI_QWEN_TOKENIZER
K100AI_QWEN_MTP_RT
```

详细模型下载和资产准备步骤见历史文档
[INSTALL.md](INSTALL.md#3-手动下载并校验-qwen-主权重)。

## DeepSeek-V4.1-Flash

先修改所选 JSON 中的：

```text
tokenizer
args 里的 --pack
GGUF TP2 的 --expert-profile
```

三套配置：

| 配置 | 硬件 | 格式 |
| --- | --- | --- |
| `configs/exl3-tp1-prod.json` | 4 卡 | EXL3 2.9 bpw |
| `configs/exl3-tp4-prod.json` | 4 卡 | EXL3 2.9 bpw |
| `configs/gguf-tp2-prod.json` | 2 卡 | GGUF 2.00 bpw |

启动：

```bash
bash scripts/k100ai/start-dsv41.sh configs/exl3-tp4-prod.json
```

也可继续使用历史 launcher：

```bash
K100AI_DSV41_PACK=/data/models/dsv41-2.00/maya-ds4-dspark \
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18097 \
  bash scripts/k100ai/serve-k100ai.sh dsv41
```

## API

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

运行固定 API 验收：

```bash
BASE_URL=http://127.0.0.1:18097 \
MODEL=deepseek-v4.1-flash \
OUT=api-v030 \
  bash scripts/k100ai/validate-api.sh
```
