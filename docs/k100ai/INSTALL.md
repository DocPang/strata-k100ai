# K100AI 安装指南

[返回首页](../../README.md) · [文档索引](README.md) · [运行指南](RUNTIME.md)

适用版本：**v0.3.1**。本版同时提供：

- Qwen3.8-Flash-Next IQ3_S 默认生产引擎
- Qwen3.8-Flash-Next IQ3_XXS 0.1.47 旧快速兼容引擎
- Qwen3.8-Flash-Next 0.1.55 旧对照引擎
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
BASE=https://github.com/DocPang/strata-k100ai/releases/download/v0.3.1
curl -fLO "$BASE/strata-k100ai-v0.3.1.tar.gz"
curl -fLO "$BASE/strata-k100ai-v0.3.1.tar.gz.sha256"
sha256sum -c strata-k100ai-v0.3.1.tar.gz.sha256
tar -xzf strata-k100ai-v0.3.1.tar.gz
cd strata-k100ai-v0.3.1
sha256sum -c SHA256SUMS
bash scripts/k100ai/setup-runtime.sh
bash scripts/k100ai/preflight.sh
```

固定镜像：

```text
harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
```

## Qwen3.8-Flash-Next

v0.3.1 默认生产引擎为 IQ3_S。模型、pack 和 MTP runtime 需要准备，合计
约 140 GB 磁盘。建议先确认 `setup-runtime.sh` 已完成，再运行恢复下载
和校验脚本：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
  bash scripts/k100ai/fetch-qwen-iq3s.sh \
  /data/k100ai-assets/qwen3.8-flash-next-iq3-s
```

脚本固定模型 revision `6ac9406afe74db316b11a3865a3382c7bedbcf34`，
检查两个 GGUF、pack 和 MTP runtime 的 SHA-256。转换在固定 DTK 2604
容器中完成。发布包自带 238 KB 的
`data/k100ai-qwen-iq3s-draft-vocab-zh60k.bin`，不需要单独下载。

默认资产结构：

```text
$K100AI_ASSETS_DIR/qwen3.8-flash-next-iq3-s/
  model/IQ3_S/*.gguf
  pack/iq3s/{experts.bin,dense.bin,index.txt,native_experts.txt,tokenizer}
  mtp/rt/{experts.bin,dense.bin,dense.txt,draft_vocab.bin}
```

启动 v0.3.1 默认生产引擎：

```bash
printf '%s  %s\n' \
  102b4b412068a68588883c13a28103600d8b078785da9dcbbda34c440e9a5074 \
  bin/strata-qwen-iq3s-r112 | sha256sum -c -

K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

`qwen-iq3s` 是同一入口的显式别名。可覆盖以下路径：

```text
K100AI_QWEN_PACK
K100AI_QWEN_NATIVE
K100AI_QWEN_PLE
K100AI_QWEN_TOKENIZER
K100AI_QWEN_MTP_RT
K100AI_QWEN_MTP_VOCAB
```

旧 IQ3_XXS 快速档使用单独入口：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen-iq3xxs
```

旧 0.1.55 对照档需要显式指定引擎，并继续使用 IQ3_XXS 资产：

```bash
K100AI_QWEN_ENGINE=bin/strata-0.1.55 \
K100AI_QWEN_PACK=/data/k100ai-assets/qwen3.8-flash-next-iq3-xxs/pack/iq3xxs \
K100AI_QWEN_NATIVE=/data/k100ai-assets/qwen3.8-flash-next-iq3-xxs/model/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
K100AI_QWEN_PLE=/data/k100ai-assets/qwen3.8-flash-next-iq3-xxs/model/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
K100AI_QWEN_TOKENIZER=/data/k100ai-assets/qwen3.8-flash-next-iq3-xxs/pack/iq3xxs/tokenizer \
K100AI_QWEN_MTP_RT=/data/k100ai-assets/qwen3.8-flash-next-iq3-xxs/mtp/rt \
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen-iq3xxs
```

IQ3_S 的版本、完整成绩和资产哈希见 [Qwen v0.3.1 指南](QWEN31.md)。
旧 IQ3_XXS 资产仍可使用
`scripts/k100ai/fetch-qwen-iq3xxs.sh`。v0.2.3 的历史安装步骤保留在
[历史安装说明](archive/v0.2.3/INSTALL.md)。

两个 Qwen 下载脚本会在模型分片下载前检查固定版本的 `gguf-py`。发布包
已经内置该目录；源码检出或旧包缺少它时，脚本会下载并校验 llama.cpp
提交 `3cf0325` 后补齐，避免大文件下载完成后才失败。

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
| `configs/k100ai/exl3-tp1-prod.json` | 4 卡 | EXL3 2.9 bpw |
| `configs/k100ai/exl3-tp4-prod.json` | 4 卡 | EXL3 2.9 bpw |
| `configs/k100ai/gguf-tp2-prod.json` | 2 卡 | GGUF 2.00 bpw |

启动：

```bash
bash scripts/k100ai/start-dsv41.sh configs/k100ai/exl3-tp4-prod.json
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

Qwen v0.3.1 API 验收：

```bash
BASE_URL=http://127.0.0.1:18096 \
MODEL=qwen3.8-flash-next \
EXPECT_DRAFT_METRICS=1 \
OUT=api-qwen-v031 \
  bash scripts/k100ai/validate-api.sh
```

DeepSeek API 验收：

```bash
BASE_URL=http://127.0.0.1:18097 \
MODEL=deepseek-v4.1-flash \
OUT=api-dsv41-v031 \
  bash scripts/k100ai/validate-api.sh
```
