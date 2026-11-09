# v0.3.0：DeepSeek-V4.1 与 Qwen3.8-Flash-Next

v0.3.0 冻结 DeepSeek-V4.1-Flash 的 R001-R179 优化结果，同时保留项目中已经
验证的 Qwen3.8-Flash-Next 推理能力。

## 包含的模型

### Qwen3.8-Flash-Next

- 默认 `0.1.47` 快速引擎
- `0.1.55` 对照引擎
- OpenAI / Anthropic 兼容 API
- IQ3_XXS、INT8 KV、MTP 和单卡 K100AI 部署

### DeepSeek-V4.1-Flash

- EXL3 四卡 TP1
- EXL3 四卡 TP4
- GGUF 双卡 TP2
- Native / DSpark Decode
- 1M context 配置

## 实测成绩

Qwen3.8-Flash-Next 沿用 v0.2.3 已验证的同一对引擎：

| 输入 | 0.1.47 | 0.1.55 |
| ---: | ---: | ---: |
| 8,190 tokens | 4.375 s，1,846.2 tok/s | 7.860 s，1,033.6 tok/s |
| 40,057 tokens | 19.066 s，2,086.1 tok/s | 34.361 s，1,161.3 tok/s |

0.1.47 的 Decode 为 `79.37 tok/s`，MTP 接受 `56 / 57`。

DeepSeek-V4.1-Flash：

| 配置 | 显卡数量 | 4K TTFT | 32K TTFT | Decode |
| --- | ---: | ---: | ---: | ---: |
| EXL3 TP1 | 4 | 2.98 s | 17.95 s | Native 18.7 tok/s |
| EXL3 TP4 | 4 | 4.04 s | 29.22 s | DSpark 43.9 tok/s |
| GGUF TP2 | 2 | 8.11 s | 52.42 s | DSpark 42.05 tok/s，中位数 |

TP1、TP4 的 32K 改善为 `3.63%`、`3.20%`，没有达到 `<14.5 s` / `<24 s`
的目标。三套配置的 PPL 变化分别为 `-0.262%`、`+0.161%`、`+0.060%`，
均通过 `+2%` 门限，固定 API 均为 4/4。

## 下载和安装

推荐下载完整包：

```bash
BASE=https://github.com/DocPang/strata-k100ai/releases/download/v0.3.0
curl -fLO "$BASE/strata-k100ai-v0.3.0.tar.gz"
curl -fLO "$BASE/strata-k100ai-v0.3.0.tar.gz.sha256"
sha256sum -c strata-k100ai-v0.3.0.tar.gz.sha256
tar -xzf strata-k100ai-v0.3.0.tar.gz
cd strata-k100ai-v0.3.0
sha256sum -c SHA256SUMS
```

安装 Python 运行依赖并检查 HCU0-3：

```bash
bash scripts/k100ai/setup-runtime.sh
bash scripts/k100ai/preflight.sh
```

启动 Qwen3.8-Flash-Next：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

启动 DeepSeek EXL3 TP4，先修改 JSON 中的模型路径：

```bash
bash scripts/k100ai/start-dsv41.sh configs/exl3-tp4-prod.json
```

## Release 文件

| 文件 | 用途 |
| --- | --- |
| `strata-k100ai-v0.3.0.tar.gz` | 完整项目、运行时和三套生产二进制 |
| `strata-0.1.47` | Qwen3.8-Flash-Next 默认引擎 |
| `strata-0.1.55` | Qwen3.8-Flash-Next 对照引擎 |
| `strata-dsv41` | DeepSeek-V4.1 最终引擎 |
| 三套 `*-prod.json` | DeepSeek 生产配置 |
| `SHA256SUMS` | 完整 tag 校验 |
| `ASSETS-SHA256SUMS` | Release 独立资产校验 |

模型权重、tokenizer 和 MTP runtime 需要另行准备。
