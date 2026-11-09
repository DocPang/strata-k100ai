# Strata for K100AI v0.3.0

这是一个面向 **Hygon K100AI / gfx928** 的本地大模型推理项目。它负责把
模型请求转换成 GPU 运算，并提供 OpenAI / Anthropic 兼容 API，方便现有客户端
直接调用。

当前版本同时保留两条生产模型线：

- **Qwen3.8-Flash-Next**：默认使用 0.1.47 快速引擎，同时保留 0.1.55 对照引擎。
- **DeepSeek-V4.1-Flash**：使用最终冻结的 `strata-dsv41` 引擎，提供三套已验证配置。

Release：
<https://github.com/DocPang/strata-k100ai/releases/tag/v0.3.0>

模型权重、tokenizer 和必要的运行时资源体积很大，不包含在仓库中，需要另行准备。

## 这个项目能做什么

部署后，用户可以启动单实例模型服务，并通过标准接口调用：

```text
POST /v1/chat/completions    OpenAI 兼容
POST /v1/messages            Anthropic 兼容
GET  /v1/models
GET  /health
```

两个服务模型名：

```text
qwen3.8-flash-next
deepseek-v4.1-flash
```

仓库同时包含 GPU 内核、模型加载、量化格式、显存管理、长上下文、API 服务、
配置、测试和发布工具。项目不是模型权重镜像；新机器需要按对应配置准备模型资产。

## 支持哪些部署

### Qwen3.8-Flash-Next

单张 K100AI 64 GiB，IQ3_XXS、INT8 KV、24,576 expert slots 和 MTP。

| 引擎 | 文件 | 用途 |
| --- | --- | --- |
| 0.1.47 | `bin/strata-0.1.47` | 默认快速档，8K/40K 验证 |
| 0.1.55 | `bin/strata-0.1.55` | 对照档 |

v0.2.3 的实测结果继续适用于这两个未变更的引擎：

| 输入 | 0.1.47 | 0.1.55 |
| ---: | ---: | ---: |
| 8,190 tokens | 4.375 s，1,846.2 tok/s | 7.860 s，1,033.6 tok/s |
| 40,057 tokens | 19.066 s，2,086.1 tok/s | 34.361 s，1,161.3 tok/s |

0.1.47 在 8,191-token 输入下 Decode 为 `79.37 tok/s`，MTP 接受
`56 / 57`。快速档允许数值漂移，不保证逐位一致。

### DeepSeek-V4.1-Flash

v0.3.0 固化了 R001-R179 的最终生产结果：

| 配置 | 硬件 | 模型格式 | 4K TTFT | 32K TTFT | Decode |
| --- | ---: | --- | ---: | ---: | ---: |
| `configs/exl3-tp1-prod.json` | 4 卡 | EXL3 2.9 bpw | 2.98 s | 17.95 s | Native 18.7 tok/s |
| `configs/exl3-tp4-prod.json` | 4 卡 | EXL3 2.9 bpw | 4.04 s | 29.22 s | DSpark 43.9 tok/s |
| `configs/gguf-tp2-prod.json` | 2 卡 | GGUF 2.00 bpw | 8.11 s | 52.42 s | DSpark 42.05 tok/s，中位数 |

TTFT 表示从提交问题到收到第一个输出 token 的时间，越低越好。Decode 表示开始
输出之后每秒生成多少 token，越高越好。

R179 的 32K TTFT 改善为 TP1 `3.63%`、TP4 `3.20%`，没有达到外部设定的
`14.5 s` / `24 s` 目标。DSpark 的 Decode 会随输出内容和 draft 命中率波动，
`42.05 tok/s` 是四组样本的中位数。

三套 DeepSeek 配置均通过 `+2%` PPL 门限和固定四项 API 测试：

| 配置 | PPL 基线 | 最终 PPL | 变化 | API |
| --- | ---: | ---: | ---: | ---: |
| EXL3 TP1 | 4.860520 | 4.847801 | `-0.262%` | 4/4 |
| EXL3 TP4 | 1.598806 | 1.601385 | `+0.161%` | 4/4 |
| GGUF TP2 | 1.741926 | 1.742965 | `+0.060%` | 4/4 |

## 选择哪一项

| 需求 | 选择 |
| --- | --- |
| 单卡运行 Qwen3.8-Flash-Next，优先 prompt 吞吐 | `scripts/k100ai/serve-k100ai.sh qwen` |
| 使用 Qwen 当前 0.1.55 作对照 | 设置 `K100AI_QWEN_ENGINE=bin/strata-0.1.55` |
| 四卡 DeepSeek，优先 4K 首字延迟 | `configs/exl3-tp1-prod.json` |
| 四卡 DeepSeek，重视持续生成速度 | `configs/exl3-tp4-prod.json` |
| 两卡 DeepSeek，或已有 GGUF 2.00 bpw 资产 | `configs/gguf-tp2-prod.json` |

同一时间只启动一套模型服务。

## 下载什么

Release 页提供：

| 文件 | 说明 |
| --- | --- |
| `strata-k100ai-v0.3.0.tar.gz` | 完整项目与三套生产二进制，推荐下载 |
| `strata-0.1.47` | Qwen Next 默认 0.1.47 引擎 |
| `strata-0.1.55` | Qwen Next 0.1.55 对照引擎 |
| `strata-dsv41` | DeepSeek-V4.1 最终生产引擎 |
| 三套 `*-prod.json` | DeepSeek 可直接检查的生产配置 |
| `SHA256SUMS` / `ASSETS-SHA256SUMS` | 完整包与独立资产的校验值 |

完整部署包下载：

```bash
BASE=https://github.com/DocPang/strata-k100ai/releases/download/v0.3.0
curl -fLO "$BASE/strata-k100ai-v0.3.0.tar.gz"
curl -fLO "$BASE/strata-k100ai-v0.3.0.tar.gz.sha256"
sha256sum -c strata-k100ai-v0.3.0.tar.gz.sha256
tar -xzf strata-k100ai-v0.3.0.tar.gz
cd strata-k100ai-v0.3.0
sha256sum -c SHA256SUMS
```

也可以检出 tag：

```bash
git clone --branch v0.3.0 --depth 1 \
  https://github.com/DocPang/strata-k100ai.git
cd strata-k100ai
sha256sum -c SHA256SUMS
```

## 安装运行依赖

```bash
bash scripts/k100ai/setup-runtime.sh
```

启动 GPU 服务前确认 HCU0-3 空闲：

```bash
bash scripts/k100ai/preflight.sh
```

## 启动 Qwen3.8-Flash-Next

Qwen 资产默认从 `$K100AI_ASSETS_DIR/qwen3.8-flash-next-iq3-xxs` 读取。
如果资产位于其他目录，通过以下变量覆盖：

```text
K100AI_QWEN_PACK
K100AI_QWEN_NATIVE
K100AI_QWEN_PLE
K100AI_QWEN_TOKENIZER
K100AI_QWEN_MTP_RT
```

启动默认 0.1.47 引擎：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

切换到 0.1.55：

```bash
K100AI_QWEN_ENGINE=bin/strata-0.1.55 \
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

## 启动 DeepSeek-V4.1-Flash

选择一套配置，并检查其中的 `tokenizer`、`--pack`，以及 GGUF TP2 的
`--expert-profile` 是否指向本机模型资产：

```bash
bash scripts/k100ai/start-dsv41.sh configs/exl3-tp4-prod.json
```

默认地址：

```text
http://127.0.0.1:18097
```

## 发第一条请求

Qwen：

```bash
curl --fail http://127.0.0.1:18096/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

DeepSeek：

```bash
curl --fail http://127.0.0.1:18097/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

## 二进制

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `bin/strata-0.1.47` | 45,674,152 B | `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9` |
| `bin/strata-0.1.55` | 49,496,856 B | `25d4ea83b883443d0a8bc68686c72f613cd0923a2897c2f7e9063939025e904e` |
| `bin/strata-dsv41` | 23,941,864 B | `4fd967c1fa6769e60298357d55683c18e5522f6689441b12fe5a97489b3a35d6` |

## 文档

- [v0.3.0 安装与启动](docs/k100ai/INSTALL_V0.3.md)
- [v0.3.0 来源与校验](docs/k100ai/PROVENANCE_V0.3.md)
- [DeepSeek R001-R179 最终优化报告](docs/k100ai/DSV41_OPTIMIZATION_FINAL.md)
- [Qwen 与 DeepSeek 运行参数](docs/k100ai/README.md)
- [配置说明](configs/README.md)

## 已知限制

- Qwen 的运行数据来自 v0.2.3 已验证的同一对引擎；v0.3.0 没有重新改变 Qwen
  内核或性能口径。
- DeepSeek 只验证 EXL3 TP1、EXL3 TP4 和 GGUF TP2 三套拓扑。
- TP1、TP4 的 32K TTFT 没有达到 `<14.5 s` / `<24 s`。
- DSpark Decode 波动明显，不能把中位数当作固定收益。
- 完整 1M context 尚未完成质量评测。
- 发布机没有重新编译出与 `strata-dsv41` 相同 SHA-256 的二进制；来源以生产
  clean rebuild、逐文件源码对账和 postinstall 记录为准。
