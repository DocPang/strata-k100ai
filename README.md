<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">在海光 K100AI 上运行强大的 AI。</h3>
<p align="center">面向 gfx928 的本地推理引擎、模型配置与可复现发布。</p>

<p align="center">
  <a href="#模型">模型</a> ·
  <a href="#安装"><b>安装</b></a> ·
  <a href="#性能">性能</a> ·
  <a href="#配置">配置</a> ·
  <a href="#文档">文档</a> ·
  <a href="#帮助">帮助</a>
</p>

<p align="center">
  <a href="https://github.com/DocPang/strata-k100ai/releases/tag/v0.3.1"><img src="https://img.shields.io/badge/version-v0.3.1-0ab887" alt="v0.3.1"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/DocPang/strata-k100ai" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/platform-K100AI%20%2F%20gfx928-5b67ff" alt="K100AI / gfx928">
  <img src="https://img.shields.io/badge/DTK-2604-3d6bff" alt="DTK 2604">
</p>

**免费、开源，在你的 K100AI 集群上运行。** 提供 OpenAI 与 Anthropic 兼容 API，可直接连接现有客户端和编程工具。

K100AI 版本 **v0.3.1** · **Linux x86_64 / gfx928 / DTK 2604** · [发布说明](docs/k100ai/release-notes/v0.3.1.md)

**v0.3.1 新增** Qwen3.8-Flash-Next IQ3_S 精确引擎：32K 原生 Decode 45.7–45.9 tok/s，MTP 68.4–71.7 tok/s，Prefill 2,846–2,869 tok/s。

## 模型

本版提供两条生产模型线。二进制随发布包提供；模型权重、tokenizer 和运行资产需要另行准备。

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeek-V4.1-Flash</b></summary>

DeepSeek-V4.1-Flash 提供三套冻结配置，分别面向四卡低首 token 延迟、四卡投机解码和双卡部署。

| 配置 | 设备 | 模型格式 | Decode |
| --- | ---: | --- | --- |
| [EXL3 TP1](configs/k100ai/exl3-tp1-prod.json) | 4× K100AI 64 GiB | EXL3 2.9 bpw | 原生解码对照 |
| [EXL3 TP4](configs/k100ai/exl3-tp4-prod.json) | 4× K100AI 64 GiB | EXL3 2.9 bpw | DSpark |
| [GGUF TP2](configs/k100ai/gguf-tp2-prod.json) | 2× K100AI 64 GiB | GGUF 2.00 bpw | DSpark |

三套配置均通过 PPL 相对基线不超过 `+2%` 和固定四项 API 测试。完整参数、质量结果和优化过程见 [DeepSeek-V4.1 指南](docs/k100ai/DEEPSEEK41.md)。

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

Qwen3.8-Flash-Next 在单张 K100AI 64 GiB 上运行。v0.3.1 的生产入口是
IQ3_S 精确引擎，旧 IQ3_XXS 快速档和 0.1.55 对照档继续保留用于兼容测试。

| 引擎 | 二进制 | 用途 |
| --- | --- | --- |
| **IQ3_S 精确档** | `bin/strata-qwen-iq3s-r112` | v0.3.1 默认生产引擎，完整路由与目标模型验证 MTP |
| 0.1.47 快速档 | `bin/strata-0.1.47` | 旧 IQ3_XXS 兼容，允许数值漂移 |
| 0.1.55 对照档 | `bin/strata-0.1.55` | 旧 canonical / A-B 对照 |

IQ3_S 精确档固定完整路由 top-10、65,536-token KV 驻留、目标模型验证的
MTP，关闭 prompt-lookup 后缀投机。旧 0.1.47 快速档不保证与对照档逐位
一致。完整参数和成绩见 [Qwen v0.3.1 指南](docs/k100ai/QWEN31.md)。

</details>

## 安装

### 下载发布包

**[下载 Strata for K100AI v0.3.1 →](https://github.com/DocPang/strata-k100ai/releases/tag/v0.3.1)**

```bash
BASE=https://github.com/DocPang/strata-k100ai/releases/download/v0.3.1

curl -fLO "$BASE/strata-k100ai-v0.3.1.tar.gz"
curl -fLO "$BASE/strata-k100ai-v0.3.1.tar.gz.sha256"
sha256sum -c strata-k100ai-v0.3.1.tar.gz.sha256

tar -xzf strata-k100ai-v0.3.1.tar.gz
cd strata-k100ai-v0.3.1
sha256sum -c SHA256SUMS
```

也可以检出对应 tag：

```bash
git clone --branch v0.3.1 --depth 1 \
  https://github.com/DocPang/strata-k100ai.git
cd strata-k100ai
sha256sum -c SHA256SUMS
```

### 安装运行环境

```bash
bash scripts/k100ai/setup-runtime.sh
bash scripts/k100ai/preflight.sh
```

运行环境固定为 DTK 2604 镜像。完整镜像 digest、设备映射和模型资产步骤见 [安装指南](docs/k100ai/INSTALL.md)。

### 启动 Qwen3.8-Flash-Next

安装运行时和 Qwen 外置资产：

```bash
bash scripts/k100ai/setup-runtime.sh

K100AI_ASSETS_DIR=/data/k100ai-assets \
  bash scripts/k100ai/fetch-qwen-iq3s.sh \
  /data/k100ai-assets/qwen3.8-flash-next-iq3-s
```

启动 v0.3.1 默认生产引擎：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

旧 IQ3_XXS 快速档通过独立入口启动：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen-iq3xxs
```

旧 0.1.55 对照引擎：

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

### 启动 DeepSeek-V4.1-Flash

先检查配置中的 `tokenizer`、`--pack` 和 `--expert-profile` 是否指向本机资产，再选择一套配置：

```bash
bash scripts/k100ai/start-dsv41.sh configs/k100ai/exl3-tp4-prod.json
```

同一时间只启动一套模型服务。切换配置前，请停止当前服务并重新执行设备占用检查。

### 发送第一条请求

Qwen：

```bash
curl --fail http://127.0.0.1:18096/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 64,
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
    "max_tokens": 64,
    "temperature": 0
  }'
```

Qwen 使用端口 `18096` 和模型名 `qwen3.8-flash-next`。服务同时提供 `/health`、`/v1/models`、`/v1/messages` 和 `/metrics`。

## 性能

### Qwen3.8-Flash-Next IQ3_S

测试条件为单张 K100AI 64 GiB、IQ3_S、INT8 KV、24,576 expert slots、
完整路由 top-10、prefill chunk 20,480。Decode 使用 32,768-token prompt
和 256-token 输出，temperature 0、无 prompt cache。MTP 使用
`spec 3`、`min-p 0.5`、`window 4096` 和 Q4 draft head。

#### Prefill 详细成绩

8K 和 32K 各使用 8 个内容不同、长度固定的 prompt。计时来自引擎
`prompt ... read in ... ms`，不是端到端 TTFT 换算。

| Prompt | 第 1 次 | 第 2 次 | 第 3 次 | 第 4 次 | 第 5 次 | 第 6 次 | 第 7 次 | 第 8 次 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 tok/s | 2,665.1 | 2,718.1 | 2,719.3 | 2,716.2 | 2,714.9 | 2,713.3 | 2,715.6 | 2,716.4 |
| 32,768 tok/s | 2,848.0 | 2,863.9 | 2,859.2 | 2,856.9 | 2,850.0 | 2,853.3 | 2,846.4 | 2,847.5 |

8K 中位 **2,715.9 tok/s**，耗时 3.013–3.074 s。32K 中位
**2,851.65 tok/s**，耗时 11.442–11.512 s。v0.3.1 发布时另行复测的
32K Native 为 2,856.4–2,869.4 tok/s，MTP 为 2,837.3–2,851.3 tok/s。

#### Decode 详细成绩

| 轮次 | Prompt Prefill | Native Decode | MTP Decode | MTP Draft 接受 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11,471.7 ms / 2,856.4 tok/s | **45.7 tok/s** | **70.8 tok/s** | 132 / 188 |
| 2 | 11,420.0 ms / 2,869.4 tok/s | **45.8 tok/s** | **68.4 tok/s** | 128 / 193 |
| 3 | 11,440.8 ms / 2,864.1 tok/s | **45.8 tok/s** | **71.7 tok/s** | 133 / 180 |
| 4 | 11,443.9 ms / 2,863.4 tok/s | **45.9 tok/s** | **69.2 tok/s** | 128 / 185 |

Native 中位 **45.8 tok/s**，MTP 中位 **70.0 tok/s**，对应约 **1.53×**
输出加速。四组 Native / MTP 的 256-token 输出 SHA-256 逐轮相同，
input exact、output exact 和 retrieval 均为 8/8 通过。

### DeepSeek-V4.1-Flash

DeepSeek 数字按标称输入长度除以记录的 API 首 token 时间换算，是端到端
等效吞吐约数，不能与上面 Qwen 引擎 Prefill 直接比较。

| 配置 | 4K 输入约数 | 32K 输入约数 | 投机 Decode |
| --- | ---: | ---: | ---: |
| EXL3 TP1 | 1,374 tok/s | 1,825 tok/s | Native 18.7 tok/s |
| EXL3 TP4 | 1,013 tok/s | 1,121 tok/s | DSpark 43.9 tok/s |
| GGUF TP2 | 505 tok/s | 625 tok/s | DSpark 中位 42.05 tok/s |

Qwen 的完整运行参数、每轮耗时和输出哈希见
[Qwen v0.3.1 指南](docs/k100ai/QWEN31.md)。DeepSeek 原始计时和 PPL 见
[DeepSeek-V4.1 指南](docs/k100ai/DEEPSEEK41.md)。

## 配置

所有 K100AI 配置集中在 [configs/k100ai](configs/k100ai/)，JSON 可以直接审阅和修改。

| 需求 | 建议 |
| --- | --- |
| 四卡，优先 4K 首 token 延迟 | [EXL3 TP1](configs/k100ai/exl3-tp1-prod.json) |
| 四卡，优先持续输出速度 | [EXL3 TP4](configs/k100ai/exl3-tp4-prod.json) |
| 两张卡，或已有 GGUF 2.00 bpw 资产 | [GGUF TP2](configs/k100ai/gguf-tp2-prod.json) |
| 单卡 Qwen，生产、完整路由和无近似 Prefill 算法 | `scripts/k100ai/serve-k100ai.sh qwen` |
| 单卡 Qwen，复现旧 IQ3_XXS 快速档 | `scripts/k100ai/serve-k100ai.sh qwen-iq3xxs` |

配置字段、资产路径和生产参数说明见 [配置指南](configs/k100ai/README.md)。

## 工具

**Claude Code · Codex CLI · OpenCode · Hermes Agent · OpenClaw**，以及更多兼容 OpenAI 或 Anthropic API 的客户端。

应用连接 API 时使用：

| 模型 | 地址 | 模型名 |
| --- | --- | --- |
| Qwen3.8-Flash-Next | `http://127.0.0.1:18096/v1` | `qwen3.8-flash-next` |
| DeepSeek-V4.1-Flash | `http://127.0.0.1:18097/v1` | `deepseek-v4.1-flash` |

默认只监听本机。需要从其他机器访问时，通过 SSH 隧道或项目支持的访问控制方式暴露服务。

## 文档

| 文档 | 内容 |
| --- | --- |
| [K100AI 文档索引](docs/k100ai/README.md) | 安装、运行、验证、来源与版本记录 |
| [安装指南](docs/k100ai/INSTALL.md) | 模型资产、运行环境和启动 |
| [运行指南](docs/k100ai/RUNTIME.md) | 引擎切换、运行参数和 API |
| [Qwen v0.3.1 指南](docs/k100ai/QWEN31.md) | IQ3_S 安装、算法边界、完整成绩和复现 |
| [DeepSeek-V4.1](docs/k100ai/DEEPSEEK41.md) | 三套配置、R001–R179 和原始计时 |
| [验证记录](docs/k100ai/VALIDATION.md) | 性能、精度、算子和 API 测试 |
| [来源与校验](docs/k100ai/PROVENANCE.md) | 二进制、源码与运行资产 |
| [复现指南](docs/k100ai/REPRODUCIBILITY.md) | 固定输入、构建与验收 |

## 帮助

<a id="how-does-it-work"></a>
<a id="how-fast-is-it"></a>
<a id="something-went-wrong"></a>

[报告问题](https://github.com/DocPang/strata-k100ai/issues/new/choose) · [讨论与提问](https://github.com/DocPang/strata-k100ai/issues) · [参与贡献](https://github.com/DocPang/strata-k100ai/blob/release/v0.3.1/CONTRIBUTING.md) · [上游 Strata](https://github.com/Niko1221/Strata)

## 使用边界

- 当前发布环境为 **K100AI/gfx928 + DTK 2604**，其他架构需要单独构建和验证。
- DeepSeek 仅验证 EXL3 TP1、EXL3 TP4 和 GGUF TP2 三套拓扑。
- Qwen v0.3.1 生产模板验证到 32K 输入并配置 65,536-token KV。旧引擎的
  262K 上限属于历史兼容路径，完整窗口尚未完成质量评测。
- DeepSeek 1M 是配置上限，完整窗口尚未完成质量评测。
- TP1、TP4 的 32K 首 token 延迟没有达到既定目标。
- 模型、量化 pack、MTP、Engram 和 DSpark 分别适用其上游许可证。

[MIT 许可证](LICENSE) · [上游项目](https://github.com/Niko1221/Strata) · 由独立维护者适配和发布。
