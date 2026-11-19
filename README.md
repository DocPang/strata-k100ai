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
  <a href="https://github.com/DocPang/strata-k100ai/releases/tag/v0.3.0"><img src="https://img.shields.io/badge/version-v0.3.0-0ab887" alt="v0.3.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/DocPang/strata-k100ai" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/platform-K100AI%20%2F%20gfx928-5b67ff" alt="K100AI / gfx928">
  <img src="https://img.shields.io/badge/DTK-2604-3d6bff" alt="DTK 2604">
</p>

**免费、开源，在你的 K100AI 集群上运行。** 提供 OpenAI 与 Anthropic 兼容 API，可直接连接现有客户端和编程工具。

K100AI 版本 **v0.3.0** · **Linux x86_64 / gfx928 / DTK 2604** · [发布说明](docs/k100ai/release-notes/v0.3.0.md)

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

Qwen3.8-Flash-Next 在单张 K100AI 上运行，发布包保留默认快速引擎和对照引擎。

| 引擎 | 二进制 | 用途 |
| --- | --- | --- |
| **0.1.47** | `bin/strata-0.1.47` | 默认快速档，启用 gfx928 prompt MMAC 与 MTP |
| 0.1.55 | `bin/strata-0.1.55` | 对照档，便于精度和性能比较 |

快速档允许数值漂移，不保证与对照档逐位一致。运行参数和资产准备见 [K100AI 运行指南](docs/k100ai/RUNTIME.md)。

</details>

## 安装

### 下载发布包

**[下载 Strata for K100AI v0.3.0 →](https://github.com/DocPang/strata-k100ai/releases/tag/v0.3.0)**

```bash
BASE=https://github.com/DocPang/strata-k100ai/releases/download/v0.3.0

curl -fLO "$BASE/strata-k100ai-v0.3.0.tar.gz"
curl -fLO "$BASE/strata-k100ai-v0.3.0.tar.gz.sha256"
sha256sum -c strata-k100ai-v0.3.0.tar.gz.sha256

tar -xzf strata-k100ai-v0.3.0.tar.gz
cd strata-k100ai-v0.3.0
sha256sum -c SHA256SUMS
```

也可以检出对应 tag：

```bash
git clone --branch v0.3.0 --depth 1 \
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

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

如需切换到 0.1.55 对照引擎：

```bash
K100AI_QWEN_ENGINE=bin/strata-0.1.55 \
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

### 启动 DeepSeek-V4.1-Flash

先检查配置中的 `tokenizer`、`--pack` 和 `--expert-profile` 是否指向本机资产，再选择一套配置：

```bash
bash scripts/k100ai/start-dsv41.sh configs/k100ai/exl3-tp4-prod.json
```

同一时间只启动一套模型服务。切换配置前，请停止当前服务并重新执行设备占用检查。

### 发送第一条请求

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

### 输入处理

输入速度单位为 **tok/s**，数值越高越好。Qwen 使用引擎 prefill 计时。DeepSeek 的 4K/32K 数字按标称输入长度除以已记录的 API 首 token 时间换算，是**端到端等效吞吐约数**；原始记录未保存精确 prompt token 数和引擎内部 prefill 用时，不能与 Qwen 直接比较。

| 模型 / 配置 | 测试输入 | 输入速度 | 计时口径 |
| --- | ---: | ---: | --- |
| Qwen 0.1.47 | 8,190 tokens | **1,846.2 tok/s** | 引擎 prefill |
| Qwen 0.1.47 | 40,057 tokens | **2,086.1 tok/s** | 引擎 prefill |
| DeepSeek EXL3 TP1 | 标称 4K | 约 1,374 tok/s | 端到端约数 |
| DeepSeek EXL3 TP1 | 标称 32K | 约 1,825 tok/s | 端到端约数 |
| DeepSeek EXL3 TP4 | 标称 4K | 约 1,013 tok/s | 端到端约数 |
| DeepSeek EXL3 TP4 | 标称 32K | 约 1,121 tok/s | 端到端约数 |
| DeepSeek GGUF TP2 | 标称 4K | 约 505 tok/s | 端到端约数 |
| DeepSeek GGUF TP2 | 标称 32K | 约 625 tok/s | 端到端约数 |

### 输出处理

输出速度仅列投机解码开启时的实测值。DeepSeek 使用 DSpark，Qwen 使用 MTP。

| 模型 / 配置 | 投机算法 | 输出速度 | 说明 |
| --- | --- | ---: | --- |
| DeepSeek EXL3 TP4 | DSpark | **43.9 tok/s** | 生产安装后的 smoke |
| DeepSeek GGUF TP2 | DSpark | **42.05 tok/s** | 四组样本中位数 |
| Qwen 0.1.47 | MTP `spec=6` | **79.37 tok/s** | 8,191-token 输入，64-token 输出 |

DeepSeek TP1 的发布配置关闭 DSpark，现有记录只有原生解码 18.7 tok/s，因此不列入投机解码成绩表。DSpark 输出速度会随内容和 draft 命中率波动。Qwen 验收见[验证记录](docs/k100ai/VALIDATION.md#qwen-精度和快速档验收)，DeepSeek 原始计时和 PPL 见 [DeepSeek-V4.1 指南](docs/k100ai/DEEPSEEK41.md)。

## 配置

所有 K100AI 配置集中在 [configs/k100ai](configs/k100ai/)，JSON 可以直接审阅和修改。

| 需求 | 建议 |
| --- | --- |
| 四卡，优先 4K 首 token 延迟 | [EXL3 TP1](configs/k100ai/exl3-tp1-prod.json) |
| 四卡，优先持续输出速度 | [EXL3 TP4](configs/k100ai/exl3-tp4-prod.json) |
| 两张卡，或已有 GGUF 2.00 bpw 资产 | [GGUF TP2](configs/k100ai/gguf-tp2-prod.json) |
| 单卡 Qwen，优先输入吞吐 | `scripts/k100ai/serve-k100ai.sh qwen` |

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
| [DeepSeek-V4.1](docs/k100ai/DEEPSEEK41.md) | 三套配置、R001–R179 和原始计时 |
| [验证记录](docs/k100ai/VALIDATION.md) | 性能、精度、算子和 API 测试 |
| [来源与校验](docs/k100ai/PROVENANCE.md) | 二进制、源码与运行资产 |
| [复现指南](docs/k100ai/REPRODUCIBILITY.md) | 固定输入、构建与验收 |

## 帮助

<a id="how-does-it-work"></a>
<a id="how-fast-is-it"></a>
<a id="something-went-wrong"></a>

[报告问题](https://github.com/DocPang/strata-k100ai/issues/new/choose) · [讨论与提问](https://github.com/DocPang/strata-k100ai/issues) · [参与贡献](https://github.com/DocPang/strata-k100ai/blob/release/v0.3.0/CONTRIBUTING.md) · [上游 Strata](https://github.com/Niko1221/Strata)

## 使用边界

- 当前发布环境为 **K100AI/gfx928 + DTK 2604**，其他架构需要单独构建和验证。
- DeepSeek 仅验证 EXL3 TP1、EXL3 TP4 和 GGUF TP2 三套拓扑。
- Qwen 256K、DeepSeek 1M 是配置上限，完整窗口尚未完成质量评测。
- TP1、TP4 的 32K 首 token 延迟没有达到既定目标。
- 模型、量化 pack、MTP、Engram 和 DSpark 分别适用其上游许可证。

[MIT 许可证](LICENSE) · [上游项目](https://github.com/Niko1221/Strata) · 由独立维护者适配和发布。
