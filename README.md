<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="300">
  </picture>
</p>

<h1 align="center">Strata for K100AI</h1>

<p align="center">
  <a href="#模型">模型</a> ·
  <a href="#安装">安装</a> ·
  <a href="#硬件与配置">硬件与配置</a> ·
  <a href="#性能">性能</a> ·
  <a href="#工具">工具</a> ·
  <a href="#帮助">帮助</a>
</p>

这是 Strata 的海光 K100AI 适配版，运行在 Linux x86_64 / gfx928 / DTK 2604，提供 OpenAI 和 Anthropic 兼容 API。

[下载 v0.3.2](https://github.com/DocPang/strata-k100ai/releases/tag/v0.3.2) · [发布说明](docs/k100ai/release-notes/v0.3.2.md) · [文档](docs/k100ai/README.md)

## 模型

一张 64 GiB K100AI 可以跑 **Qwen3.8-Flash-Next IQ3_S**。这是默认 Qwen 配置，保留完整的 top-10 专家路由，MTP 候选由目标模型验证。单卡在 32K 输入下，原生解码中位 **45.8 tok/s**，开启 MTP 后为 **70.0 tok/s**，各测四次。详见 [Qwen 指南](docs/k100ai/QWEN31.md)。

**DeepSeek-V4.1-Flash** 有双卡 GGUF 和四卡 EXL3 配置，沿用 v0.3.0 的引擎、参数和实测记录。选好配置后，按 [DeepSeek 安装步骤](docs/k100ai/INSTALL.md#deepseek-v41-flash) 准备对应的模型资产。

## 安装

在 K100AI 主机上下载并校验发布包。主机需要 Docker 和可用的海光驱动。Qwen 的模型和运行资产约占 140 GB 磁盘，由安装脚本另行准备。

```bash
BASE=https://github.com/DocPang/strata-k100ai/releases/download/v0.3.2
curl -fLO "$BASE/strata-k100ai-v0.3.2.tar.gz"
curl -fLO "$BASE/strata-k100ai-v0.3.2.tar.gz.sha256"
sha256sum -c strata-k100ai-v0.3.2.tar.gz.sha256 &&
  tar -xzf strata-k100ai-v0.3.2.tar.gz &&
  cd strata-k100ai-v0.3.2 &&
  sha256sum -c SHA256SUMS &&
  bash scripts/k100ai/setup-runtime.sh &&
  bash scripts/k100ai/preflight.sh
```

准备资产并启动 Qwen。

```bash
export K100AI_ASSETS_DIR=/data/k100ai-assets
bash scripts/k100ai/fetch-qwen-iq3s.sh "$K100AI_ASSETS_DIR/qwen3.8-flash-next-iq3-s" &&
  PORT=18096 bash scripts/k100ai/serve-k100ai.sh qwen
```

DeepSeek、旧 IQ3_XXS 和自定义资产路径见 [安装指南](docs/k100ai/INSTALL.md)。

<a id="配置"></a>

## 硬件与配置

以下卡数按每张 K100AI 64 GiB 计算。

| 卡数 | 配置 | 用途 |
| ---: | --- | --- |
| 1 | [Qwen IQ3_S](configs/k100ai/qwen-iq3s-r112.template.json) | 256K 上下文，64K KV 驻留 |
| 2 | [DeepSeek GGUF TP2](configs/k100ai/gguf-tp2-prod.json) | GGUF 2.00 bpw，开启 DSpark |
| 4 | [DeepSeek EXL3 TP1](configs/k100ai/exl3-tp1-prod.json) | EXL3 2.9 bpw，优先首 token 延迟 |
| 4 | [DeepSeek EXL3 TP4](configs/k100ai/exl3-tp4-prod.json) | EXL3 2.9 bpw，开启 DSpark |

Qwen IQ3_S 默认支持 262,144-token 上下文，其中 65,536 个 KV 位置驻留在显存。已用 261,632 token 输入和 256 token 输出验证长文检索。设备映射、配置字段和引擎切换见 [配置指南](configs/k100ai/README.md)和 [运行指南](docs/k100ai/RUNTIME.md)。

<a id="how-fast-is-it"></a>

## 性能

### Qwen3.8-Flash-Next IQ3_S · v0.3.1

单张 K100AI 64 GiB，32,768-token 输入、256-token 输出，temperature 0，关闭 prompt cache。每种模式测四次，Prefill 和 Decode 均按引擎日志计时。

| 解码模式 | Prefill（tok/s） | Decode（tok/s） | Decode 中位（tok/s） |
| --- | ---: | ---: | ---: |
| 原生 | 2,856.4–2,869.4 | 45.7–45.9 | 45.8 |
| MTP | 2,837.3–2,851.3 | 68.4–71.7 | **70.0** |

四组原生和 MTP 输出的 SHA-256 逐组一致。另有一组独立 Prefill 测试，8K 和 32K 各用八个不同的 prompt，中位速度分别为 **2,715.9** 和 **2,851.65 tok/s**。[逐轮数据与参数](docs/k100ai/QWEN31.md)

### DeepSeek-V4.1-Flash · v0.3.0

沿用原发布成绩。4K 和 32K 输入速度由标称 token 数除以 API 首 token 时间换算，反映的是端到端等效吞吐。四卡 Decode 来自生产安装后的验收，双卡 DSpark 取四组样本中位数。

| 配置 | 4K 输入约值（tok/s） | 32K 输入约值（tok/s） | Decode（tok/s） |
| --- | ---: | ---: | ---: |
| 四卡 EXL3 TP1 | 1,374 | 1,825 | 原生 18.7 |
| 四卡 EXL3 TP4 | 1,013 | 1,121 | DSpark 43.9 |
| 双卡 GGUF TP2 | 505 | 625 | DSpark 42.05（中位） |

[首 token 时间、测试样本与 PPL](docs/k100ai/DEEPSEEK41.md)

## 工具

Codex CLI、Claude Code 等客户端可通过兼容 API 连接。表中是 OpenAI 兼容地址。Anthropic 兼容接口使用同一端口下的 `/v1/messages`。

| 模型 | OpenAI Base URL | 模型名 |
| --- | --- | --- |
| Qwen | `http://127.0.0.1:18096/v1` | `qwen3.8-flash-next` |
| DeepSeek | `http://127.0.0.1:18097/v1` | `deepseek-v4.1-flash` |

另开终端，用一条请求确认 Qwen 能生成回复。

```bash
curl --fail http://127.0.0.1:18096/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"你好"}],"max_tokens":32,"temperature":0}'
```

[API 示例与验收脚本](docs/k100ai/RUNTIME.md#api)

<a id="how-does-it-work"></a>

## 文档

[文档索引](docs/k100ai/README.md)汇总了安装、配置和各版本说明。查性能和精度用 [验证记录](docs/k100ai/VALIDATION.md)，从源码构建或复跑测试用 [复现指南](docs/k100ai/REPRODUCIBILITY.md)。

<a id="something-went-wrong"></a>

## 帮助

遇到安装或启动问题，提交 Issue 时附上版本、卡数和相关日志。

[提交 Issue](https://github.com/DocPang/strata-k100ai/issues/new/choose) · [参与贡献](https://github.com/DocPang/strata-k100ai/blob/v0.3.2/CONTRIBUTING.md) · [上游 Strata](https://github.com/Niko1221/Strata) · [MIT 许可证](LICENSE)
