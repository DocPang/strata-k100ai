<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="280">
  </picture>
</p>

<h1 align="center">Strata for K100AI</h1>

<p align="center">在海光 K100AI 上运行 DeepSeek V4.1 与 Qwen3.8，提供 OpenAI / Anthropic 兼容 API。</p>

<p align="center">
  <a href="https://github.com/DocPang/strata-k100ai/releases/latest"><b>下载 v0.2.3</b></a> ·
  <a href="#实测性能">性能</a> ·
  <a href="#安装与运行">安装</a> ·
  <a href="docs/k100ai/INSTALL.md">完整手动步骤</a> ·
  <a href="docs/k100ai/REPRODUCIBILITY.md">源码复现</a>
</p>

K100AI 版本 **v0.2.3** · 上游基线 **Strata v0.2.9** · **Linux x86_64 / gfx928 / DTK 2604**

## 实测性能

### DeepSeek-V4.1-Flash · 四卡

**4× K100AI 64 GiB**，IQ2_XXS、FP4 KV、DSpark Q8_0；模型按 `10,10,10,10` 分层切分，routed experts 全部驻留 GPU。

| 项目 | 实测结果 | 测试条件 |
| --- | ---: | --- |
| 混合内容长输出 | 14.5–27.7 tok/s | 256-token 输出，测速档 2K–32K、64K–256K；DSpark 开启 |
| DSpark 关闭基线 | 24.32 / 24.25 tok/s | 相同 25-token 输入、64-token 输出 |
| **短输出 DSpark 峰值** | **69.19 tok/s** | 25-token 输入，64-token 规则重复输出，DSpark 开启 |
| DSpark 对照峰值 | 2.82× | 仅限上述 25-token 输入、64-token 输出的 off / on / off 对照 |
| 输入处理速度 | 300.45 tok/s | 4,095-token prompt，13.629 秒 |
| 长输入稳定性 | 120,029 tokens | 独特 prompt tokens，请求完成、服务稳定 |

69.19 tok/s 是真实测得的短输出峰值，不是适用于任意内容或任意输出长度的持续吞吐。
该测试的 64-token 输出是 `455,14083,2645,344,260,1688,2645,16` 连续重复 8 次；
DSpark 在 11 轮内接受 **52 / 54** 个 draft，命中率 96.3%。相同短序列在 DSpark
关闭时为 24.32 / 24.25 tok/s，因此 2.82× 也只描述这一有利工况。

### DeepSeek-V4.1-Flash · 十档长输出

单并发、`temperature=0`、`top_p=1`、DSpark 开启，257,945 token 以内各档均无 KV
cache 命中。输出速度使用服务端 timings；TTFT 和总耗时使用客户端墙钟。

| 目标输入 | 实际 prompt | TTFT | 预填充 | 输出 | 输出 tokens | 说明 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 556 | 2.079 s | 121.6 tok/s | 26.8 tok/s | 256 | 正常 |
| 2,048 | 2,092 | 3.685 s | 274.9 tok/s | 14.5 tok/s | 256 | 正常 |
| 4,096 | 4,141 | 5.768 s | 349.7 tok/s | 16.9 tok/s | 256 | 正常 |
| 8,192 | 8,237 | 9.948 s | 415.0 tok/s | 22.0 tok/s | 160 | 自然 EOS |
| 12,288 | 12,333 | 13.632 s | 452.4 tok/s | 25.8 tok/s | 256 | 正常 |
| 16,384 | 16,429 | 37.289 s | 201.8 tok/s | **3.0 tok/s** | 256 | **异常，排除出正常区间统计** |
| 32,768 | 32,813 | 38.409 s | 426.9 tok/s | 27.7 tok/s | 256 | 正常 |
| 65,536 | 65,581 | 81.431 s | 399.8 tok/s | 26.0 tok/s | 256 | 正常 |
| 131,072 | 131,117 | 191.209 s | 333.8 tok/s | 24.2 tok/s | 256 | 正常 |
| 257,901 | 257,945 | 355.155 s | 363.5 tok/s | 26.3 tok/s | 256 | 正常 |

16K 档的 3.0 tok/s 是异常点：引擎日志的前约 128-token 阶段卡在 1.69 tok/s，
随后约 111-token 阶段恢复到 27.50 tok/s。相邻 12K、32K 档分别正常，因此它不是
上下文容量或单调长上下文退化。排除该异常后，正常档输出速度中位数约 **25.8 tok/s**。
完整 1M 窗口尚未灌满做质量评测。

配置上下文上限为 **1,048,576 tokens**；完整 1M 窗口尚未灌满做质量评测。[测试条件与验证记录 →](docs/k100ai/VALIDATION.md#deepseek-四卡dspark-和长输入)

### Qwen3.8-Flash-Next · 单卡

**1× K100AI 64 GiB**，IQ3_XXS、INT8 KV、MTP `spec=6`。相同输入和环境下，两档引擎的 prefill（输入处理）吞吐：

| 输入长度 | 0.1.47 默认快速档 | 0.1.55 对照档 | 快速档提升 |
| --- | ---: | ---: | ---: |
| 8,190 tokens | **1,846.2 tok/s** | 1,033.6 tok/s | 约 1.8× |
| 40,057 tokens | **2,086.1 tok/s** | 1,161.3 tok/s | 约 1.8× |

快速档在 8,191-token 输入、64-token 输出测试中生成速度为 **79.37 tok/s**，MTP 接受 **56 / 57**。配置上下文上限为 **262,144 tokens**。

默认使用 0.1.47，原因是 0.1.55 的 gfx928 prompt attention 仍有性能回归。快速档允许数值漂移：8K / 40K 验证中 top-1 相同，logits 余弦相似度为 0.9915 / 0.9870，40K 的前 8 个 greedy token 相同；它不保证逐位一致。[精度与性能明细 →](docs/k100ai/VALIDATION.md#qwen-精度和快速档验收)

## 安装与运行

需要 Linux x86_64、K100AI/gfx928、Hygon 驱动、HyHAL 和 Docker。使用固定的 DTK 2604 镜像；完整镜像 digest、模型地址和 SHA-256 见[手动安装文档](docs/k100ai/INSTALL.md)。

### 1. 下载并安装运行包

```bash
BASE_URL=https://github.com/DocPang/strata-k100ai/releases/download/v0.2.3
curl -fLO "$BASE_URL/strata-v0.2.9-k100ai-dual.tar.gz"
curl -fLO "$BASE_URL/strata-v0.2.9-k100ai-dual.tar.gz.sha256"
sha256sum -c strata-v0.2.9-k100ai-dual.tar.gz.sha256
mkdir /data/strata-k100ai-v0.2.3
tar -xzf strata-v0.2.9-k100ai-dual.tar.gz -C /data/strata-k100ai-v0.2.3 --strip-components=1
cd /data/strata-k100ai-v0.2.3
sha256sum -c SHA256SUMS
bash scripts/k100ai/setup-runtime.sh
```

归档名中的 `v0.2.9` 是上游基线，下载链接中的 `v0.2.3` 是本仓库发布版本。

### 2. 准备模型资产

权重、MTP runtime 和模型 pack 需要另行下载或生成，不随运行包分发。先按[安装文档第 2 步](docs/k100ai/INSTALL.md#2-安装-python-运行时)设置路径与镜像变量，再准备所需模型。默认资产目录为 `/data/k100ai-assets`：

- **DeepSeek**：[下载、校验权重与 DSpark](docs/k100ai/INSTALL.md#6-手动下载并校验-deepseek-资产) → [生成 pack](docs/k100ai/INSTALL.md#7-人工生成-deepseek-pack)。
- **Qwen**：[下载、校验权重](docs/k100ai/INSTALL.md#3-手动下载并校验-qwen-主权重) → [生成 pack](docs/k100ai/INSTALL.md#4-人工生成-qwen-pack) → [准备 MTP](docs/k100ai/INSTALL.md#5-人工准备-qwen-mtp)。

pack 生成步骤会导出 tokenizer，Qwen expert profile 随运行包提供。下载和转换需要数百 GB 磁盘空间。

### 3. 启动服务

准备好资产后，按需选择一条命令；DeepSeek 使用四卡，Qwen 使用单卡。

```bash
# DeepSeek：API http://127.0.0.1:18097/v1
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18097 bash scripts/k100ai/serve-k100ai.sh dsv41
```

```bash
# Qwen：API http://127.0.0.1:18096/v1
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 bash scripts/k100ai/serve-k100ai.sh qwen
```

启动器检查设备占用和资产文件；按 `Ctrl-C` 停止服务。默认仅监听本机。手动 Docker 启动命令见[安装文档](docs/k100ai/INSTALL.md#9-人工渲染并启动-qwen)。

### 4. 发送请求

以 DeepSeek 为例：

```bash
curl --fail http://127.0.0.1:18097/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"请用一句话介绍你自己。"}],"max_tokens":64,"temperature":0}'
```

接入 Qwen 时改用端口 `18096`、模型名 `qwen3.8-flash-next`。服务同时提供 `/health`、`/v1/models`、`/v1/messages` 和 `/metrics`。

## 模型与优化

| 模型 / 用途 | 包内二进制 | 引擎 | 加速 | 设备 |
| --- | --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | `bin/strata-dsv41` | 0.1.55 | DSpark | 4× K100AI |
| Qwen 默认快速档 | `bin/strata` | 0.1.47 | MTP | 1× K100AI |
| Qwen 对照档 | `bin/strata-0.1.55` | 0.1.55 | MTP | 1× K100AI |

主要优化覆盖 gfx928 HIP 适配、Qwen 原生 prompt MMAC 与 fused expert、DeepSeek 四卡分层切分与专家驻留、FP4 KV，以及 MTP / DSpark 推测解码。内核路径和参数见[优化说明](docs/k100ai/OPTIMIZATION.md)。

## 复现与文档

发布页同时提供运行包、独立源码包、`RELEASE.json` 和 `validation-evidence.tar.gz`。源码包包含主源码、Qwen 0.1.47 快速档和固定 llama.cpp 快照；构建脚本校验二进制 SHA-256。

| 文档 | 内容 |
| --- | --- |
| [完整手动安装](docs/k100ai/INSTALL.md) | 模型地址、文件校验、pack / MTP 生成、Docker 命令 |
| [运行参数](docs/k100ai/README.md) | 引擎切换、启动器行为、API 验收 |
| [源码复现](docs/k100ai/REPRODUCIBILITY.md) | 固定镜像、源码输入、从零构建 |
| [验证记录](docs/k100ai/VALIDATION.md) | 性能、数值差异、算子与 API 测试 |
| [来源与校验](docs/k100ai/PROVENANCE.md) | 二进制、模型与运行资产的来源和摘要 |
| [版本记录](CHANGELOG.md) | 版本变更 |

## 使用范围

- 当前发布验证环境为 **K100AI/gfx928 + DTK 2604**。其他 GPU 架构需要单独构建和验证。
- **本版不支持图片输入**；Qwen 256K 和 DeepSeek 1M 是配置上限，未完成整个窗口的质量评测。
- 从 v0.2.1 升级时请解包到新目录，重新准备或迁移资产；旧 `install.sh`、`start.sh`、`compose.yaml` 不再适用。
- 项目代码采用 [MIT](LICENSE)；模型、量化 pack、MTP、Engram 与 DSpark 分别适用其上游许可证。

基于 [Strata](https://github.com/Niko1221/Strata) 适配，由独立维护者发布。
