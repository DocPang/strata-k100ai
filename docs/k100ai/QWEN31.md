# Qwen3.8-Flash-Next v0.3.1 指南

[返回首页](../../README.md) · [文档索引](README.md) · [安装](INSTALL.md) · [运行](RUNTIME.md) · [成绩证据](../../data/k100ai/qwen-iq3s-r112-evidence/README.md)

## 结论

v0.3.1 的 Qwen 生产引擎是 `bin/strata-qwen-iq3s-r112`。它在单张
K100AI 64 GiB 上运行 IQ3_S，32K Native Decode 为
**45.7–45.9 tok/s**，目标模型验证的 MTP Decode 为
**68.4–71.7 tok/s**，Prefill 为 **2,837.3–2,869.4 tok/s**。

该路径固定完整路由 top-10，关闭 prompt-lookup 后缀投机，不启用
Top-k 专家截断或其他近似 Prefill 算法。MTP 是候选建议，所有 token
仍由目标模型验证后提交。

## 版本与资产

| 项目 | v0.3.1 |
| --- | --- |
| 运行平台 | Linux x86_64，K100AI / gfx928，DTK 2604 |
| 引擎 | `bin/strata-qwen-iq3s-r112` |
| 引擎 SHA-256 | `102b4b412068a68588883c13a28103600d8b078785da9dcbbda34c440e9a5074` |
| 模型 | [Qwen3.8-Flash-Next GSQ-RCO IQ3_S](https://www.modelscope.cn/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |
| 模型 revision | `6ac9406afe74db316b11a3865a3382c7bedbcf34` |
| 设备数 | 1 |
| KV | INT8 |
| 上下文 / KV 驻留 | 65,536 / 65,536 |
| 路由 | 完整 top-10 |
| Prefill chunk | 20,480 |
| MTP | spec 3，min-p 0.5，window 4096，Q4 all |

模型、pack 和 MTP runtime 不随主发布包提供。发布包内提供 238 KB 的
draft vocabulary `data/k100ai-qwen-iq3s-draft-vocab-zh60k.bin`。其来源为
[Strata-rdna3 commit `7df6538d`](https://github.com/nicejunjie/Strata-rdna3/blob/7df6538d6c886093df39d74b36ed0a3396bf9467/rdna3/draft_vocab_zh60k.bin)。

## 安装

先安装项目运行依赖：

```bash
bash scripts/k100ai/setup-runtime.sh
bash scripts/k100ai/preflight.sh
```

准备 IQ3_S 模型、专家 pack、MTP runtime 和 tokenizer：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
  bash scripts/k100ai/fetch-qwen-iq3s.sh \
  /data/k100ai-assets/qwen3.8-flash-next-iq3-s
```

脚本先确认 `.python-packages` 和固定版本的 `gguf-py`。发布包已经包含
`third_party/llama.cpp/gguf-py`；源码检出缺少它时，脚本会在大模型下载前
从固定 llama.cpp 提交 `3cf0325` 补齐并校验归档 SHA-256。

脚本会恢复下载、检查文件大小和 SHA-256，并在 DTK 2604 容器中完成 pack
和 MTP runtime 转换。目录结构如下：

```text
/data/k100ai-assets/qwen3.8-flash-next-iq3-s/
  model/IQ3_S/
    Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
    Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
  pack/iq3s/
    experts.bin
    dense.bin
    index.txt
    native_experts.txt
    conversions.json
    experts.bin.src.json
    tokenizer/
  mtp/rt/
    experts.bin
    dense.bin
    dense.txt
    draft_vocab.bin
    draft_vocab_zh60k.bin
```

如果资产位于其他目录，可设置：

```text
K100AI_QWEN_PACK
K100AI_QWEN_NATIVE
K100AI_QWEN_PLE
K100AI_QWEN_TOKENIZER
K100AI_QWEN_MTP_RT
K100AI_QWEN_MTP_VOCAB
```

## 启动

`qwen` 和 `qwen-iq3s` 都启动 v0.3.1 生产引擎：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

`K100AI_MODEL=qwen-iq3s` 与位置参数 `qwen-iq3s` 等价。

旧 IQ3_XXS 快速档使用独立入口：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen-iq3xxs
```

旧 0.1.55 对照引擎仍可通过 `K100AI_QWEN_ENGINE=bin/strata-0.1.55`
切换。旧引擎不属于 v0.3.1 精确路径，也不共享本次成绩。

## 算法路径

### 精确选择

- QSA top-k 使用精确阈值和确定性的最低-index 并列规则。
- H16 两遍 radix 选择与 32-bit 父实现保持相同 ids，不把更浅的
  candidate top-k 当作近似捷径。
- 专家 prefill 默认保留全部 10 个路由专家。
- `STRATA_PREFILL_TOPK=10` 在发布模板中显式固定。源码中的 6/8 是
  需要人工开启的近似诊断路径。
- `--suffix-draft 0` 关闭 prompt-lookup 后缀投机。

### MTP

MTP 只生成候选。目标模型对每个候选窗口执行完整验证，拒绝的 draft 不会
进入输出。发布测试的 4 个 Native / MTP 配对全部具有相同 256-token
输出 SHA-256。

### 数值边界

“无近似 Prefill 算法”描述路由和候选选择路径。模型权重使用 IQ3_S，
KV 使用 INT8，QSA 分数使用发布配置中的 16-bit 表示。本版不声称 IQ3_S
路径与 BF16/FP16 模型逐位相同。

## 详细成绩

### 测试条件

| 项目 | fixed 值 |
| --- | --- |
| 设备 | HCU0，单张 K100AI 64 GiB |
| 模型 | GSQ-RCO IQ3_S |
| Decode prompt / output | 32,768 / 256 tokens |
| Temperature | 0 |
| Prompt cache | 0 |
| Conversation cache | 0 |
| Expert cache | 24,576 slots |
| Prefill chunk | 20,480 |
| KV / context | 65,536 resident / 65,536 |
| 质量门禁 | input exact、output exact、retrieval |

### Native Decode

| 轮次 | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s | TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11,471.7 | 2,856.4 | 5,600.4 | 45.7 | 11.5378 s |
| 2 | 11,420.0 | 2,869.4 | 5,585.7 | 45.8 | 11.4863 s |
| 3 | 11,440.8 | 2,864.1 | 5,586.9 | 45.8 | 11.5058 s |
| 4 | 11,443.9 | 2,863.4 | 5,580.0 | 45.9 | 11.5092 s |

Native 中位 **45.8 tok/s**，每 token `21.797–21.877 ms`，中位
`21.8215 ms`。

### MTP Decode

| 轮次 | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s | Draft 接受 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 11,549.0 | 2,837.3 | 3,618.1 | 70.8 | 132 / 188 |
| 2 | 11,492.9 | 2,851.2 | 3,744.6 | 68.4 | 128 / 193 |
| 3 | 11,492.3 | 2,851.3 | 3,570.1 | 71.7 | 133 / 180 |
| 4 | 11,493.5 | 2,851.0 | 3,697.0 | 69.2 | 128 / 185 |

MTP 中位 **70.0 tok/s**，每 token `13.946–14.627 ms`，中位
`14.287 ms`。相对 Native 的 45.8 tok/s 为 **1.53×**。

### 8K Prefill

| 轮次 | Prompt tokens | Prefill ms | Prefill tok/s | 同次 MTP Decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8,192 | 3,074 | 2,665.1 | 68.9 |
| 2 | 8,192 | 3,014 | 2,718.1 | 69.5 |
| 3 | 8,192 | 3,013 | 2,719.3 | 72.4 |
| 4 | 8,192 | 3,016 | 2,716.2 | 71.0 |
| 5 | 8,192 | 3,017 | 2,714.9 | 71.9 |
| 6 | 8,192 | 3,019 | 2,713.3 | 73.1 |
| 7 | 8,192 | 3,017 | 2,715.6 | 67.1 |
| 8 | 8,192 | 3,016 | 2,716.4 | 70.8 |

8K 中位 **2,715.9 tok/s**。

### 32K Prefill

| 轮次 | Prompt tokens | Prefill ms | Prefill tok/s | 同次 MTP Decode |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32,768 | 11,505 | 2,848.0 | 70.8 |
| 2 | 32,768 | 11,442 | 2,863.9 | 69.2 |
| 3 | 32,768 | 11,460 | 2,859.2 | 71.5 |
| 4 | 32,768 | 11,470 | 2,856.9 | 69.1 |
| 5 | 32,768 | 11,498 | 2,850.0 | 68.6 |
| 6 | 32,768 | 11,484 | 2,853.3 | 72.1 |
| 7 | 32,768 | 11,512 | 2,846.4 | 66.4 |
| 8 | 32,768 | 11,508 | 2,847.5 | 71.2 |

32K 中位 **2,851.65 tok/s**。发布复测覆盖 2,837.3–2,869.4 tok/s。

### 输出一致性

| 配对 | Native / MTP SHA-256 |
| ---: | --- |
| 1 | `93c693fb9293847b6ae181863bca9be71a8421db3c7f0f974f495464f5f04de2` |
| 2 | `831981ddc871d8469176a00848809c42c397935c3e4924d12533d505b7dbb268` |
| 3 | `76849a298e423649b13f5cfefb62c4afbf0a12b80ba34d62c25ddd07f89bd490` |
| 4 | `c49911d19a98f43a8da58464a70221647fcacdbbc588285288c80e3d15c6bfd0` |

## 外部资产校验

### 模型

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| IQ3_S 分片 1 | 54,817,524,224 B | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| IQ3_S 分片 2 | 28,800,138,432 B | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

### Pack

| 文件 | SHA-256 |
| --- | --- |
| `experts.bin` | `ed9d0a872b4d9fb992a05f3f5114107b0689767f0d95e2fb7e2ffe091000e67c` |
| `dense.bin` | `d2af3d8331e5c7a1985d1d589747230f9d56719a6d33c7cddc9ee6b81cde6b13` |
| `index.txt` | `b2a014878c3b61d6a6f915e194055a5e68ede34f9465dd1cb13249d902f008ac` |
| `native_experts.txt` | `d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d` |
| `conversions.json` | `51df3cd6ffd0d38c2da8e2d3a95a604b4a96c20b06fc639a9bf477282a42b86a` |
| `experts.bin.src.json` | `5a18d0c1f89ac64c08998010c81536fcd6b67557778f8f2dfc36539ad0150cad` |

### MTP

| 文件 | SHA-256 |
| --- | --- |
| `experts.bin` | `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f` |
| `dense.bin` | `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a` |
| `dense.txt` | `991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd` |
| `draft_vocab.bin` | `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91` |
| `draft_vocab_zh60k.bin` | `e93f8cd23075d98ccdb4bf10d10c69613a25ec449afc062d0c9dbd33dfb96122` |

## 限制

- Decode 成绩仅覆盖单卡、greedy、32K prompt 和 256 输出。
- MTP 加速取决于 draft 接受率。本批样本区间是 68.4–71.7 tok/s。
- 65,536 是 v0.3.1 生产模板的上下文上限。32K 输入已完成实测，
  完整 65,536-token 语义质量尚未验证。
- 旧 IQ3_XXS 快速档、0.1.55 对照档与 DeepSeek 结果不属于本次
  IQ3_S 精确路径的成绩。
