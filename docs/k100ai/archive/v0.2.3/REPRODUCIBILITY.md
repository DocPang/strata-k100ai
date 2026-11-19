# K100AI 独立复现指南

本页定义公开可复现发布所需的输入、构建步骤、资产准备方法和验收门槛。适用
范围仍严格限定为 Hygon K100AI `gfx928`、DTK 2604 和 Linux x86_64。

## 发布物分工

发布由两个归档组成：

1. `strata-v0.2.9-k100ai-dual.tar.gz`：运行时、四个已构建二进制、配置、
   文档、校验清单和 `gguf-py`。不含模型权重或 MTP runtime。
2. `strata-v0.2.9-k100ai-dual-source.tar.gz`：主源码树、Qwen 0.1.47
   快速源码树和固定 llama.cpp 源树，用于从源码重建四个二进制。

模型资产必须从原始模型仓库取得。发布包不重新分发 Qwen 或 DeepSeek
权重，使用者需要自行接受相应模型许可证。

两个归档都使用排序后的路径、零时间戳、空 owner/group 和固定 gzip 头。
因此同一输入树在支持的 Python 版本上应得到相同的归档 SHA-256；发布的
运行包还带有相邻的 `.sha256` sidecar。

## 固定环境

- 设备：Hygon K100AI，`gfx928`
- 软件：DTK 2604
- 容器：`harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68`
- Qwen 运行设备：1 张 HCU
- DeepSeek 验证设备：4 张 HCU
- 原生 NVIDIA/CUDA 不在本版本支持范围内

镜像 digest 是必需输入，不能只使用可变的 `:latest` 或版本标签。

## 运行包验收

```bash
mkdir -p /data/strata-v029
tar -xzf strata-v0.2.9-k100ai-dual.tar.gz \
  -C /data/strata-v029 --strip-components=1
cd /data/strata-v029
sha256sum -c SHA256SUMS
bash scripts/k100ai/setup-runtime.sh
bash scripts/k100ai/preflight.sh
```

`setup-runtime.sh` 必须在固定镜像内运行，并在成功后写入
`.python-packages/.k100ai-ready`。

## Qwen 资产

运行前准备好：

- 两个固定 ModelScope revision 的 IQ3_XXS GGUF shard
- 由这两个 shard 生成的 IQ3_XXS pack
- 由固定 Qwen BF16 checkpoint 的 31 个 `mtp.*` tensor 生成的 MTP runtime
- 上游 Strata v0.2.9 中的 `data/draft_vocab.bin`

一条命令完成下载、校验和生成：

```bash
bash scripts/k100ai/fetch-qwen-iq3xxs.sh \
  /data/k100ai-assets/qwen3.8-flash-next-iq3-xxs
```

该流程使用：

- ModelScope repository：
  `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`
- snapshot commit：
  `1868010e27505d4b8cc15970cdcc4dd76005d7de`
- MTP checkpoint revision：
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`

两个模型 shard 的 SHA-256 分别为：

```text
219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15
316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113
```

已经实测可重建的 pack 和 runtime 校验：

| 文件 | SHA-256 |
| --- | --- |
| `pack/iq3xxs/experts.bin` | `e76956703f0cb960fa959cc0c33b0e38284ee98c8eb360417c74e7238c7bf005` |
| `pack/iq3xxs/dense.bin` | `d2af3d8331e5c7a1985d1d589747230f9d56719a6d33c7cddc9ee6b81cde6b13` |
| `pack/iq3xxs/index.txt` | `1afe12b36cd2d13bf122b741fa0e81e1f751319cc3c451e54ba75524d825f148` |
| `pack/iq3xxs/native_experts.txt` | `ae28cf7a2a1c9063c4b55d6c8cf86d4d079dcddcf597f9e8697c9157d462c0ee` |
| `pack/iq3xxs/conversions.json` | `68beec784ea64aeecace591b33debb8ba331a9ad257a3e1b68c5c701dd22b267` |
| `pack/iq3xxs/experts.bin.src.json` | `30bf3328b49b4b3444dbe29bdae9b1fe65e86d33fda10fffbdb70bb9a5316644` |
| `mtp/rt/experts.bin` | `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f` |
| `mtp/rt/dense.bin` | `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a` |
| `mtp/rt/dense.txt` | `991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd` |
| `mtp/rt/draft_vocab.bin` | `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91` |

本清单中的 42.9 GB pack 和 787 MiB MTP runtime 已从源输入重新生成并逐文件
核对，不只是记录已有值的下载清单。

启动示例：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
K100AI_QWEN_NATIVE=/data/k100ai-assets/qwen3.8-flash-next-iq3-xxs/model/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
K100AI_QWEN_PLE=/data/k100ai-assets/qwen3.8-flash-next-iq3-xxs/model/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
K100AI_QWEN_PACK=/data/k100ai-assets/qwen3.8-flash-next-iq3-xxs/pack/iq3xxs \
K100AI_QWEN_MTP_RT=/data/k100ai-assets/qwen3.8-flash-next-iq3-xxs/mtp/rt \
PORT=18096 bash scripts/k100ai/serve-k100ai.sh qwen
```

## DeepSeek 资产

运行前准备好五个 backbone shard、一个 Engram 文件、DSpark drafter，并用固定
recipe 生成 `maya-ds4-dspark` pack：

```bash
bash scripts/k100ai/fetch-dsv41-iq2xxs.sh /data/models/dsv41-2.00
bash scripts/k100ai/prepare-dsv41-pack.sh \
  /data/models/dsv41-2.00 \
  /data/models/dsv41-2.00/maya-ds4-dspark
```

DeepSeek 权重来自：

- `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`
- DSpark：`unsloth/DeepSeek-V4.1-Flash-1017-GGUF`

五个 backbone shard、Engram 和 DSpark 都按 ModelScope 返回的逐文件提交
revision 固定下载地址，不跟随可变的 `master`。下载脚本随后再校验固定
大小与 SHA-256。DSpark 当前文件名为
`dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf`，固定 revision 为
`6cd2eacaad3f5d9b61e6913c188a934a3a5eac22`。

已实测可重建的 pack 校验：

| 文件 | SHA-256 |
| --- | --- |
| `ds4.txt` | `87e04d460f8f9ab82a14cb3793fe0c48f85348d81da5f2e93d200bc5dbc2d6f4` |
| `tensors.txt` | `f91733dc81ff262a0e6f65c0c5c3b86aa1390cfcd144bba75858357d62a50284` |
| `engram.bin` | `08d55d6433f389b14aa115f4ea4eb31d6a2d63df1ae40821f03facf4c2d49142` |
| `tokenizer/chat_template.jinja` | `4d61d22c90ea0cdf15c7e298f7215f23eeffd20f8eed8fb517fda5fb64693d99` |

运行仍使用包内冻结的 `data/dsv41/profile-2.00.bin` 和
`--profile-readonly`。profile 校验：

```text
8db93e483b34b15dbf63a0826601f1db2107a8cfea73774e7d21d8559e0a9568
```

## 从源码重建

解压 source bundle 后，目录结构为：

```text
strata-v0.2.9-k100ai-dual-source/
  main/
  qwen-fast-0.1.47/
  llama.cpp/
  BUILD_INPUTS.json
```

快速 Qwen 0.1.47：

```bash
bash main/scripts/k100ai/build-qwen-0.1.47.sh \
  qwen-fast-0.1.47 \
  llama.cpp \
  /data/build-qwen-0.1.47
```

脚本固定容器路径为 `/work/strata`、`/work/llama` 和 `/work/build`，并拒绝
接受不同于 `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9`
的产物。该哈希已经从干净源码重建并实测一致。

当前 Qwen 0.1.55 和 DeepSeek：

```bash
bash main/scripts/k100ai/build-current.sh \
  main \
  llama.cpp \
  /data/build-current
```

构建产物位于 `qwen/strata`、`dsv41/strata-dsv41` 和
`dsv41/ds4_gguf_test`。发布校验要求它们分别匹配运行时包内的 SHA-256。
脚本使用固定容器路径 `/work/strata`、`/work/llama`、`/work/out`；
更换容器内路径会改变编译器写入二进制的源路径，即使代码相同也可能改变
产物哈希。

源码归档中的 `BUILD_INPUTS.json` 还会记录三棵源码树的文件数、字节数和
内容 manifest SHA-256。源码归档排除 `.git`、构建目录、模型权重、
`data/experts.bin`、`data/dense.bin`、`data/dense.txt` 和
`data/draft_vocab*.bin`。

## Clean-room 验收

在独立目录、独立 Docker 卷和无构建缓存的机器上执行：

```bash
sha256sum -c SHA256SUMS
bash scripts/k100ai/setup-runtime.sh
bash scripts/k100ai/preflight.sh
BASE_URL=http://127.0.0.1:18096 MODEL=qwen3.8-flash-next \
  EXPECT_DRAFT_METRICS=1 OUT=api-qwen bash scripts/k100ai/validate-api.sh
BASE_URL=http://127.0.0.1:18097 MODEL=deepseek-v4.1-flash \
  OUT=api-dsv41 bash scripts/k100ai/validate-api.sh
```

验收必须同时确认：

- 四个二进制哈希一致
- Qwen `max_context=262144`、`engine=0.1.47`、`drafting=mtp`
- DeepSeek `max_context=1048576`、FP4 KV、`drafting=dspark`
- DeepSeek profile 运行前后不变
- 服务退出后没有测试容器或 HCU0-3 lease

## 明确边界

- 当前只承诺 gfx928 + DTK 2604。相同源码在其他架构或工具链上的产物和
  性能不在本次发布保证范围内。
- 完整 262144-token Qwen 和完整 1,048,576-token DeepSeek 窗口没有灌满
  语义质量测试。
- Qwen 默认 0.1.47 经用户接受存在数值漂移，0.1.55 的 prompt 性能回归
  仍未修复。
- DeepSeek 模型仓库没有提供单一、可解析的 snapshot revision；因此本指南
  固定每个实际下载文件的 ModelScope revision，而不是声称仓库有一个
  固定 snapshot。
- 外部模型文件受各自许可证约束，发布归档不分发这些文件。
