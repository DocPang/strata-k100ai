# Provenance

## Strata 源码

- Upstream：`https://github.com/Niko1221/Strata.git`
- v0.2.9 base：`c62d5e3a15b40ea51c96aae9e8d36928965f53ab`
- Current engine source：`8544d2fa6cddf61fb47edc0c3bd8a5d7d638140e`
- Qwen 0.1.47 fast source：`2150b1acdb48c386a9afcd64e63e2369b5a5e108`
- Branch：`k100ai/dsv41-v029`
- Product version：`0.2.9`
- Current engine version：`0.1.55`
- Qwen default fast engine version：`0.1.47`

当前引擎构建使用 `8544d2fa` 的干净源码树。该提交新增
`src/prefill/k100ai_exact.cuh`，只对经过逐格式验证的 gate/up 与 down
格式启用 gfx928 精确 MMAC，并保留原有升序 K、FP32 累加语义。

运行和打包修复位于后续提交，最终包中的 `BUILD.json` 用
`package_source_commit` 记录该树的准确提交。运行时、launcher、文档和
metadata 改动没有伪装成新的引擎源码。Qwen 默认快速档来自更早的
K100AI prompt 优化线；它不声称能从本包内 v0.2.9 源码直接重建。

## 二进制

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `bin/strata`（Qwen 默认快速档） | 45,674,152 B | `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9` |
| `bin/strata-0.1.55`（Qwen 当前对照档） | 49,484,096 B | `437308792b7ec08724aa90e4b11736b4f40f41c10613faa9688fbdc1723ba7ab` |
| `bin/strata-dsv41` | 22,576,544 B | `6543311f453bee2c6d61653a24f0b6282f9cb3568b17a8b889c2c5e073b12286` |
| `bin/ds4_gguf_test` | 18,816,624 B | `0d6c8455d76815b810a38c5529bc868087e71641008e0c86843ecd8b34dd815c` |

0.1.55 的 Qwen 与 DeepSeek 二进制来自同一 DTK 2604/gfx928 构建。
0.1.47 快速档是已有 K100AI 0.1.47 构建，也在同一 gfx928/DTK 2604 运行时
上重新验证，并已从保留的 0.1.47 源码和固定 llama.cpp 快照按 canonical
容器路径重建出相同 SHA-256。当前源码构建设置：

```text
STRATA_HIP_GFX906=ON
STRATA_ENABLE_HIP=OFF
STRATA_ENABLE_CUDA=ON
STRATA_PREFILL_MMQ=ON
STRATA_MMQ_KQUANTS=ON
STRATA_Q6K_EXPERTS=ON
STRATA_NATIVE_EXPERTS=ON
CMAKE_HIP_ARCHITECTURES=gfx928
STRATA_GFX928_DOT4=1
CMAKE_BUILD_TYPE=Release
```

这里的 `STRATA_ENABLE_CUDA=ON` 是项目源码分支和 DTK/HIP 编译映射，不能
解释为已用 NVIDIA `nvcc` 完成原生 CUDA 编译。原生 CUDA 工具链仍是本次
发布的未验证项。

构建 recipe 需要：

- 固定 digest 的 DTK 2604 镜像
- `/opt/hyhal:/opt/hyhal:ro`
- `ROCM_PATH=/opt/dtk`、`HIP_PATH=/opt/dtk`、`HIP_PLATFORM=amd`
- 构建期将 Strata 源码、llama.cpp 快照和输出分别挂到
  `/work/strata`、`/work/llama`、`/work/build` 或 `/work/out`

`STRATA_GGML_DIR` 只在构建时读取；发布脚本、配置、runtime 和
`BUILD.json` 不保留 `/data` build/research 绝对路径。

## Qwen3.8-Flash-Next

实验模型目录：

```text
/data/experimental-models/qwen38-flash-next-gsq-rco-iq3-xxs-20261015
```

模型文件清单 SHA-256：
`191f27853d60bfaa764ec2afc23b9feb2d6f46eeabb6d6232f94a280e471c6f9`。

Pack：

```text
/data/experiments/strata_iq3xxs_gfx928_20261015/packs/iq3xxs
```

| 文件 | SHA-256 / 说明 |
| --- | --- |
| `experts.bin` | 42,912,972,800 B，由两个已校验 IQ3_XXS shard 生成 |
| `dense.bin` | 1,538,035,200 B |
| `conversions.json` | `68beec784ea64aeecace591b33debb8ba331a9ad257a3e1b68c5c701dd22b267` |
| `index.txt` | `1afe12b36cd2d13bf122b741fa0e81e1f751319cc3c451e54ba75524d825f148` |
| `native_experts.txt` | `ae28cf7a2a1c9063c4b55d6c8cf86d4d079dcddcf597f9e8697c9157d462c0ee` |

Expert profile：

```text
/data/experiments/strata_iq3xxs_gfx928_20261015/strata/data/expert-profile.bin
SHA-256 8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf
```

外部 MTP runtime：

```text
/data/experiments/strata_iq3xxs_gfx928_20261015/mtp-local-rt-r1
```

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `experts.bin` | 707,788,800 B | `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f` |
| `dense.bin` | 116,099,072 B | `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a` |
| `dense.txt` | 1,851 B | `991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd` |
| `draft_vocab.bin` | 425,196 B | `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91` |

该 runtime 约 787 MiB，必须放在发布包外部。发布归档只保存路径、校验值和
加载流程，不包含模型权重或 draft 文件。

Tokenizer 文件的既有校验记录：

| 文件 | SHA-256 |
| --- | --- |
| `chat_template.jinja` | `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce` |
| `merges.txt` | `4ba698529766c7a2f5052083ea10cdf2485577f5237a332ee291054eb11e3332` |
| `token_type.json` | `5088c8c298fc06af8382ddb3b76c888703ac2634e82263b97eedcc2ad202738b` |
| `tokenizer.json` | `c0bd435efbc6c3b28964b47999afe16c4ad268c7c7c4e8cb3af410796a6f30bf` |
| `vocab.json` | `4ba64f0332abcfb0b600b7df1537d1e836e68009d5fb7cdb77339738dc6365c4` |

## DeepSeek V4.1

- ModelScope collection：
  `https://www.modelscope.cn/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`
- 下载 revision：五个 backbone shard 和 Engram 按 ModelScope 返回的逐文件
  revision 固定，清单记录在主源码
  `configs/k100ai/release-inputs.json`
- 模型目录：`/data/models/dsv41-2.00`
- 量化：backbone IQ2_XXS；Engram IQ4_NL
- 活动 pack：`/data/models/dsv41-2.00/maya-ds4-dspark`

模型 `SHA256SUMS` 文件：
`ba0bf88586c9959c4cc4ee0d7d300408d621b146c9ab7e4e9ad1499a44d0cbf6`。
已对 5 个 backbone shard 和 Engram 执行 `sha256sum -c SHA256SUMS`。

冻结 placement profile：

```text
data/dsv41/profile-2.00.bin
SHA-256 8db93e483b34b15dbf63a0826601f1db2107a8cfea73774e7d21d8559e0a9568
```

服务必须同时使用 `--profile-readonly`。包内 profile 与活动 pack 内可变的
`expert-profile.bin` 不是同一资产。

DSpark drafter：

```text
/data/models/dsv41-2.00/dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
```

- Size：7,976,914,240 B
- SHA-256：
  `6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a`
- ModelScope revision：
  `6cd2eacaad3f5d9b61e6913c188a934a3a5eac22`

它是 DSpark 推理输入，不替代 backbone 的完整 6-expert 主路径。

## Qwen 快速档来源

K100AI 0.1.47 旧服务二进制：

```text
/data/strata-serving-next02-hcu4567-20261023/bin/strata
SHA-256 bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9
```

发布包将该二进制以 `bin/strata` 名称作为 Qwen 默认快速档，不另设
0.1.47 命名的第二份二进制；其身份由 `BUILD.json` 的
`qwen_engines.default_fast` 和上表 SHA-256 共同记录。重新验证包括
8,190/40,057-token prompt、MTP draft layer、greedy 输出、40K 输出前缀和
first-logits 漂移。它不是 bitwise oracle；0.1.55 canonical 对照仍在
`bin/strata-0.1.55`。

## 禁止输入

- 已明确标记为 stale 的 partial four-shard 目录
- 未通过上述 SHA-256 清单的 partial shard
- 把旧 0.1.47 当作 bitwise oracle

## 主机和设备边界

- 主机：`hygon-compute`
- 只使用 HCU0-3 / `renderD128-131`
- HCU4-7 的研究服务不停止、不重启、不重配，也不参与本版本测试
- 保留并保持 `exited`：
  - `strata-next-hcu0-xxs-20261015`
  - `strata-next-hcu1-xxs-20261016`
  - `geak-k100ai-hcu23-20261030`
