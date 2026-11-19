# K100AI 来源与校验

[返回首页](../../README.md) · [文档索引](README.md) · [安装指南](INSTALL.md)

适用版本：**v0.3.1**。

## 生产二进制

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `bin/strata-qwen-iq3s-r112` | 53,884,424 B | `102b4b412068a68588883c13a28103600d8b078785da9dcbbda34c440e9a5074` |
| `bin/strata-0.1.47` | 45,674,152 B | `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9` |
| `bin/strata-0.1.55` | 49,496,856 B | `25d4ea83b883443d0a8bc68686c72f613cd0923a2897c2f7e9063939025e904e` |
| `bin/strata-dsv41` | 23,941,864 B | `4fd967c1fa6769e60298357d55683c18e5522f6689441b12fe5a97489b3a35d6` |

v0.3.1 的 Qwen 生产二进制是 `strata-qwen-iq3s-r112`。0.1.47 和
0.1.55 两个旧二进制沿用 v0.2.3 已验证版本，只用于兼容入口和 A-B。
DeepSeek 二进制在 R178 完成增量构建与 clean rebuild 同哈希验证，随后
通过 R179 四卡 A/B 和生产 postinstall。

## DeepSeek 源码

DeepSeek 发布源码取自：

```text
/data/dsv41-opt-v022-20261104/src
```

| 文件 | SHA-256 |
| --- | --- |
| `src/dsv41/ds4_model.cu` | `fa4644560c1fd3bd7e1aa259791b817a001a9d7ee48b1206ecd5cb228b0a548a` |
| `src/dsv41/ds4_kernels.cu` | `3734325b18913f6a533965cd8b274ff298f15ec676eeb7e07d791c9a4e3daa5a` |
| `src/dsv41/ds4_iq.cu` | `052524a03e65aa0139fe9e7b69bbbaf81c91a3ea78a6ab30f4aabc9b133a5bf8` |
| `src/prefill/wmma_gemm_mmac928.cuh` | `466cb00ec187199fc15970783ddedbfffad299a29e63358414e30ab3c0577458` |
| `src/dsv41/vendor/exl3.cu` | `257a97a4e7fe0d5c37a167fc7a8c4f463db8b381d742b9b8bf0ce69c88aa6e2c` |
| `include/strata/dsv41/exl3.hpp` | `4979877507a7a0ac73ca1b1d5f1e549fdf0fb4cbfaf5a1c3cdf365181a47a60b` |

生产源的 `src/`、`include/`、`cmake/`、`third_party/`、`tools/`、
`data/`、`docker/`、`serve/`、`scripts/` 和 `tests/` 已完成逐文件内容对账。

## Qwen 来源

两个旧 Qwen 二进制的源码、配置、测试和打包脚本来自 v0.2.3 发布树，
保留原始 SHA-256。

v0.3.1 的 IQ3_S 引擎来源为 K100AI Qwen 研究树 HEAD
`ca83b39d212c35bb03eba1960578f273294f3e68`。R112 只替换 IQ3_S/IQ4_NL
专家的 mode 8 派发，基础对象与锚定 R54 构建固定。关键源码 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `src/kernels/cuda/iq_kernels.cu` | `031918c19e542a1687888eb769d677a3c760f4c7e69a291032448ad8dffc6ee1` |
| `src/kernels/cuda/qsa_select.cu` | `7a9156c96ce24b69e0131304362b55d3ba7642d083a0bac93225bebfbec8f542` |
| `src/prefill/prefill.cpp` | `2188db7c673085069fec066ecf7b90f9d67aa2c2163f6f642c11c6ab5471701b` |

完整快照、构建命令和精确性说明随 Release
`strata-qwen-iq3s-r112-source.tar.gz` 提供。

## Qwen IQ3_S 运行资产

| 资产 | 大小 | SHA-256 |
| --- | ---: | --- |
| IQ3_S 模型分片 1 | 54,817,524,224 B | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| IQ3_S 模型分片 2 | 28,800,138,432 B | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |
| `pack/iq3s/experts.bin` | 50,292,326,400 B | `ed9d0a872b4d9fb992a05f3f5114107b0689767f0d95e2fb7e2ffe091000e67c` |
| `pack/iq3s/dense.bin` | 1,538,035,200 B | `d2af3d8331e5c7a1985d1d589747230f9d56719a6d33c7cddc9ee6b81cde6b13` |
| `pack/iq3s/index.txt` | 93,685 B | `b2a014878c3b61d6a6f915e194055a5e68ede34f9465dd1cb13249d902f008ac` |
| `pack/iq3s/native_experts.txt` | 3,304 B | `d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d` |
| `mtp/rt/experts.bin` | 707,788,800 B | `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f` |
| `mtp/rt/dense.bin` | 116,099,072 B | `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a` |
| `mtp/rt/dense.txt` | 1,851 B | `991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd` |
| `mtp/rt/draft_vocab.bin` | 425,196 B | `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91` |
| 发布包 `data/k100ai-qwen-iq3s-draft-vocab-zh60k.bin` | 238,428 B | `e93f8cd23075d98ccdb4bf10d10c69613a25ec449afc062d0c9dbd33dfb96122` |
| `data/k100ai-qwen-iq3s-rocblas.txt` | 1,593 B | `2552ded20306907e3ed82df5a4dd6b977916ea3652895097389014b846f85490` |

模型来自 [ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO GGUF](https://www.modelscope.cn/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)，
revision 为 `6ac9406afe74db316b11a3865a3382c7bedbcf34`。
`scripts/k100ai/fetch-qwen-iq3s.sh` 会下载、恢复并校验这些外置资产。
pack 工具使用 llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`
的 `gguf-py`。发布包直接携带该目录；缺失时脚本会校验固定 ZIP 的
SHA-256 `cbe23c594282ead2937abb3f008e51fcec4609d9256652fff42c7cc1c21ea47b`
后恢复。
简体中文 60K draft vocabulary 固定自
[Strata-rdna3 `7df6538d`](https://github.com/nicejunjie/Strata-rdna3/blob/7df6538d6c886093df39d74b36ed0a3396bf9467/rdna3/draft_vocab_zh60k.bin)。

## 构建环境

```text
平台      linux-x86_64-gfx928-dtk2604
镜像      harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
架构      gfx928
```

发布机没有重新编译出与 DeepSeek 二进制相同哈希，因此来源以生产 clean
build、逐文件源码对账和生产 postinstall 为证据。
