# K100AI 来源与校验

[返回首页](../../README.md) · [文档索引](README.md) · [安装指南](INSTALL.md)

适用版本：**v0.3.0**。

## 生产二进制

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `bin/strata-0.1.47` | 45,674,152 B | `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9` |
| `bin/strata-0.1.55` | 49,496,856 B | `25d4ea83b883443d0a8bc68686c72f613cd0923a2897c2f7e9063939025e904e` |
| `bin/strata-dsv41` | 23,941,864 B | `4fd967c1fa6769e60298357d55683c18e5522f6689441b12fe5a97489b3a35d6` |

Qwen 两个二进制沿用 v0.2.3 已验证版本。DeepSeek 二进制在 R178 完成增量构建
与 clean rebuild 同哈希验证，随后通过 R179 四卡 A/B 和生产 postinstall。

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

Qwen 源码、配置、测试和打包脚本来自 v0.2.3 发布树。两个二进制保留原始
SHA-256，便于确认它们与 v0.2.3 验证结果对应。

## 构建环境

```text
平台      linux-x86_64-gfx928-dtk2604
镜像      harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
架构      gfx928
```

发布机没有重新编译出与 DeepSeek 二进制相同哈希，因此来源以生产 clean
build、逐文件源码对账和生产 postinstall 为证据。
