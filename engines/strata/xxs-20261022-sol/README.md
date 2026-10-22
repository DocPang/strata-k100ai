# xxs-20261022-sol — rocBLAS solution 锁定版

引擎 SHA-256 `19b367e015d006694737ba067f0a4ac25da09c40be19b0d669fc1e09444152a3`（29,327,416 B）。在 rc.6 定版 `43002a72…` 的行为基础上，增加"按形状指定 rocBLAS Tensile solution"的能力，门控变量 `STRATA_GEMM_SOL`。

本目录**不分发二进制或运行包**，只记录引擎身份、环境、补丁与实测。重建路径见 [docs/SOURCE_BUILD.md](../../../docs/SOURCE_BUILD.md)，叠加本目录 `patches/` 的三个脚本。

## 这一版做了什么

对 `Gemm::f16_inplace` 与 `Gemm::native` 的 padr 分支各插入一处查表 hook：当 `STRATA_GEMM_SOL` 命中 `N×K` 时，把 `rocblas_gemm_ex` 的 solution 索引固定到实测最优值；未命中或未设置变量时完全走原路径。

```bash
STRATA_GEMM_SOL=2560x6144=20377,10240x2560=20385
```

- `20377`：`N=2560, K=6144`（引擎内略优于 20376，−0.10%）
- `20385`：`N=10240, K=2560`（该调用点 −27.1%）
- 编号来自 `rocblas_gemm_ex_get_solutions` 枚举的 512 条候选；引擎真实调用参数为 `lda=6144 ldb=6144 **ldc=2N**`

## 补丁顺序（必须按此顺序，全部 fail-closed）

| 顺序 | 文件 | 目标 | 锚点 | 作用 |
| --- | --- | --- | --- | --- |
| 1 | `patches/patch_gemsol.py` | `src/prefill/gemm.cu` | 2 | 插入 solution 查表 hook（两处） |
| 2 | `patches/patch_gemsol2.py` | `src/prefill/gemm.cu` | 1 | 补 `10240x2560` 的第二处 hook |
| 3 | `patches/patch_cmake_rocblas.py` | `CMakeLists.txt` | 1 | 显式 `-lrocblas` |

锚点未全部匹配则拒绝写入（`ANCHOR_FAILED`）。对已打补丁的文件重复应用会因锚点计数为 0 而失败——这是设计行为。

## 实测（详见 [docs/BENCHMARKS.md](../../../docs/BENCHMARKS.md)）

| 场景 | 控制 | 本版 | Δ |
| --- | --- | --- | --- |
| ctx128 交叉 A/B（126,857 token，卡4） | 61,271.7 ms | 60,265.7 ms | **−1.64%** |
| 生产 23k token（hcu0/hcu1） | — | — | 组合后 −11.2%（含源码增量） |

精度：KL 8.3e−05，top-1 一致（`a=896 b=896 SAME`），maxabsdiff 1.74572，n=248,320。取消 `STRATA_GEMM_SOL` 即回到位精确路径。

## 边界（不要外推）

- 本版相对 `sources/xxs-20261016-gold` 共有 **7 个文件**的增量，本仓库只归档了其中 2 个（`gemm.cu`、`CMakeLists.txt`）。另外 5 个（`gemm.hpp`、`dequant_bf16.cu`、`gateup_exact.hpp`、`kernels.cu`、`moe_mmq.cu`）来自 2026-10-20~22 的实验构建链，**逐项补丁来源未归档**。
- 因此 `clean_rebuild_verified = false`，且**不能**宣称可从公开材料重建出同一二进制。
- **不要**同时开启 `STRATA_DENSE_TN=1`：`Gemm::native` 在 TN 分支 `return native_tn(...)` 提前返回，会绕过两处 hook；实测两开关合并只有 −1.24%，比任一单独都差。
- `STRATA_GEMM_SOL` 命中失败是静默的（走默认 solution），验证命中需用 `STRATA_GEMM_ATTR=1` 或 stride 打印。
