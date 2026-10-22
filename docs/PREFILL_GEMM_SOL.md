# 单卡 prefill 的 rocBLAS solution 锁定（2026-10-22）

针对 `Qwen3.8-Flash-Next` IQ3_XXS 单卡 prefill（ctx128 = 126,857 token，chunk 30602）的一次定向优化与完整归因记录。引擎见 [engines/strata/xxs-20261022-sol](../../engines/strata/xxs-20261022-sol/)。

## 结论

| 项 | 结果 |
| --- | --- |
| 立项目标（端到端 ≥20%） | **未达成** |
| 本次可交付 | solution 锁定 **−1.64%**（位精确基线之上） |
| 生产实测（含未发布源码增量） | 23k token −11.2%，126k token ≈ −8.9% |
| 精度 | KL 8.3e−05，top-1 一致；取消变量即回位精确 |

## 方法：三件必须先立好的事

### 1. 归因探针（环境无 rocprof/omniperf）

| 开关 | 作用 |
| --- | --- |
| `STRATA_PREFILL_TIMING=1` | 分阶段耗时（gate/up、qsa attn、gdn、hc read…） |
| `STRATA_GEMM_ATTR=1` | 每次 GEMM 的 (N, K, ms, calls)，把阶段耗时拆到具体形状 |

`STRATA_GEMM_ATTR` 输出：2944 次调用合计 21,244 ms，四个主导形状占 **89.7%**（2560×6144 32.0%、10240×2560 30.4%、6144×2560 16.6%、12288×2560 10.6%），dequant 仅 **1.8%**。

### 2. 交叉 A/B（`c1 / n1 / c2 / n2` 同窗交替）

服务端 `prefill_ms` 为权威口径。判据：`|c1−c2|/mean < 0.5%`。本轮控制组两次相差 **1.6 ms（0.003%）**，实验组相差 0.14%，Δ = **−1.64%**。

### 3. 承认两条硬约束

- **独立微基准不是有效仪器**：同一 `rocblas_gemm_ex`，独立 bench 最好 72.07 TF/s，引擎内实测 **95.6 TF/s**，比任何独立配置都快。
- **只有"同 kernel 换 Tensile solution"可迁移**；换 tile / 换 kernel / 换路径不可迁移。9 个形状的 solution 扫描里只有 2 个迁移成功。

## 结果台账

| 配置 | Δ |
| --- | --- |
| `STRATA_GEMM_SOL=2560x6144=20377,10240x2560=20385` | **−1.64%** |
| `STRATA_DENSE_TN=1`（单独） | −1.35% |
| 两者合并 | −1.24%，**劣于单用，禁止合并** |

引擎真实调用参数：`N=2560 K=6144 lda=6144 ldb=6144 **ldc=5120 (=2N)**`。所有用 `ldc=N` 做的扫描全部作废——这一条曾导致一整轮"微基准大赢、引擎零收益"。

## 已证伪的路线（不要再走）

| 方向 | 结果 |
| --- | --- |
| 小形状 solution pin（640/512） | 引擎内 **+0.01%**（未命中，引擎已是 bench-best） |
| `rocblas-gemm-tune` 离线调优 | YAML 只打 stdout、需 `Tests:` 包装、15+ 分钟/形状，工程不可行 |
| 合并小 GEMM | N 扫描 51.6→69.7 TF/s 无趋势，N=14336 反而更慢 |
| 缓存 dequant 结果 | dequant 仅 1.8%，收益上限≈0 |
| 自研 int8 dense GEMM（1/2/3/4 阶段流水、NT4） | 最好 **49.25 TF/s**，比 rocBLAS f16 的 95.6 慢 2 倍；四阶段超 LDS 上限 65,536 B |
| gate/up：`KS128` / `MT2` / `SV4` / `LB3` / `WSG32_NT4` | 全部更差（`LB3` 触发 spill，+111%） |
| hc read：`HC_SPLIT=0` + 全部 pin / `HC_MMAC` | 0% / +0.1%：upmix 已走 hipBLASLt bf16 默认路径 |
| `PF_PAD+PF_GEMM` / `PADR_LT` / `PAD_ROCBLAS=2` | −0.16% / +0.19% / +0.53%，均落噪声 |

硬件侧已实测：纯寄存器 MMAC 可达 **330.07 TFLOP/s = 99.8% int8 峰值**（条件：≥4 waves/SIMD、≥8 累加器）。**瓶颈不在硬件**，在于业务形状上 rocBLAS f16 只有 bf16 峰值的 36–42%，而引擎已在这条路径的最优点。要把 int8 吃满需要 Tensile 级内核自动生成，属另一个立项。

## 阶段预算（ctx128，总 ≈60.2–61.3 s）

| 阶段 | 占比 |
| --- | --- |
| dense GEMM 池 | 27% |
| gate/up | 17.5% |
| qsa attn | 12.8% |
| gdn recurrence | 10.9% |
| hc read | 9.1% |
| qsa select | 7.7% |

## 生产部署中发现的问题（重要）

上线时用同一实例、同一 prompt 做了三级阶梯（23k token，TTFT）：

| 二进制 | ms | vs 原生产 |
| --- | --- | --- |
| `43002a72`（rc.6 定版，原生产） | 13.270 | — |
| `43002a72` **重启后复测** | 13.270 | **0%（排除重启效应）** |
| `1502d407`（重编基线） | 12.055 | −9.15% |
| `19b367e0`（本版 + pin） | 11.785 | −11.19% |

⇒ **生产上跑了一个落后约 9% 的二进制**。原因不是构建配置：`prepared-gq8` 的 `CMakeCache.txt` 与 [engines/strata/c4/replay/build-in-container.sh](../../engines/strata/c4/replay/build-in-container.sh) 的编译选项逐项一致（`Release`/`-O3`、`--gpu-max-threads-per-block=1024`、`-DSTRATA_GFX928_DOT4=1`、两个 `-mllvm` 开关、`gfx928`）。真实原因是**相对 `sources/xxs-20261016-gold` 存在 7 个文件的未归档源码增量**，其中 5 个（`gemm.hpp`、`dequant_bf16.cu`、`gateup_exact.hpp`、`kernels.cu`、`moe_mmq.cu`）的逐项补丁来源未归档。清单与 SHA-256 记在 [引擎 manifest](../../engines/strata/xxs-20261022-sol/manifest.json) 的 `source_delta_vs_gold`。

**待办**：把这 5 个文件的增量拆回逐项补丁，才能宣称可从公开材料重建。

## 复现口径

```
suite ctx128，input 126,857 token，chunk 30,602
STRATA_EXPERT_CACHE=24576 STRATA_AB_MAX_CONTEXT=262144 STRATA_AB_SPEC=6 STRATA_AB_SKIP_RETRIEVAL=1
BASE = 生产 env（见 engines/strata/xxs-20261022-sol/environment.env）
实验臂 = BASE + STRATA_GEMM_SOL=2560x6144=20377,10240x2560=20385
```
