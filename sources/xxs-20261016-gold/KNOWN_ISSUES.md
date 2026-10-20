# 已知问题（本快照 = 生产二进制 `43002a72…` 的源码，未修改）

本文件记录对这批源码的静态审查结论。**7 个源码文件保持与生产二进制对应的原始状态，不在此快照内打补丁**——快照一旦改动就无法再自证与 `43002a72` 的对应关系。修复走下一个版本目录。

严重度按"能否被当前发布配置触发"划分，不按代码缺陷本身的观感。

---

## P1-1 `STRATA_GEMM_F32OUT` 与 `STRATA_GEMM_Y_F16` 缺少互斥（`src/prefill/gemm.cu`）

| 项 | 内容 |
| --- | --- |
| 位置 | padr 分支 `gemm.cu:1246-1250`；对照 `gemm.cu:1019`（`f16_inplace`） |
| 现状 | `f16_inplace` 是 `if (f32out && !y_f16)`；padr 分支只有 `if (f32out)`，漏了 `&& !y_f16` |
| 触发条件 | `STRATA_GEMM_F32OUT=1` **且** `STRATA_GEMM_Y_F16` 覆盖到同一调用点 |
| 后果 | padr 路径按 `CUBLAS_R_32F` + `ldc = ldy` 写出 FP32，而消费者按 `__half2float(((const __half*)Y)[i])` 读 → 数值错误，且该分支不调用 `widen_rows_f16`，不会自愈 |
| 可达性 | **当前不可达**：发布配置未设 `F32OUT`；且该项已被永久否决（相位 39,704 ms 对 16,418 ms、gdn 2,008→12,475 ms、token sha 不一致） |
| 建议修复 | `if (f32out && !y_f16)`，与 `f16_inplace` 对齐；另加组合开关的构建期断言（两者同时开启直接 abort），避免以后有人误开 |

## P1-2 公开材料不足以重建 rc.6（文档级，已修正）

| 项 | 内容 |
| --- | --- |
| 位置 | `docs/SOURCE_BUILD.md` |
| 现状 | 原文让读者"c4 补丁 + rc.5 补丁 + 4 个 FP16 脚本"重建，**这是错的** |
| 证据 | `engines/strata/xxs-20261016/patches/c4-to-xxs-20261016.patch` 中 `TOPK_H16`、`TOPK_REG_MAX`、`Q6_WAVE_PACK_LDS` 出现次数均为 **0**；这些实现只存在于本目录公开的 `qsa_select.cu` / `native_mmvq.cu` 等文件里 |
| 后果 | 按旧文档操作会得到一个"看着能编译、但缺 rc.6 三个开关"的引擎，且不会报错 |
| 修正 | 重建必须**按路径覆盖本目录的 7 个文件**并逐文件校验 SHA-256；4 个 `patch_yf16_*.py` 只作为历史成因，不作为重建手段。`docs/SOURCE_BUILD.md` 已按此改写 |
| 附带 | `patches/patch_yf16_*.py` 的路径契约是 `<workdir>/source/{include,src}/...`，而恢复文档用的是 `prepared/strata/...`；调用时须显式传入满足 `source/` 子目录的 workdir |

## P2-3 `kernel_hcdown_mmac` 的 LDS 越界写（未启用实验路径）

| 项 | 内容 |
| --- | --- |
| 位置 | `src/prefill/gemm.cu:831`（核）、`gemm.cu:860-864`（`store`） |
| 现状 | `__shared__ uint16_t sA[2][HCD_BM][HCD_LDK]`，`HCD_BM = 16*4 = 64` 行；`store` 写 `sr`、`sr+32`、`sr+64`、`sr+96`，`sr = tid>>3 ∈ [0,31]` ⇒ 最大行号 **127**，越过 64 行维 |
| 后果 | 对 `b=1` 越界到数组之外（`sA`/`sB` 相邻或越界），属未定义行为；注释自称 "128 rows x 64 k"，与 64 行的声明不符 |
| 可达性 | **当前不可达**：入口 `gemm.cu:939` 要求 `(nd+ni)%16==0`，而调用点 `prefill.cpp:2696` 传入 `LR=320, HC=4` ⇒ 324，`324%16=4` ⇒ 直接 `return false` |
| 附带缺陷 | 开关 `STRATA_HC_MMAC` 在 `prefill.cpp:2692` 用 `getenv(...) != nullptr` 解析 ⇒ **设成字符串 `0` 也算开启**，与项目其它门控的 `v[0]=='1'` 约定不一致 |
| 建议 | 保持禁用。若要启用，需同时修：LDS 行数（64→128 或把 staging 减半）、`store` 的行映射、以及开关解析改为 `v[0]=='1'` |

## P2-4 工作区仍按 FP32 容量分配（`src/prefill/prefill.cpp`）

`GEMM_Y_F16` 只改了**写出格式**，没有改**分配容量**：

| 缓冲 | 分配语句 | 说明 |
| --- | --- | --- |
| `m.qkv` | `prefill.cpp:1294` `a.take<float>(T * C)` | 写紧凑 FP16，仍按 FP32 分配 |
| `m.Qf` | `prefill.cpp:1299` `b.take<float>(T * 12288)` | 同上 |
| `m.z` | `prefill.cpp:1294` `a.take<float>(T * ZV)` | 同上 |
| `m.shared` | `prefill.cpp:1331` `c.take<float>(T * N)` | 同上 |

- 这是**容量浪费，不是正确性问题**（多分配不写入，结果不受影响）。
- 30,602 token 的 chunk 下，单个缓冲理论减半：`Qf ≈ 0.70 GiB`、`qkv ≈ 0.58 GiB`、`z ≈ 0.35 GiB`、`shared ≈ 0.15 GiB`。
- 这些缓冲分属 `a`/`b`/`c` 三个 arena，存在生命周期复用，**不能相加**当作实际可省显存。
- 价值不在提 tok/s，而在降低工作区峰值，给更大的 prefill chunk、更多 KV 或并发腾空间。
