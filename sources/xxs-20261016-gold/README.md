# xxs-20261016-gold — v0.1.0-rc.6 源码快照

引擎二进制 SHA-256 `43002a7262e50d836f01642d1a5846565d8cd0734ec1d0ef67f534aeb11aafac`（29,290,960 B），对应 Release [v0.1.0-rc.6](https://github.com/DocPang/strata-k100ai/releases/tag/v0.1.0-rc.6) 的 `strata-xxs-20261016-gold-linux-amd64.tar.gz`。

## 这个版本改了什么

父版本 `strata/xxs-20261016`（`d87a4678…`）之上，本版只做一件事：**把 prefill 的 GEMM Y 累加器改成 16 位紧凑图像**，门控变量 `STRATA_GEMM_Y_F16=qkv|qf|sg|su|sd|z`。

机制：`Gemm::native` 在 padr 路径上用 `ldc = y_f16 ? ldy : 2*ldy` 直接写紧凑 FP16 图像并跳过 `widen_rows_f16`；所有消费者按同一索引读 `__half2float(((const __half*)Y)[i])`，构造上逐位等价。

必须覆盖 `Gemm::native` 的**全部 5 条**输出路径（`strata_pf_gemm_f16_ld`、padr rocBLAS、padr hipBLASLt、padded-X、`f16_inplace`）。首版曾因漏改 `f16_inplace`（N=640/K=640 不走 padr）在非 padr 形状报 `non-finite first logits`，修复见 `patches/patch_yf16_fix.py`。

## 补丁顺序（必须按此顺序）

| 顺序 | 文件 | 作用 |
| --- | --- | --- |
| 1 | `patches/patch_yf16_qkv.py` | `m.qkv`，12 个锚点 |
| 2 | `patches/patch_yf16_more.py` | `qf` / `sg` / `su` / `sd`（锚点用空白容错正则） |
| 3 | `patches/patch_yf16_fix.py` | 把 `y_f16` 串进 `Gemm::f16` → `f16_inplace`，修非 padr 形状 NaN |
| 4 | `patches/patch_yf16_z.py` | `m.z`（幂等判据用 `z_f16`） |

补丁全部 fail-closed：锚点未全部匹配则拒绝写入。

## 文件清单

| 文件 | SHA-256（前 16） | 字节 |
| --- | --- | --- |
| `include/strata/prefill/gemm.hpp` | `c0306fc035515563` | 5,669 |
| `include/strata/prefill/kernels.hpp` | `e4c82486d6e4f3ae` | 13,011 |
| `src/prefill/gemm.cu` | `0ab7168bbfc90157` | 75,042 |
| `src/prefill/kernels.cu` | `4ad32864e5fbfec3` | 168,827 |
| `src/prefill/prefill.cpp` | `c532288ac64ae379` | 301,710 |
| `src/kernels/cuda/qsa_select.cu` | `6839faf71f3c9eec` | 130,898 |
| `src/kernels/cuda/native_mmvq.cu` | `d93fd9ebc8e88d90` | 145,338 |

后两个文件在本版补丁链中未改动，列出是为了让"这 7 个文件即本版全部增量"这一点可自证。完整校验值见 `manifest.json`。

## 范围说明

- 本目录**只是增量**，不是完整引擎树。平台适配、MMVQ、QSA/MMAC、GDN、gate/up、down、workspace、融合、MTP 等此前累计工作见 `engines/strata/xxs-20261016/patches/c4-to-xxs-20261016.patch`。
- `vs-upstream-fb58e0db.diff` 是**这 7 个文件**相对上游 `fb58e0db` 的累计差异，共 5,097 行，包含历史轮次的改动，不等于整棵树的差异。
- 重编译不保证字节相同；字节身份以 `engine_sha256` 为准。

## 验证状态

- 同批 A/B/A（卡4/卡5，基线臂与候选臂同卡、逐位一致）：250K `142,360.6 → 137,476.4 ms`（−3.43%），30K `−5.49%`。
- 逐位门：250K `outsha=08d2c11d70446fe6`、32K/decode `97e6833014e9deab`，`fallbacks=0`。
- 生产两卡部署后 `/health` OK，二进制 SHA 与发布包一致。
- **未做**：从本目录重编译并与 `43002a72` 做字节级比对（`clean_rebuild_verified: false`）。
