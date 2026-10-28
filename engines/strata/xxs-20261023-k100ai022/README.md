# xxs-20261023-k100ai022 — v0.2.1（文本引擎同 v0.2.0，新增多模态）

引擎 SHA-256 `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9`（45,674,152 B）。
运行包 `strata-xxs-20261023-k100ai022-linux-amd64.tar.gz`（29,472,859 B，SHA-256 `dc6a0e96…`）：在 v0.2.0 文本引擎之外加入 `runtime/bin/strata-vision`（HIP 图片编码器），并重算 `PROVENANCE.json` 与 `FILES.sha256`（206 项）。

v0.2.1 新增的视觉编码器 SHA-256 `9c4dc14303e508c0c45814d886d56456a0b0b70a8732e04f9555c0491fdb41d4`（119,439,216 B），配置 `gpu: true`、`max_tokens: 1024`，源码与补丁见 [sources/vision-20261028](../../../sources/vision-20261028/)。文本引擎与 v0.2.0 完全相同，下面的文本成绩继续适用。

## 这一版做了什么

把 k100ai 优化集重基到上游 strata **0.2.2（`011c9e60`）**，源码增量 36 个文件（30 改 / 6 增），见 [sources/xxs-20261023-k100ai022](../../../sources/xxs-20261023-k100ai022/)。主要项：

- down 投影的 prefetch 路径（单 LDS plane、TH=128）+ 单 plane 分块：hc read 阶段从表观落后 38% 变为快 11%
- hc read 的 `[wd|wi]` 宽 GEMM（`STRATA_K100AI_HC_FUSE`，默认开）
- `moe_combine` 的 `SD` 特化：combine 411 → 371 ms
- gate/up exact IQ2、wsg16 tile、down GEMM、IQ4 预展开
- decode 图预热（`STRATA_MTP_WARM_AT_START`，默认开）

## 成绩（对上一版 rc.7，同卡同会话交替）

| 套件 | 预填 | 比值 | decode（每请求） | 比值 |
| --- | ---: | ---: | ---: | ---: |
| ctx32 30,603 | 14,232 / 13,947 ms | 0.988 / 0.985 | 84.8 / 149.7 / 135.1 ms | 0.944 / 0.984 / 0.991 |
| lt 7,556 | 4,497 / 3,851 ms | 1.003 / 0.974 | 83.6 / 140.5 / 129.2 ms | 0.955 / 0.995 / 0.998 |
| ctx128 126,857 | 67,401 / 67,350 ms | 0.986 / 0.986 | 85.5 / 369.5 / 321.7 ms | 0.966 / 0.975 / 0.968 |

完整拆分见 [docs/BENCHMARKS.md](../../../docs/BENCHMARKS.md)，原始记录见 [benchmarks.json](benchmarks.json)。

## 注意

- 与 rc.7 的差值包含上游基线本身的变化（rc.7 基于 `fb58e0db`，本版基于 `011c9e60`）。
- 输出与 rc.7 逐字节一致，唯一例外是 hc 融合（`STRATA_K100AI_HC_FUSE=0` 可逐位还原）。
- `clean_rebuild_verified` 为 `false`：同构建目录内逐字节可复现，换目录因内嵌构建路径而变化。
