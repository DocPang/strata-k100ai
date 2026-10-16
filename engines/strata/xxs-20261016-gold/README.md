# IQ3_XXS 优化引擎（gold）：2026-10-16

本目录对应 v0.1.0-rc.6，分发 `43002a7262e5` 固定引擎，完整摘要和大小见 [manifest.json](manifest.json)。

相对上一版 [xxs-20261016](../xxs-20261016/README.md)（`d87a4678a21b`），本版在 prefill 与 decode 两条路径上继续做了增量优化：

- `STRATA_GEMM_Y_F16=qkv|qf|sg|su|sd|z`：六个投影族的 GEMM Y 累加器改走 16 位路径。
- `STRATA_TOPK_H16=1` + `STRATA_TOPK_REG_MAX=7168`：topk 走 2-pass h16 核，并用寄存器缓存最大值，减少每轮的重复扫描。
- `STRATA_SCORES_MMAC_NB` 由 `1` 改为 `2`：scores 矩阵计算的合并批次翻倍。
- `STRATA_Q6_WAVE_PACK_LDS=1`：Q6K packed-wave 走 LDS 路径（decode）。
- `STRATA_MMVQ_WV2_GROUP=1`：MMVQ WV2 分组。

## 已否决、未纳入本引擎

- `STRATA_TOPK_H16_T`（把 h16 topk 核的线程数从 256 提到 1024）：250K 下实测整模 **+11.96%**、qsa select **+86%**；以单独的 `REG_MAX` 基线（142,119 ms）计则是 **+13.7%**。同批次 6 个臂的输出 SHA 完全一致，说明回归纯粹出在性能而非结果。源码保留默认关闭的实例化，键不设置。

## 性能与校验范围

[benchmarks.json](benchmarks.json) 保存打包时对**正在运行的生产引擎**做的验证请求，以及同二进制在实验卡上的工程记录。两组数字的测量条件不同，不可混用：

- 打包验证（生产卡 HCU0）：约 39K token 输入 20.38 s、约 66K token 输入 25.73 s；短提示 + 300 token 输出，TTFT 0.70–1.06 s。
- 工程记录（实验卡 HCU5）：250K prefill 142,119 ms / 1,759 tok/s；32K 15,597 ms / 1,962 tok/s；decode 约 82 tok/s。

**本包不包含源码快照或增量补丁**，`clean_rebuild_verified` 为 `false`，请勿把本目录当作"可从公开材料重建出相同二进制"的证明。权重、供应商库、MTP 草稿模型均不在包内。
