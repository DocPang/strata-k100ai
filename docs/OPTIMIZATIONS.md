# 累计优化与发布范围

**v0.2.0（当前）** 分发 `bdbfd934…`，重基到上游 0.2.2。本版新增：down 投影的 prefetch 路径与单 plane 分块、hc read 的 `[wd|wi]` 宽 GEMM（`STRATA_K100AI_HC_FUSE`）、`moe_combine` 的 `SD` 特化、gate/up exact IQ2 / wsg16 / down GEMM / IQ4 预展开，以及 decode 图预热（`STRATA_MTP_WARM_AT_START`）。对照 rc.7 的实测与模块级拆分见 [完整成绩](BENCHMARKS.md)。

以下 rc.6 / rc.7 的条目是历史记录。

v0.1.0-rc.6 固定分发 `43002a7262e5` 引擎；上一版 v0.1.0-rc.5 分发 `d87a4678a21b`。rc.6 在 rc.5 之上继续增加若干内核路径，两者都包含此前 K100AI 平台适配与已保留的内核优化。本页将累计实现、按版本区分的新增项分别说明。

“采用”表示该实现或配置包含在发布方案中；具体执行仍受量化类型、张量形状和运行分派约束。已有上游内核的适配与开关选择，不等于新增一种算法；也不能将所有历史单项收益归到本次发布。

## 平台与量化计算

| 累计内容 | 实现或启用方式 | 适用范围与边界 |
| --- | --- | --- |
| gfx928 / DTK 平台适配 | HIP 兼容层、架构分派、dot4/MMAC 相关编译配置及 DTK 库链接 | 平台适配；构建选项中的 gfx906 后端命名不等于实际目标设备是 gfx906 |
| ggml / llama.cpp 衔接 | HIP 通用头、供应商接口与 gfx928 MMQ 配置 | c4 基础源码恢复配方保留相关补丁，不宣称已在其他后端验证 |
| Q6_K / Q8_0 packed 布局 | Q6_PACKED、Q8_PACKED、Q6_WAVE_PACK_PREP、Q6_WAVE_PACK | 发布环境显式启用，按张量类型分派；XXS 的目标输出头是 Q5_K，不能将 Q6_K 头部优化计为其收益 |
| MoE 常驻专家排序 | MMQ_RESIDENT_SORT_NE、MMQ_RESIDENT_SORT_NATIVE | 全层专家驻留等条件满足时生效；不是所有层和路径都执行排序 |
| Gate/up exact 与 MMAC | MMQ_EXACT_ALL、GATEUP_EXACT_IQ2；`gateup_exact.hpp`、`kgu_wsg16.hpp` | 从已有类型扩展到 IQ2_XS/IQ2_XXS，覆盖该 XXS 权重的 48 层 gate/up；exact 子路径不代表整个模型中间结果逐位一致 |
| Down 展开与计算 | IQ4_NL exact preexpand、Q2_0 展开、int8 MMAC、staging、DOWN_MT | 保留 `iq4_preexpand_exact.hpp` 与相关 MMQ 分派；按格式、行数和形状命中 |
| GEMM 形状自适应 | `moe_mmq.cu` 按专家组 max_rows 选择 tile | 编译入分派逻辑，不对应一个额外的环境变量 |
| 稠密 GEMM 布局与接口 | PF_PAD_ROCBLAS、DTK rocBLAS/hipBLASLt 适配 | 发布配置启用步长 padding；不表示所有投影均走 hipBLASLt，也不沿用失败候选路径的性能估计 |
| 稠密 GEMM solution 锁定 | `STRATA_GEMM_SOL=<N>x<K>=<solution>`，见 [PREFILL_GEMM_SOL.md](PREFILL_GEMM_SOL.md) | 仅对命中形状换 Tensile solution，未命中走上游默认；与 `STRATA_DENSE_TN` 互斥（TN 分支提前返回会绕过 hook），不可合并 |

## GDN、Hyper-connection 与数据复用

| 累计内容 | 实现或启用方式 | 适用范围与边界 |
| --- | --- | --- |
| 八 lane GDN 递归 | GDN_HEAD=1；GDN_OCT 未设置时默认开启 | 使用 `gdn_rec_oct_kernel`；不是后续 chunked scan 实验 |
| 无用中间写回消除 | GDN_NOY=1 | 适用路径不再写出无消费者的 FP32 结果；保留后续所需数据 |
| HC upmix 融合 | HC_UPMIX 未设置时默认开启 | gfx928 up 投影与混合步骤融合，位于 prefill 数据路径 |
| HC split / staged read 与预取 | HC_SPLIT=2、GR_DOWN_PREFETCH=2 | decode/验证相关路径按设备检查与实际分派执行；与未启用的 prefill HC_SPLIT2 实验开关不同 |
| 唯一行量化后 gather | GATHER_QONCE 未设置时默认开启 | MMQ 数据准备中先量化 T 行，再按专家索引复制，避免对重复的 T×K 槽重新量化 |
| SwiGLU 与量化融合 | PF_FUSED_SQ=1 | 用融合计算减少中间 H 的读写及单独量化启动 |
| 工作区布局与记账 | PF_GATED_IN_REGION、EMB_REUSE_ACCOUNT、PF_DMSCRATCH_COND、PREFILL_RING | gated 共享工作区、embedding 别名复用、按需 scratch 与暂存 ring；降低无必要的分配或保守重复记账 |

## QSA、Attention 与 Decode

| 累计内容 | 实现或启用方式 | 适用范围与边界 |
| --- | --- | --- |
| Scores 网格轴序 | SCORES_SWAP 未设置时默认开启 | 改变 CTA 遍历次序，改善相邻 query 的数据复用；不是修改模型选择预算 |
| MMAC scores 与 LDS 合并装载 | SCORES_MMAC=2、SCORES_MMAC_LDS=1；合并批次 SCORES_MMAC_NB 由 1 提至 2（rc.6） | rc.5 引入 LDS 合并装载，是已测长上下文收益的必要条件；rc.6 将合并批次翻倍 |
| FP16 scores 缓冲 | SEL_SCORES_FP16=1 | 本次主要新增；每分数 4→2 字节，改变中间精度，不能宣称所有输入逐位等价 |
| Topk 访问与调度 | TOPK_STRIDE 默认开启、SEL_BATCH 未显式覆盖；rc.6 增加 TOPK_H16=1 与 TOPK_REG_MAX=7168 | lane-major 访问，批次按 max_blocks 分档；rc.6 的 topk 走 2-pass h16 核并用寄存器缓存最大值。核的线程数扩容 `TOPK_H16_T` 已测为显著回归，本版不设置 |
| Attention V / K 暂存及索引 | INT8 V 暂存、合并读取、页索引计算；PA_KSTAGE 默认开启并模板化 | K 暂存模板化属于本次主要新增；保留此前 V 侧和地址计算优化 |
| Lane/cell 分派与 PLE 调度 | ATTN_LANECELL=1、PLE_PRELAUNCH=1 | 作用于各自 attention/PLE 路径；具体命中以实际分派为准 |
| MMVQ LDS 与 wave 路径 | MMVQ_WV2_LDS=1；rc.6 增加 MMVQ_WV2_GROUP=1；保留 wave 分派 | rc.5 启用并验证 LDS 路径；rc.6 的 WV2 分组只有核级逐位与微基准证据，整模 A/B 未做 |
| Q6K packed-wave LDS | Q6_WAVE_PACK_LDS=1（rc.6，decode） | 作用于 Q6K packed-wave 路径；按张量类型分派，不适用于 XXS 的 Q5_K 输出头 |
| 稠密 GEMM 累加器精度 | GEMM_Y_F16=qkv\|qf\|sg\|su\|sd\|z（rc.6） | 六个投影族的 GEMM Y 累加器改走 16 位路径；改变中间精度，正确性范围见成绩说明 |

环境变量表中省略的 `STRATA_` 前缀适用于所有开关。默认开启的实现无需在环境文件中重复声明；未见变量不等于缺失优化。

## 按版本的新增项单项记录

### v0.1.0-rc.6

| 项目 | 已有测试记录 |
| --- | --- |
| GEMM_Y_F16（六投影族） | **同批 A/B/A**：250K 142.3606 → 137.4764 s（−3.43%）；32K 15.8336 → 14.9644 s（−5.49%）；输出逐位一致 |
| TOPK_REG_MAX=7168 | **同批 A/B/A**：250K 三臂 144.572 / 142.119 / 144.205 s，整体 −1.57%、qsa select −10.45%；32K 中性；输出逐位一致 |
| TOPK_H16=1 | 250K qsa select 22,667 → 20,703（−8.7%）、整模约 −1.2%；128K select −1.4%；输出逐位一致 |
| SCORES_MMAC_NB 1→2 | 同批对照样本已并入上表的基线臂，未单独保存 |
| Q6_WAVE_PACK_LDS=1（decode） | **同批 A/B/A/B**：decode 81.36 → 82.07 tok/s、窗口 40.45 → 40.05 ms，10/10 配对一致；幅度接近样本漂移下限 |
| MMVQ_WV2_GROUP=1（decode） | 组件级：核级输出逐位一致、孤立微基准 −24.8%（nc=5、300 次调用）。**整模未复现**：两批共 8 个启用样本与 6 个未启用样本，decode 81.96 对 82.00 tok/s（差 ≤0.1%），落在样本漂移内；不影响 prefill 成绩 |
| TOPK_H16_T（线程数扩容） | **已否决**：250K 整模 +11.96%、qsa select +86%（相对非 regmax 基线）；相对单独 REG_MAX 基线 +13.7%。6 个臂输出 SHA 一致 |

### v0.1.0-rc.5

| 项目 | 已有同批测试记录 |
| --- | --- |
| MMAC scores + LDS 合并装载 | 250K 单项整模耗时约 −12.2% |
| IQ2 exact 扩展 | 250K 单项整模耗时约 −6.8% |
| FP16 scores | 250K 单项整模耗时约 −2.3% |
| K 暂存模板化 | 寄存器记录为 VGPR 131→119，改善对应 attention 相位 |
| MMVQ LDS | 指定 decode 负载 A/B/A/B 中，窗口平均耗时约 −2.2% |

这些数字来自不同单项对照，不能相加。整版口径的同批 A/B/A（rc.5 → rc.6）为 32K 15.620 → 14.885 秒、128K 70.715 → 65.848 秒、250K 146.780 → 137.249 秒，见 [BENCHMARKS.md](BENCHMARKS.md)。源码快照恢复不等于完整干净重建验证，见 [VALIDATION.md](VALIDATION.md)。

## 部署配置与上游能力

| 配置 | 本项目采用方式 |
| --- | --- |
| IQ3_XXS 权重 | 固定分片校验值，独立生成 native pack；量化权重本身来自模型发布者 |
| 专家缓存 | 24576 槽全部驻留，避免专家回退到主机内存 |
| KV | 最大上下文 262144、INT8、全显存驻留；使用上游 KV 能力 |
| Prefill 与显存规划 | prefill=30602、reserve=2048 MiB，工作区受限时先降低 prefill |
| MTP | 使用上游 MTP 机制和匹配 runtime，本版本采用 spec=4、min-p=0.70 |
| 调度 | 单槽 FIFO；多卡独立实例，不等于 TP2 |
| 会话缓存与接口 | 保留上游 HTTP/API 与缓存能力，默认关闭 prompt/conversation cache 以复现冷 prefill |

Docker 组合安装、模型下载命令、权重校验、设备选择和服务排查属于部署工程，不单独作为内核性能提升计算。

## 源码核对依据

- [c4 Strata 累计补丁](../engines/strata/c4/replay/patches/strata-c4.patch)与 [ggml/llama.cpp 适配补丁](../engines/strata/c4/replay/patches/llama-gfx928.patch)。
- [c4 到 XXS 快照的增量](../engines/strata/xxs-20261016/patches/c4-to-xxs-20261016.patch)，覆盖 10 个变更文件。**rc.6 的增量源码不在本仓库内**，其清单见 [rc.6 引擎目录](../engines/strata/xxs-20261016-gold/manifest.json)，`clean_rebuild_verified` 为 `false`。
- [当前环境配置](../engines/strata/xxs-20261016-gold/environment.env)（rc.6）及 [rc.5 快照的源码恢复校验表](../engines/strata/xxs-20261016/source-recovery.json)。

c4 配方保留 21 个修改文件、4 个新增实现头文件；ggml/llama.cpp 补丁包含 2 个修改文件和 1 个新增配置。这是文件变更范围，不是独立优化项数量，也不代表每个历史候选路径均在当前配置启用。

## 未计入本版本的项目

rc.6 之后的 GDN chunked 平台修复、TP2、不满足性能或正确性检查而关闭的候选（含 `TOPK_H16_T` 线程数扩容）、仅有组件测量但未保留的实验，均不计入当前优化成果。计时、日志和硬件屋顶线探针属于研究工具，不列为运行性能优化。
