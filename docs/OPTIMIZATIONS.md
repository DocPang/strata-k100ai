# 累计优化与发布范围

v0.1.0-rc.5 固定分发 `d87a4678a21b` 引擎。该程序同时包含此前 K100AI 平台适配、已保留的内核优化及本次 XXS 长上下文改动。本页将累计实现与本次新增项分别说明。

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
| MMAC scores 与 LDS 合并装载 | SCORES_MMAC=2、SCORES_MMAC_LDS=1、SCORES_MMAC_NB=1 | 本次主要新增；LDS 合并装载是已测长上下文收益的必要条件 |
| FP16 scores 缓冲 | SEL_SCORES_FP16=1 | 本次主要新增；每分数 4→2 字节，改变中间精度，不能宣称所有输入逐位等价 |
| Topk 访问与 selection 批次 | TOPK_STRIDE 默认开启；SEL_BATCH 未显式覆盖 | lane-major 访问，批次按 max_blocks 分档；不是后续 16 位两遍 radix 实验 |
| Attention V / K 暂存及索引 | INT8 V 暂存、合并读取、页索引计算；PA_KSTAGE 默认开启并模板化 | K 暂存模板化属于本次主要新增；保留此前 V 侧和地址计算优化 |
| Lane/cell 分派与 PLE 调度 | ATTN_LANECELL=1、PLE_PRELAUNCH=1 | 作用于各自 attention/PLE 路径；具体命中以实际分派为准 |
| MMVQ LDS 与 wave 路径 | MMVQ_WV2_LDS=1；保留 wave 分派 | 本次启用与验证 LDS 路径；未启用额外的 WV2_GROUP 分组配置 |

环境变量表中省略的 `STRATA_` 前缀适用于所有开关。默认开启的实现无需在环境文件中重复声明；未见变量不等于缺失优化。

## 本次新增项的单项记录

| 项目 | 已有同批测试记录 |
| --- | --- |
| MMAC scores + LDS 合并装载 | 250K 单项整模耗时约 −12.2% |
| IQ2 exact 扩展 | 250K 单项整模耗时约 −6.8% |
| FP16 scores | 250K 单项整模耗时约 −2.3% |
| K 暂存模板化 | 寄存器记录为 VGPR 131→119，改善对应 attention 相位 |
| MMVQ LDS | 指定 decode 负载 A/B/A/B 中，窗口平均耗时约 −2.2% |

这些数字来自不同单项对照，不能相加。最终组合的 15.71 / 70.83 / 146.79 秒对应固定输入和配置，见 [BENCHMARKS.md](BENCHMARKS.md)。源码快照恢复不等于完整干净重建验证，见 [VALIDATION.md](VALIDATION.md)。

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
- [c4 到当前 XXS 快照的增量](../engines/strata/xxs-20261016/patches/c4-to-xxs-20261016.patch)，覆盖 10 个变更文件。
- [当前环境配置](../engines/strata/xxs-20261016/environment.env)及[源码恢复校验表](../engines/strata/xxs-20261016/source-recovery.json)。

c4 配方保留 21 个修改文件、4 个新增实现头文件；ggml/llama.cpp 补丁包含 2 个修改文件和 1 个新增配置。这是文件变更范围，不是独立优化项数量，也不代表每个历史候选路径均在当前配置启用。

## 未计入本版本的项目

后续 GDN chunked 平台修复、16 位 topk 及之后的研究构建、TP2、不满足性能或正确性检查而关闭的候选、仅有组件测量但未保留的实验，均不计入当前优化成果。计时、日志和硬件屋顶线探针属于研究工具，不列为运行性能优化。
