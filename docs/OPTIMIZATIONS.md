# 优化内容

v0.1.0-rc.5 固定使用 `d87a4678a21b` 引擎，针对 Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS 和 gfx928 优化。

| 优化 | 开关 / 实现 | 已有依据 |
| --- | --- | --- |
| MMAC scores 与合并装载 | SCORES_MMAC=2、SCORES_MMAC_LDS=1、SCORES_MMAC_NB=1 | 250K 单项同批整模耗时约 −12.2%；LDS 合并装载是该结果的必要条件 |
| IQ2_XS / IQ2_XXS exact gate/up | GATEUP_EXACT_IQ2=1 | 补齐 19 层，48 层 gate/up 全部具备 exact 路径；250K 单项整模约 −6.8% |
| FP16 scores 中间缓冲 | SEL_SCORES_FP16=1 | 250K 单项整模约 −2.3%；改变中间精度，不宣称所有输入逐位等价 |
| Attention K 暂存模板化 | 编译期分派，PA_KSTAGE 默认开启 | 已有记录中 VGPR 131→119；减少暂存实例的寄存器负担 |
| MMVQ LDS | MMVQ_WV2_LDS=1 | 指定 decode 负载 A/B/A/B 中，窗口平均耗时约 −2.2% |
| 显存与调度 | XXS、全专家驻留、256K 全 KV、单槽 FIFO | 避免主机专家访问和额外会话槽挤占显存；多卡使用独立实例 |

以上单项收益来自各自同批实验，不应相加得到组合收益。组合实测以 [BENCHMARKS.md](BENCHMARKS.md) 为准。优化服务继续使用 c4 来源的 Python 前端、协议与转换工具；许可和源码恢复见 [SOURCE_BUILD.md](SOURCE_BUILD.md)。

本版不包含后续 GDN chunked 平台修复、16 位 topk 或之后的研究构建。`STRATA_SEL_BATCH` 不显式覆盖，沿用引擎按最大上下文缓冲尺寸选择的值。
