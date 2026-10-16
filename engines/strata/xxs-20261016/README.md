# IQ3_XXS 优化引擎：2026-10-16

本目录对应 v0.1.0-rc.5，分发已完成指定负载 GPU 测试的冻结程序 `d87a4678a21b`，完整摘要和大小见 [manifest.json](manifest.json)。

- SCORES_MMAC=2 + LDS=1 + NB=1：合并装载与矩阵计算。
- SEL_SCORES_FP16=1：2 字节 selection 分数缓冲。
- GATEUP_EXACT_IQ2=1：IQ2_XS/IQ2_XXS exact 扩展。
- MMVQ_WV2_LDS=1：decode 稠密计算路径。
- PA_KSTAGE 默认开启，模板化已编入；不设置 SEL_BATCH 覆盖值。

[benchmarks.json](benchmarks.json)保存经脱敏的原始结果字段。三档成绩及正确性限制见 [BENCHMARKS.md](../../../docs/BENCHMARKS.md)。后续平台修复与 16 位 topk 不包含在本引擎中。

`patches/c4-to-xxs-20261016.patch` 是从已恢复 c4 源码到所保留历史快照的增量，输入及备份选择见 [source-recovery.json](source-recovery.json)。完整干净重建和全部继承对象的来源校验尚未完成；该补丁不作为重新构建出相同二进制的证明。
