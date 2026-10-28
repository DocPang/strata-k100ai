# 发布验证

## v0.2.1

- 文本引擎不变：`bdbfd934…`（45,674,152 B）。新增图片编码器 `runtime/bin/strata-vision`，SHA-256 `9c4dc14303e508c0c45814d886d56456a0b0b70a8732e04f9555c0491fdb41d4`（119,439,216 B），HIP/gfx928。
- 运行包 `strata-xxs-20261023-k100ai022-linux-amd64.tar.gz`：29,472,859 B，SHA-256 `dc6a0e96…`，`FILES.sha256` 206 项全部校验通过（新增 `bin/strata-vision`）。
- 单图 760×520（475 image tokens）：GPU 编码 1.72 s；CPU 8 线程对照 18.0 s。GPU/CPU embeddings cosine mean 0.999293、min 0.947855、rel-L2 3.24%。
- 1800×1200 图按 1024 image tokens 编码，端到端 12.5 s（含 309 tokens 生成）。
- 长上下文 40K / 100K / 230K token 请求期间 VRAM 峰值 60,246 MiB，剩余 5,274 MiB；KV 未额外占显存。
- 正确性关键项：修复 `norm_f32<1024>` 的 launch-bounds 问题后，embeddings 才与 CPU 一致；未修时 cosine 约 0.05。
- 端到端客户端：DSH 桌面 `describe_image` 10 s、内置 `read_image` 6 s，两张测试图的标题、字段、表格与柱状图数值均正确。
- 未覆盖：更多图片类型/分辨率的质量评测、并发图片请求、长时间浸泡、其他 DTK/架构。

## v0.2.0

- 引擎 `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9`（45,674,152 B），运行包 `strata-xxs-20261023-k100ai022-linux-amd64.tar.gz`（SHA-256 `359baf70…`，205 项校验通过）。
- 对照是**上一版 rc.7 引擎**（`ec7021b0…`），同卡、同会话交替：预填 0.985–0.988（30K/126K）、decode 9/9 持平或领先；模块级见 [完整成绩](BENCHMARKS.md)。
- 输出与 rc.7 逐字节一致，唯一例外是 hc 融合（默认开，`STRATA_K100AI_HC_FUSE=0` 可逐位还原）。
- `clean_rebuild_verified` 为 `false`：同构建目录内逐字节可复现，换目录因内嵌构建路径而变化。

以下 rc.6 / rc.7 的记录是历史，保持原样。

## 固定程序与测试来源

v0.1.0-rc.6 分发 `strata-xxs-20261016-gold-linux-amd64.tar.gz`，包含以下冻结二进制：

```text
43002a7262e50d836f01642d1a5846565d8cd0734ec1d0ef67f534aeb11aafac
```

该文件为 2026-10-16 20:20（UTC+08:00）的增量构建，大小 29290960 字节，当天已部署到生产实例；发布时直接复制并校验原文件，不重新编译为另一个程序。上一版 v0.1.0-rc.5 分发 `d87a4678a21b`，其摘要、成绩与检查记录保留在 [rc.5 引擎目录](../engines/strata/xxs-20261016/) 供对照。

## 已有 GPU 记录

- rc.5 的 32K/128K/250K 同批 A/B/A prefill、250K 第二批重复测试、prefill=20480 对照，以及 MMVQ LDS 的 32K 输入 / 128-token 输出 A/B/A/B decode 检查，原始字段见 [rc.5 样本汇总](../engines/strata/xxs-20261016/benchmarks.json)。
- rc.6 对 32K / 128K / 250K 三档都做了**同批 A/B/A**（本版二进制对上一版 rc.5 二进制，同卡同批、各用自己的发布环境）：32K 15.620 → 14.885 s、128K 70.715 → 65.848 s、250K 146.780 → 137.249 s；每档两个版本的输出逐位一致。
- 单开关对照（同卡同批、只差 `STRATA_GEMM_Y_F16`）：32K −5.49%、250K −3.43%。
- 解码侧对 `STRATA_MMVQ_WV2_GROUP` 做了 balanced A/B/A/B（两批共 8 个启用样本、6 个未启用样本）：81.96 对 82.00 tok/s，差异落在样本漂移内，即**组件级收益未在整模复现**。
- rc.6 打包时对 HCU0 上正在运行的生产引擎（同一二进制）发起验证请求，原始字段见 [rc.6 样本汇总](../engines/strata/xxs-20261016-gold/benchmarks.json)。
- rc.6 部署后在两个生产实例上的在线验证记录：120,670-token 请求 prefill 65.281 s（1848.5 tok/s）、32,230-token 请求 17.113 s（1883.4 tok/s）；配置为 `--prefill 20480`，与上述实验卡协议不同。

两次生产验证请求先后执行，第二次的专家缓存已被第一次预热，不是受控 A/B；墙钟时间含客户端开销，不是引擎内部相位计时。定版批次的每个臂输出 token 列表 SHA 完全一致（250K `08d2c11d70446fe6`、32K/解码 `97e6833014e9deab`）；rc.5 批次中短输入有一个 EOS 后 token 差异记录。FP16 scores 与 `GEMM_Y_F16` 改变中间精度，不构成所有提示的位等价保证。已有测试不是多任务质量评测或长期浸泡测试。

## 发布打包检查

- 从 Release 重新下载 `strata-xxs-20261016-gold-linux-amd64.tar.gz`：8934284 字节，SHA-256 `241655d69476df9fb4159589bb7399d95daeba190c882544016f16b10d4b3155`，与 [artifact.sha256](../engines/strata/xxs-20261016-gold/artifact.sha256) 一致。
- 归档共 224 个条目；解包后 `runtime/FILES.sha256` 列出的 205 个文件全部校验通过。
- 解包得到的 `runtime/bin/strata` SHA-256 等于上面的发布摘要，大小 29290960 字节。
- 运行包继续使用已审核的 c4 Python 服务、转换工具和辅助数据，只有 `bin/strata` 相对 rc.5 不同。rc.5 的 15 项本地配置/离线安装测试、派生镜像构建、动态库解析、工具帮助入口与原生 lazy 服务检查记录见 [rc.5 发布检查记录](../engines/strata/xxs-20261016/release-validation.json)；这些检查本轮没有在 rc.6 上重跑。
- 仓库本地检查在本次文档更新后运行：`bash -n install.sh start.sh stop.sh scripts/config.sh` 通过；`python3 -m unittest discover -s tests` 15 项全通过，其中离线安装契约用重新下载的 rc.6 运行包以 `--local-only --bundle` 执行；文档内 bash 代码块语法检查通过。Compose 校验需要 Docker Compose 插件，本轮未执行。

本次打包不启动新的 GPU 研究实验、不切换生产服务。

## 源码与未完成项

附加源码补丁根据保留的源码与变更前备份恢复，校验表见 [source-recovery.json](../engines/strata/xxs-20261016/source-recovery.json)。补丁可在已恢复 c4 源码上回放；这不等于验证历史继承对象全部来自同一源码快照，也不等于完成干净重建。**rc.6 相对该快照的增量源码不在本仓库内**，见 [rc.6 引擎清单](../engines/strata/xxs-20261016-gold/manifest.json)。

仍需完成：完整干净源码重建、通用质量对照、该服务配置的 GPU 端到端与长时间稳定性验证、其他驱动/DTK 组合和 TP2 的兼容矩阵。公开默认不加载控制向量。
