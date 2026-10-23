# 参数参考

当前推荐使用 `profiles/next-iq3xxs-single/strata.json`；IQ3_S 列为历史参考。编辑 `args` 中的参数及对应值后，停止并重建目标容器。优化环境通过 `engines/strata/xxs-20261023-k100ai022/environment.env` 传入；配置文件的 `env` 可覆盖指定变量。

## 模型与设备

| 参数 | 说明 |
| --- | --- |
| --gpu 0 | 使用容器可见的 GPU 0；宿主物理卡通过 Docker render 节点映射选择 |
| --pack | 与当前权重匹配的 native pack 目录 |
| --native | GGUF 第一分片，包含主模型数据 |
| --ple-gguf | PLE 所在的第二分片 |
| --ple-io mmap | 以 mmap 访问 PLE 数据，使用宿主页缓存 |
| --mmap-experts | 使用专家 mmap 相关加载路径；实际驻留由缓存参数和日志确认 |
| --expert-profile | 专家分布辅助数据文件 |
| --mtp | 独立 MTP runtime 目录 |
| --eos-ids 248046 | 参考模型的结束 token ID |

默认文件名分别为 `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-...gguf` 和 `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-...gguf`。实际文件名不同时应修改配置，不依赖重命名权重规避类型核验。

## 容量与计算

| 参数 | IQ3_S | IQ3_XXS | 说明 |
| --- | ---: | ---: | --- |
| --expert-cache | 24576 | 24576 | 全部专家槽；减少会影响驻留和搬运 |
| --prefill | 20480 | 30602 | prefill 分块上限，与临时工作区和 TTFT 有关 |
| --max-context | 262144 | 262144 | 会话总上下文容量 |
| --kv | int8 | int8 | KV 存储格式 |
| --kv-resident | 32768（历史） | 省略 | KV 显存缓存容量；0 为完整驻留，c4 的非零有效下限为 20480 |
| --vram-reserve-mib | 2048 | 2048 | GPU 内存规划预留，单位 MiB |
| --pool-workers | 12 | 12 | CPU 专家工作线程数量 |
| --no-prefill-borrow | 开启 | 开启 | 禁止 prefill 借用专家区域，保留参考配置的驻留策略 |

容量估算、OOM 调整顺序及系统内存开销见 [MEMORY.md](MEMORY.md)。

## MTP

| 参数 | IQ3_S | IQ3_XXS | 说明 |
| --- | ---: | ---: | --- |
| --spec | 4 | 4 | 草稿长度；需联合接受率和窗口成本评估 |
| --spec-min-p | 0.70 | 0.70 | 草稿概率阈值 |

更长草稿会增加草稿计算、验证窗口及相关状态。接受率较低时可能降低吞吐，不能仅按 spec 数值估算速度。独立 MTP 权重和 runtime 必须与配置共同准备。

## 服务缓存

| 参数 | 默认 | 说明 |
| --- | ---: | --- |
| --prompt-cache | 0 | prompt 缓存配置 |
| --prompt-cache-every | 16384 | prompt 缓存间隔 |
| --conversation-cache-mib | 0 | 宿主会话缓存预算，MiB |
| --conversation-cache-slots | 4 | 会话缓存条目数 |
| --conversation-cache-min-free-mib | 8192 | 会话缓存保留的主存余量阈值 |

这些参数用于前缀/会话复用，不等同于并发解码配置。推荐配置省略 `--batch` 和 `--batch-mtp`，保持单槽 FIFO。

## 优化环境

XXS 使用新引擎环境文件，IQ3_S 历史配方使用 c4 环境。新增推荐变量：

| 变量 | 设置与作用 |
| --- | --- |
| STRATA_PF_FUSED | 1，启用 fused native 路径 |
| STRATA_PF_FUSED_NATIVE_EXACT_SKIP | 1，保留已支持的 exact 路径；本版通过 GATEUP_EXACT_IQ2 扩展到全部 48 层 |
| STRATA_SEL_BATCH | 不显式设置，使用引擎按缓冲尺寸选择的默认值；不是会话 batch |

本版（rc.6）在基础优化环境上启用 GEMM_Y_F16=qkv\|qf\|sg\|su\|sd\|z、TOPK_H16=1、TOPK_REG_MAX=7168、SCORES_MMAC=2、SCORES_MMAC_LDS=1、SCORES_MMAC_NB=2、SEL_SCORES_FP16=1、GATEUP_EXACT_IQ2=1、MMVQ_WV2_LDS=1、MMVQ_WV2_GROUP=1 和 Q6_WAVE_PACK_LDS=1。`STRATA_SEL_BATCH` 不覆盖；`STRATA_TOPK_H16_T` 已测为显著回归，同样不设置（见 [优化环境说明](OPTIMIZATIONS.md)）。完整环境文件位于 `engines/strata/xxs-20261023-k100ai022/environment.env`；上一版为 `engines/strata/xxs-20261022-sol/environment.env`。其他主要变量如下：

| 范围 | 主要变量 |
| --- | --- |
| PLE / attention | STRATA_PLE_PRELAUNCH=1、STRATA_ATTN_LANECELL=1 |
| Q6 / Q8 | STRATA_Q6_PACKED=1、STRATA_Q6_WAVE_PACK_PREP=1、STRATA_Q6_WAVE_PACK=1、STRATA_Q8_PACKED=1；rc.6 增加 STRATA_Q6_WAVE_PACK_LDS=1 |
| GR / GDN | STRATA_GR_DOWN_PREFETCH=2、STRATA_HC_SPLIT=2、STRATA_GDN_HEAD=1、STRATA_GDN_NOY=1 |
| MoE | STRATA_MMQ_RESIDENT_SORT_NE=1、STRATA_MMQ_RESIDENT_SORT_NATIVE=1、STRATA_EXPERT_V2=1、STRATA_DOWN_MT=1、STRATA_MMQ_EXACT_ALL=1 |
| Prefill / GEMM | STRATA_HIP_WMMA=1、STRATA_HIP_PROMPT_F16=1、STRATA_PF_PAD_ROCBLAS=1、STRATA_PREFILL_RING=8；rc.6 增加 STRATA_GEMM_Y_F16=qkv\|qf\|sg\|su\|sd\|z（六个投影族的 16 位 Y 累加器） |
| Prefill / GEMM（rc.7） | 新增 `STRATA_GEMM_SOL=2560x6144=20377,10240x2560=20385`：按形状锁定 rocBLAS Tensile solution；与 `STRATA_DENSE_TN` 互斥 |
| QSA / scores / topk | STRATA_SCORES_MMAC=2、STRATA_SCORES_MMAC_LDS=1、STRATA_SCORES_MMAC_NB=2、STRATA_TOPK_H16=1、STRATA_TOPK_REG_MAX=7168 |
| Decode 稠密 | STRATA_MMVQ_WV2_LDS=1、STRATA_MMVQ_WV2_GROUP=1 |
| 工作区与融合 | STRATA_PF_GATED_IN_REGION=1、STRATA_EMB_REUSE_ACCOUNT=1、STRATA_PF_DMSCRATCH_COND=1、STRATA_PF_FUSED_SQ=1 |
| KV | STRATA_KV_PREFETCH=1 |
| 关闭项 | STRATA_PREFILL_CPU_SHARE=0、STRATA_SH_STREAM=0 |

`STRATA_SH_STREAM=1` 在 gfx928 历史实验中出现过崩溃，维持关闭。其他开关的有效范围取决于量化类型、形状和实际分派。c4 保留 FP16 中间结果策略，数值结果不保证与原版位级相同。

## 可选控制向量

默认不加载控制向量。需要复现带控制向量的参考配置时，在容器中挂载对应文件，并向 `args` 增加：

```json
"--control-vector-scaled", "/extras/vector.gguf:1.0",
"--control-vector-layer-range", "4", "44",
"--cvec-mode", "project",
"--cvec-dir", "per-layer"
```

历史参考文件 SHA-256 为 `59f9ae8b4372107ae79044278a4406bdfe8f8861fe599aa5848515f0d072be38`。控制向量会改变输出行为，与性能参数独立管理。

## 辅助入口

`settings.json`、`deployment.env`、`install.sh`、`start.sh` 和 Compose 为保留的辅助接口。`PROFILE` 可选 `next-iq3s-single` 或 `next-iq3xxs-single`。辅助接口读取 `settings.json`；本页的原生启动读取 `strata.json`，两者修改互不自动同步。建议一次部署固定使用其中一种配置方式。
