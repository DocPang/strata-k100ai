# 参数说明

宿主路径、显卡、端口和 API key 在 `deployment.env`；引擎参数在 `profiles/next-iq3s-single/settings.json`。修改后在没有请求时执行 `./stop.sh`，再 `./start.sh`，会重新加载模型。

## 用户可调整的引擎参数

| JSON 项 | 引擎参数 | 默认 | 含义 |
| --- | --- | ---: | --- |
| prefill | --prefill | 20480 | prefill 分块上限；增大可能增加 workspace 与显存压力 |
| max_context | --max-context | 262144 | 最大上下文；不是流畅性或并发承诺 |
| expert_cache | --expert-cache | 24576 | 专家槽数量；此配方以全部专家常驻为目标 |
| kv_resident | --kv-resident | 32768 | INT8 KV 的显存驻留窗口；其余 KV 使用宿主内存 |
| spec | --spec | 4 | MTP 草稿长度；更大不保证更快 |
| spec_min_p | --spec-min-p | 0.70 | 草稿接受控制；收益依赖任务与接受率 |
| vram_reserve_mib | --vram-reserve-mib | 2048 | 显存预留 MiB |
| conversation_cache_mib | --conversation-cache-mib | 16384 | 宿主会话缓存预算 MiB；不是 GPU 并发槽数 |
| conversation_cache_slots | --conversation-cache-slots | 4 | 会话缓存条目数量；不是同时解码请求数 |

prefill 和 kv_resident 不能超过 max_context。降低上下文时同时检查这三项。未知项、无效数值和不完整文件名会拒绝启动，避免参数拼错后悄悄使用默认值。

## 固定参数

完整值见 [profile.json](../profiles/next-iq3s-single/profile.json)。除路径绑定外，包含：

- `--native` GGUF 第一分片，`--ple-gguf` 第二分片，`--ple-io mmap`。
- `--pack /pack`、`--mmap-experts`、冻结的 `--expert-profile`。
- `--kv int8`、`--pool-workers 12`、`--eos-ids 248046`。
- `--mtp /mtp`，MTP 运行时独立准备。
- `--prompt-cache 8`、`--prompt-cache-every 16384`。
- `--conversation-cache-min-free-mib 8192`、`--no-prefill-borrow`。
- 服务默认 temperature=0；具体请求和上游服务支持的采样设置仍可能改变行为。

未加入 `--batch` 或 `--batch-mtp`，此版不声明支持已验证的多请求同时解码。会话缓存槽数与推理并发是不同概念。

## GPU 优化环境

完整环境固定在 profile，不建议普通用户逐个更改：

| 范围 | 主要开关 |
| --- | --- |
| PLE / attention | STRATA_PLE_PRELAUNCH=1、STRATA_ATTN_LANECELL=1 |
| Q6 / Q8 packed | STRATA_Q6_PACKED=1、STRATA_Q6_WAVE_PACK_PREP=1、STRATA_Q6_WAVE_PACK=1、STRATA_Q8_PACKED=1 |
| GR / GDN | STRATA_GR_DOWN_PREFETCH=2、STRATA_HC_SPLIT=2、STRATA_GDN_HEAD=1、STRATA_GDN_NOY=1 |
| MoE | STRATA_MMQ_RESIDENT_SORT_NE=1、STRATA_MMQ_RESIDENT_SORT_NATIVE=1、STRATA_EXPERT_V2=1、STRATA_DOWN_MT=1、STRATA_MMQ_EXACT_ALL=1 |
| Prefill / GEMM | STRATA_HIP_WMMA=1、STRATA_HIP_PROMPT_F16=1、STRATA_PF_PAD_ROCBLAS=1、STRATA_PREFILL_RING=8 |
| 显存 / 融合 | STRATA_PF_GATED_IN_REGION=1、STRATA_EMB_REUSE_ACCOUNT=1、STRATA_PF_DMSCRATCH_COND=1、STRATA_PF_FUSED_SQ=1 |
| KV | STRATA_KV_PREFETCH=1 |
| 已关闭 | STRATA_PREFILL_CPU_SHARE=0、STRATA_SH_STREAM=0 |

`STRATA_SH_STREAM=1` 在历史 gfx928 实验中触发过崩溃，不应开启。部分名称保留上游历史命名，不能据名字判断实际走 CUDA/WMMA 或所有张量均命中；以 gfx928 代码分支和实际类型为准。

保留历史已接受的 FP16 中间结果策略。整套配置不是位级无损方案，特定 exact 路径的证明不能扩展到全部计算。

## 控制向量

默认**不加载**控制向量，避免默认改变模型行为；因此不宣称输出与使用控制向量的历史服务一致。

确实需要复现历史控制向量设置时，用户自行准备文件，将其目录指定为 `EXTRAS_DIR`，设置：

```dotenv
CONTROL_VECTOR=/extras/your-vector.gguf
```

此入口使用 scale=1.0、layers 4..44、project/per-layer。历史参考文件 SHA-256 为 `59f9ae8b4372107ae79044278a4406bdfe8f8861fe599aa5848515f0d072be38`。文件不随项目分发；使用其他文件需自行验证。
