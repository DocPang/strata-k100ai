# K100AI 优化说明

[返回首页](../../README.md) · [验证记录](VALIDATION.md)


下表列出 v0.2.2 的运行路径，包括 gfx928 内核适配、布局与参数选择。性能结果见 [验证记录](VALIDATION.md)。

| 范围 | 本版采用的优化 |
| --- | --- |
| K100AI 平台适配 | gfx928 HIP 兼容层、DTK 2604 编译链接、DTK/rocBLAS/hipBLASLt 接口适配，以及 Strata 与 ggml/llama.cpp MMQ 的衔接 |
| Qwen prompt attention | 默认 0.1.47 使用 gfx928 native prompt MMAC；当前 0.1.55 暂未恢复这条路径，日志会明确报告不可用 |
| gate/up 专家计算 | fused gate/up、exact IQ2 展开、wsg16 tile 分派；0.1.55 的安全 exact 集合覆盖 IQ3_XXS、IQ3_S、IQ4_XS |
| down 专家计算 | Q2_0 展开与 int8 MMAC、IQ4_NL exact pre-expansion、staging 和行 tile 调整 |
| Prompt GEMM | F16 prompt GEMM、按形状选择 GEMM tile、rocBLAS stride padding，以及已验证形状的 Tensile solution |
| Hyper-connection | gfx928 HC upmix、split/staged read、down prefetch 和单 LDS plane 分块，减少重复读取与分界 launch |
| GDN 与路由 | 八 lane GDN 递归、关闭无消费者的 FP32 写回、常驻 expert 排序，以及 native/MMQ/fused 路径协同 |
| 数据复用 | 每个唯一 token 行只量化一次，再按 expert 索引 gather；融合 SwiGLU、量化和共享工作区 |
| MTP | 加载外置 Q2_0 MTP runtime，使用 `spec=6`、接受阈值 0.7、draft graph 预热和会话缓存 |
| DeepSeek 四卡 | 每卡 10 层的 `10,10,10,10` 切分，全部 routed experts 驻留 HCU，1024-token prefill chunk |
| DeepSeek KV 与 Engram | 1M-token FP4 KV、resident KV、2 层 Engram 和 48 次 lookup |
| DeepSeek drafting | DSpark Q8_0 drafter，置信度 0.65；最终 profile 记录 CPU/RAM expert lane 为 0 |
| 服务运行时 | OpenAI 与 Anthropic 兼容 API、健康检查、Prometheus metrics、MTP/DSpark draft 计数 |
| 安全启动 | Qwen 只映射一张 HCU，DeepSeek 只映射四张；宿主端口绑定 127.0.0.1，不自动停止其他容器 |
| 可复现发布 | 固定容器 digest、固定模型 revision、逐文件 SHA-256、canonical `/work/...` 构建路径和确定性 tar/gzip |

Qwen 0.1.55 仍有一个明确未完成项：gfx928 prompt attention 没有达到 0.1.47 的原生 MMAC 路径，因此单卡 prompt 比快速档慢约 1.8 倍。
