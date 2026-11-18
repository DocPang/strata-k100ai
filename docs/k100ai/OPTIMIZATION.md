# K100AI 优化说明

[返回首页](../../README.md) · [文档索引](README.md) · [验证记录](VALIDATION.md)

下表记录 K100AI 的基础优化路径，包括 gfx928 内核适配、布局与参数
选择。Qwen 当前生产路径是 v0.3.1 IQ3_S，旧 0.1.47 / 0.1.55 路径单独
标注。DeepSeek-V4.1 的 R001–R179 结论见 [DEEPSEEK41.md](DEEPSEEK41.md)。

| 范围 | 本版采用的优化 |
| --- | --- |
| K100AI 平台适配 | gfx928 HIP 兼容层、DTK 2604 编译链接、DTK/rocBLAS/hipBLASLt 接口适配，以及 Strata 与 ggml/llama.cpp MMQ 的衔接 |
| Qwen v0.3.1 prompt attention | QSA prompt MMAC、完整分数选择、确定性 top-k 和完整路由 top-10；不启用专家截断 |
| Qwen v0.3.1 专家计算 | IQ3_S/IQ4_NL mode 8 fused gate/up 与 LDS down，10 个兼容层关闭旧 s26 路径，`STRATA_S26=1` 可回退 |
| Qwen v0.3.1 dense | Q6_K 列不变解码、wave-local LDS、Q8_1 打包与 Q6 packed path |
| Qwen v0.3.1 状态上传 | `--no-publish-kernel`，把逐元素 mapped-host 读取改为 pinned host 异步 DMA |
| Prompt GEMM | F16 prompt GEMM、按形状选择 GEMM tile、rocBLAS stride padding，以及已验证形状的 Tensile solution |
| Hyper-connection | gfx928 HC upmix、split/staged read、down prefetch 和单 LDS plane 分块，减少重复读取与分界 launch |
| GDN 与路由 | 八 lane GDN 递归、关闭无消费者的 FP32 写回、常驻 expert 排序，以及 native/MMQ/fused 路径协同 |
| 数据复用 | 每个唯一 token 行只量化一次，再按 expert 索引 gather；融合 SwiGLU、量化和共享工作区 |
| Qwen v0.3.1 MTP | 外置 Q2_0 runtime，`spec=3`、min-p 0.5、window 4096、Q4 all，目标模型验证后提交 |
| Qwen 旧引擎 | 0.1.47 IQ3_XXS 保留 prompt MMAC、`spec=6` 和数值漂移快速档；0.1.55 保留 canonical/A-B 对照 |
| DeepSeek 四卡 | 每卡 10 层的 `10,10,10,10` 切分，全部 routed experts 驻留 HCU，1024-token prefill chunk |
| DeepSeek KV 与 Engram | 1M-token FP4 KV、resident KV、2 层 Engram 和 48 次 lookup |
| DeepSeek drafting | DSpark Q8_0 drafter，置信度 0.65；最终 profile 记录 CPU/RAM expert lane 为 0 |
| DeepSeek gfx928 研究路径 | 显式开启的 prompt expert、dense、attention MMAC 与 QSA packed layout；精确路径保留为回退 |
| 服务运行时 | OpenAI 与 Anthropic 兼容 API、健康检查、Prometheus metrics、MTP/DSpark draft 计数 |
| 安全启动 | Qwen 只映射一张 HCU，DeepSeek 只映射四张；宿主端口绑定 127.0.0.1，不自动停止其他容器 |
| 可复现发布 | 固定容器 digest、固定模型 revision、逐文件 SHA-256、canonical `/work/...` 构建路径和确定性 tar/gzip |

v0.3.1 的 IQ3_S 路径固定 65,536-token KV 驻留、24,576 expert slots
和 `--suffix-draft 0`。32K 输出为 Native 45.8 tok/s、MTP 中位
70.0 tok/s，Prefill 中位 2,851.65 tok/s。实现和限制见
[Qwen v0.3.1 指南](QWEN31.md)。

历史 0.1.55 对照档仍有 gfx928 prompt attention 性能回归，单卡 prompt
比旧 0.1.47 快速档慢约 1.8 倍。该结论只描述历史兼容路径。
