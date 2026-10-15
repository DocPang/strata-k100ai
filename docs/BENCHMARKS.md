# 参考成绩

以下为 K100AI/gfx928、64 GiB 显存的已有记录，非跨硬件速度承诺。除注明外，数据来自 2026-10-15 的隔离卡实验。父版 exact+fused 为 `ad743dbf9cc15026758cfceb26e2dd0aeb25700b295fba028e64ca4b33bc18c0`；本次 `5d87f09e…` 增加 selection 参数的构建不冒用父版的测量身份。

## 权重与部署对照

同一 HCU4 顺序运行，max-context=262144、expert-cache=24576、INT8 KV、kv-resident=32768、prefill=20480。IQ3_S 使用其参考引擎及 spec=4，XXS 使用 exact+fused 及 spec=6；属于各自部署配方对照，不是仅替换单个内核的实验。

| 指标 | IQ3_S | IQ3_XXS + exact/fused | XXS 相对变化 |
| --- | ---: | ---: | ---: |
| 专家原始数据 | 46.8384 GiB | 39.9658 GiB | −6.8726 GiB |
| 加载后总显存 | 60.71 GiB | 52.46 GiB | −8.25 GiB |
| Decode，8192 入 / 512 出 | 62.5 tok/s | 60.3 tok/s | −3.5% |
| Prefill，8192 token | 1388.5 tok/s | 1412.9 tok/s | +1.8% |
| Prefill，32768 token | 1598.4 tok/s | 1549.8 tok/s | −3.0% |
| Prefill，131072 token | 1235.3 tok/s | 1239.9 tok/s | +0.4% |
| Prefill，261120 token | 745.5 tok/s | 731.7 tok/s | −1.8% |

Decode 为报告记录的三次中位结果。总显存与专家原始数据是不同口径；不能将 −8.25 GiB 全部解释为专家压缩。

## exact/fused 改动

同一 XXS 权重及同卡条件下：

| 输入长度 / 项目 | c4 基线 | 全 fused | exact+fused |
| --- | ---: | ---: | ---: |
| Prefill 8192 | 1239.2 tok/s | 1286.7 tok/s | 1412.9 tok/s |
| Prefill 32768 | 1501.1 tok/s | 1503.2 tok/s | 1549.8 tok/s |
| Prefill 131072 | 1223.9 tok/s | — | 1239.9 tok/s |
| Decode 8192/512 | 60.6 tok/s | — | 60.3 tok/s |

父版 exact-skip 保留 29 层 exact gate/up，仅让 19 层 IQ2_XS/IQ2_XXS 使用 fused native int8。128、2048、8192、16384 输入的四组提示在**同一 XXS 权重的新旧引擎之间**贪心输出 SHA 一致。此结果不表示 IQ3_S 与 XXS 输出相同，也不等于完整质量评估。

## 全 KV 驻留

XXS + `ad743dbf…`，256K 最大上下文、spec=6：

| 指标 | kv-resident=32768 | 省略 kv-resident |
| --- | ---: | ---: |
| 加载后空闲显存 | 10946 MiB | 8186 MiB |
| Decode，8192/512 | 60.3 tok/s | 61.0 tok/s |
| Prefill，8192 | 1412.9 tok/s | 1423.8 tok/s |
| 该组输出 SHA | 相同 | 相同 |

显存差为 2760 MiB（约 2.70 GiB）。这是历史实验快照；新 selection 缓冲会增加工作区，实际剩余空间以当前加载和长输入峰值为准。

## 单卡 batch 与 FIFO

父版 XXS 引擎、INT8 KV。不同上下文和测试程序分别列出，总吞吐与每路生成速率不混用。

| 条件 | 配置 | 单请求或单路 | 并发总吞吐 / 现象 |
| --- | --- | ---: | --- |
| 256K | 单槽 | 61.0 tok/s 单请求 | 请求 FIFO 排队 |
| 256K | batch=4 | 33.1 tok/s 单请求 | 4 客户端总计 19.9 tok/s，空闲显存 0 |
| 256K | batch=2 + batch-mtp | 55.9 tok/s 单请求 | 2 客户端总计 30.3 tok/s，空闲 54 MiB |
| 256K | batch=4 + batch-mtp + auto 专家缓存 | 26.7 tok/s 单请求 | 4 客户端总计 17.7 tok/s；专家槽降至 13961 |
| 64K | 单槽 decode 对照 | 55.5 tok/s | 全部专家驻留 |
| 64K | batch=4 + batch-mtp | 约 15 tok/s 每路 | 4 路合计 56.7 tok/s |

256K 下引擎记录每槽约 3.65 GiB。该实现及硬件组合中，batch 未取得足以抵消槽位和调度成本的收益，因此发布配置使用单槽 FIFO。多卡使用独立实例和网关分发，聚合吞吐需按实际请求负载测量。

## QSA selection 参数

来源：STATUS §186，build #57 `d0c643af…`，ctx128、prefill=30602、kv-resident=32768，带相位计时；每臂两次 full-prefill，前后参考共四次。

| 项目 | sel_batch=256 参考 | sel_batch=1024 | 变化 |
| --- | ---: | ---: | ---: |
| 总 prefill 耗时 | 83.5188 s | 82.3925 s | −1.35% |
| topk 相位（两遍累计） | 约 13096 ms | 9506.4 ms | −27.4% |
| 输出 SHA | `18d773e6bcdc1d47…` | 相同 | 该组一致 |

该结果不是 XXS、全 KV 驻留、prefill20480 的新发布构建验收。`STRATA_SEL_BATCH` 控制每次 selection 处理的 query 数，与会话 `--batch` 无关。

后续 §187 同批扫描表明选择批次并非越大越好：

| 测试 | sel_batch=256 | sel_batch=4096 | 结论 |
| --- | ---: | ---: | --- |
| 约 128K（126856 token） | 83.1816 s，参考四样本均值 | 81.5953 s | 较宽 selection 有收益 |
| 约 250K（249986 token） | 185.636 s | 202.634 s | 更大工作区引发竞争，结果反转 |

后续实验的权重、缓冲与内核状态与 HCU0 发布配方不同，不能直接推广。当前配置按指定使用 1024；需要回到父版 selection 宽度时，设置 `STRATA_SEL_BATCH=256`。1024 在新 XXS 全驻留配方上的最终收益仍需独立 A/B。

## 晚间长上下文引擎（build #63，2026-10-16 00:02）

引擎 SHA-256：`ecd44f1c900c700032d6bff48b265ee3daebffa0c6779f7cf5f33361773e6151`。已对照服务器上的二进制摘要、构建记录和 `h5tiers2.run.log`。

测试使用 **IQ3_S**、HCU5、max-context=262144、expert-cache=24576、INT8 KV、kv-resident=32768、prefill=30602、spec=4；每档两次完整 prefill，日志记录 REUSED=0、fallbacks=0。这组结果与本项目 IQ3_XXS 全 KV 驻留配方不同。

| 输入档位 | 实际输入 token | prefill 两次耗时 | 平均耗时 | 平均有效吞吐（token / 平均秒数） |
| --- | ---: | ---: | ---: | ---: |
| 32K | 30603 | 16.5741 / 16.2848 s | 16.4295 s | 1862.7 tok/s |
| 64K | 61326 | 36.0554 / 35.7367 s | 35.8961 s | 1708.4 tok/s |
| 128K | 126857 | 82.2487 / 81.9773 s | 82.1130 s | 1544.9 tok/s |
| 250K | 249986 | 182.8205 / 182.3684 s | 182.5945 s | 1369.1 tok/s |

历史生产记录为约 255K 输入、363.4 s。与本轮 249986 token、182.6 s 相比，耗时下降约 49.8%；**输入、引擎和部署配置不同，该数字只作历史参考，不是同负载 A/B 的两倍加速结论**。

新版累计包含 prompt 工作区与分块调整、QSA scores 网格轴序调整、topk 访问与 selection 并行度调整，以及 attention K 数据的合并暂存。不能把累计改善全部归因于 `STRATA_SEL_BATCH=1024`。

可直接归因的最新同批 A/B 为 build #62 的 K 暂存开关：

| 输入档位 | KSTAGE=0 | KSTAGE=1 | 整模耗时变化 | attention 相位变化 |
| --- | ---: | ---: | ---: | ---: |
| 128K | 84.2901 s | 81.9903 s | −2.73% | −21.0% |
| 250K | 187.0160 s | 183.0941 s | −2.10% | −20.1% |

两档各自的完整生成 token SHA 与对照一致。build #63 将该开关默认启用（`STRATA_PA_KSTAGE=0` 回退）；MMAC scores 仍默认关闭，250K 上已有变慢记录。selection 默认按最大上下文缓冲尺寸分档：`max_blocks <= 32770` 用 4096，否则 256，环境变量可覆盖；它不是按每条请求实际长度动态选择。

正确性范围：32K/64K/128K 的固定答案检查通过。250K 日志的 `retrieval_answer_match=False`，该检查沿用另一条提示的固定答案，本轮通过 `STRATA_AB_SKIP_RETRIEVAL=1` 跳过；该档有新旧生成 SHA 一致记录，但尚不能表述为独立的 250K 检索答案验收通过。以上各档只生成 16 token，不构成长输出 decode 或稳定性验收。

IQ3_XXS 另有 build #60（`4cde924e…`）的迁移记录：同 HCU4、全 KV 驻留、prefill=20480 下，8K 从 1423.8 降至 1272.5 tok/s（−10.6%），32K 从 1549.8 降至 1494.9（−3.5%），131K 从 1239.9 升至 1310.5（+5.7%），decode 60.3 tok/s 持平。该轮配置的短长输入收益方向不同；build #63 尚缺同条件的 XXS 全档验证，不能将上表作为 XXS 的发布成绩。

## 其他已有检查

- 父版服务：健康检查、SSE 完成标记、工具调用及结果回填通过。
- 父版 XXS 缓存：cold → identical → continued → other-conversation → restore-parked 五项通过。
- 小规模质量检查：IQ3_S 与 XXS 的四项客观题均为 4/4；规模不足以建立通用质量等价结论。
- 1M 试验可启动短请求，但仅余 38 MiB 且 prefill 从 20480 降至 6144；GGUF 原生 context_length=262144，发布配置采用 256K。

记录来源：`REPORT_IQ3XXS_20261015.md` §§2、6、8，`KV_AND_CONCURRENCY_20261015.md`，`STATUS_20261013.md` §§178–191；build #60 的 XXS 迁移记录来自对应验证任务。本文保留测量条件和汇总，不发布内部接口、凭据或原始对话。
