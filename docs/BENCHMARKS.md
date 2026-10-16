# 参考成绩

本文汇总 K100AI（gfx928、64 GiB 显存）的既有测试结果。除注明外，测试日期为 2026-10-15。不同引擎、权重或输入内容的结果分别列出，不据此推导发布附件的性能。

## v0.1.0-rc.5：IQ3_XXS 固定引擎实测

本版本运行包分发的二进制为 `d87a4678a21b3c482658e53e6e766f7e54c2033853e6367bbf39e8e088ca4d6b`，构建于 2026-10-16 01:51（UTC+08:00）。以下成绩直接对应该二进制，来自同批 A/B/A 测试；发布打包期间未重新运行 GPU 实验。

测试配置：IQ3_XXS、单张 K100AI、max-context=262144、expert-cache=24576、INT8 KV、不传 kv-resident、prefill=30602、spec=4、spec-min-p=0.70、vram-reserve-mib=2048。关闭 prompt/conversation cache；每臂两次完整 prefill。参考臂与优化臂使用同一二进制和输入，优化臂增加下列五个变量：

```ini
STRATA_SCORES_MMAC=2
STRATA_SCORES_MMAC_LDS=1
STRATA_SCORES_MMAC_NB=1
STRATA_SEL_SCORES_FP16=1
STRATA_GATEUP_EXACT_IQ2=1
```

| 实际输入 token | 优化臂 prefill 两个样本 | 平均耗时 | 有效吞吐 | 相对同批参考配置的耗时下降 |
| --- | --- | ---: | ---: | ---: |
| 30603 | 15.8794 / 15.5411 s | 15.7103 s | 1948.0 tok/s | 约 11.5% |
| 126857 | 70.8586 / 70.7970 s | 70.8278 s | 1791.1 tok/s | 约 17.8% |
| 249986 | 147.0392 / 146.5480 s | 146.7936 s | 1703.0 tok/s | 约 23.3% |

250K 第二批优化臂为 147.0584 / 146.6623 s，平均 146.8604 s。将 chunk 降至 20480 的独立记录为 149.7275 / 149.7749 s，平均 149.7512 s。

Decode 单独测试使用 32K prefill 后生成 128 token。`STRATA_MMVQ_WV2_LDS=1` 的 A/B/A/B 检查中，窗口平均耗时由 41.98 ms 降到 41.04 ms（约 −2.2%）；优化样本生成速率为 77.33 / 81.94 / 77.65 / 82.23 tok/s。此负载与历史 8192 入、512 出不同，不能据此推导 decode 从 60 提高到 82 tok/s。

正确性范围：250K 对照与复测的输出 SHA 相同，前缀 `08d2c11d70446fe6`。32K/128K 优化臂前缀为 `84f393bfc11ffe91`，已有一个 EOS 后 token 差异记录。测试为固定提示家族、16-token 长输入输出，不等于完整质量评测。FP16 scores 与纯 FP32 打分并非所有中间结果逐位相同。

[公开测试结果汇总](../engines/strata/xxs-20261016/benchmarks.json)保留输入长度、样本耗时和程序摘要，不包含内部端点、凭据或完整提示。后续实验构建不属于本发布版本。

## 引擎标识与发布关系

当前发布引擎见上节。下表为历史结果使用的程序，使用二进制 SHA-256 前 12 位标识引擎；完整摘要见本页各节或构建清单。

| 引擎 | 标识 | 用途与发布关系 |
| --- | --- | --- |
| c4 基线引擎 | `c4be512a7d25` | rc.1—rc.3 的运行程序；rc.4 保留该运行包供 IQ3_S 配置使用 |
| XXS exact+fused 基准引擎 | `ad743dbf9cc1` | 已完成 XXS 测试的对照程序；下文的混合计算及全 KV 驻留成绩来自该程序 |
| rc.4 历史发布引擎 | `5d87f09e07bb` | 在 XXS 基准引擎上增加 selection 参数及配套显存记账；尚无该二进制的独立 GPU 性能测试结果 |
| 长上下文实验引擎 | `ecd44f1c900c` | 2026-10-16 构建，累计包含 QSA selection 与 attention 等优化；未作为 rc.4 运行包发布 |

## 历史权重与部署对照

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

XXS exact+fused 基准引擎的 exact-skip 保留 29 层 exact gate/up，仅让 19 层 IQ2_XS/IQ2_XXS 使用 fused native int8。128、2048、8192、16384 输入的四组提示在**同一 XXS 权重的新旧引擎之间**贪心输出 SHA 一致。此结果不表示 IQ3_S 与 XXS 输出相同，也不等于完整质量评估。

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

XXS exact+fused 基准引擎（`ad743dbf9cc1`）、INT8 KV。不同上下文和测试程序分别列出，总吞吐与每路生成速率不混用。

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

来源：STATUS §186，selection 参数实验引擎 `d0c643af96a9`，ctx128、prefill=30602、kv-resident=32768，带相位计时；每臂两次 full-prefill，前后参考共四次。

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

上述 IQ3_S 实验与 rc.4 的 XXS 全 KV 驻留配置不同。rc.4 显式设置 1024；设置 `STRATA_SEL_BATCH=256` 可使用与 XXS 基准引擎相同的批次。1024 在 rc.4 发布配置中的性能影响尚未独立测试。

## 长上下文实验结果（2026-10-16，`ecd44f1c900c`）

构建时间：2026-10-16 00:02（UTC+08:00）。引擎 SHA-256：`ecd44f1c900c700032d6bff48b265ee3daebffa0c6779f7cf5f33361773e6151`。已对照服务器上的二进制摘要、构建记录和 `h5tiers2.run.log`。

测试使用 **IQ3_S**、HCU5、max-context=262144、expert-cache=24576、INT8 KV、kv-resident=32768、prefill=30602、spec=4；每档两次完整 prefill，日志记录 REUSED=0、fallbacks=0。这组结果与本项目 IQ3_XXS 全 KV 驻留配方不同。

| 输入档位 | 实际输入 token | prefill 两次耗时 | 平均耗时 | 平均有效吞吐（token / 平均秒数） |
| --- | ---: | ---: | ---: | ---: |
| 32K | 30603 | 16.5741 / 16.2848 s | 16.4295 s | 1862.7 tok/s |
| 64K | 61326 | 36.0554 / 35.7367 s | 35.8961 s | 1708.4 tok/s |
| 128K | 126857 | 82.2487 / 81.9773 s | 82.1130 s | 1544.9 tok/s |
| 250K | 249986 | 182.8205 / 182.3684 s | 182.5945 s | 1369.1 tok/s |

历史生产记录为约 255K 输入、363.4 s。与本轮 249986 token、182.6 s 相比，耗时下降约 49.8%；**输入、引擎和部署配置不同，该数字只作历史参考，不是同负载 A/B 的两倍加速结论**。

该实验引擎累计包含 prompt 工作区与分块调整、QSA scores 网格轴序调整、topk 访问与 selection 并行度调整，以及 attention K 数据的合并暂存。不能把累计改善全部归因于 `STRATA_SEL_BATCH=1024`。

使用对照引擎 `6e5c89cf`，在同一批测试中仅改变 K 暂存开关，结果如下：

| 输入档位 | KSTAGE=0 | KSTAGE=1 | 整模耗时变化 | attention 相位变化 |
| --- | ---: | ---: | ---: | ---: |
| 128K | 84.2901 s | 81.9903 s | −2.73% | −21.0% |
| 250K | 187.0160 s | 183.0941 s | −2.10% | −20.1% |

两档各自的完整生成 token SHA 与对照一致。长上下文实验引擎 `ecd44f1c900c` 将该开关默认启用（`STRATA_PA_KSTAGE=0` 回退）；MMAC scores 仍默认关闭，250K 上已有变慢记录。selection 默认按最大上下文缓冲尺寸分档：`max_blocks <= 32770` 用 4096，否则 256，环境变量可覆盖；它不是按每条请求实际长度动态选择。

正确性范围：32K/64K/128K 的固定答案检查通过。250K 日志的 `retrieval_answer_match=False`，该检查沿用另一条提示的固定答案，测试设置 `STRATA_AB_SKIP_RETRIEVAL=1` 跳过该检查；该档有新旧生成 SHA 一致记录，但尚不能表述为独立的 250K 检索答案验收通过。以上各档只生成 16 token，不构成长输出 decode 或稳定性验收。

较早的长上下文实验引擎（`4cde924efb21`）另有 IQ3_XXS 测试记录：同 HCU4、全 KV 驻留、prefill=20480 下，8K 从 1423.8 降至 1272.5 tok/s（−10.6%），32K 从 1549.8 降至 1494.9（−3.5%），131K 从 1239.9 升至 1310.5（+5.7%），decode 60.3 tok/s 持平。该轮配置的短长输入收益方向不同；长上下文实验引擎 `ecd44f1c900c` 尚未完成相同条件下的 XXS 全档测试，不能将上表作为 XXS 的发布成绩。

## 其他已有检查

- XXS exact+fused 基准引擎的服务接口：健康检查、SSE 完成标记、工具调用及结果回填通过。
- XXS exact+fused 基准引擎的会话缓存：cold → identical → continued → other-conversation → restore-parked 五项通过。
- 小规模质量检查：IQ3_S 与 XXS 的四项客观题均为 4/4；规模不足以建立通用质量等价结论。
- 1M 试验可启动短请求，但仅余 38 MiB 且 prefill 从 20480 降至 6144；GGUF 原生 context_length=262144，发布配置采用 256K。

记录来源：`REPORT_IQ3XXS_20261015.md` §§2、6、8，`KV_AND_CONCURRENCY_20261015.md`，`STATUS_20261013.md` §§178–191；长上下文实验引擎 `4cde924efb21` 的 XXS 迁移记录来自对应验证任务。本文保留测量条件和汇总，不发布内部接口、凭据或原始对话。
