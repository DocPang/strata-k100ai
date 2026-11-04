# DeepSeek-V4.1-Flash K100AI 2x 候选

本轮在 4 张 Hygon K100AI（gfx928、DTK 2604）上，以原有的 IQ2/IQ1_M
权重、四卡布局、每 token 6 个路由专家和路由规则不变为前提，完成了一个
默认关闭、可逐项启用的 MMAC 预填候选。HCU0–3 用于本任务，HCU4–7 未改动。

## 最终结果

固定基线与最终候选使用相同的 mixed prompts、`max_tokens=32`、冷提示和
零复用条件：

| Prompt | 固定基线 | 最终候选 | 端到端加速 |
|---|---:|---:|---:|
| mixed-4K | 14.378 s | 5.8299 s | 2.47x |
| mixed-8K | 24.833 s | 9.6721 s | 2.57x |
| mixed-32K | 90.887 s | 38.2866 s | 2.37x |

九组提示各跑三次，所有运行的 `cached_tokens_max` 均为 0：

| 长度 | 中文中位 | 代码中位 | mixed 中位 |
|---|---:|---:|---:|
| 4K | 5.532 s | 5.579 s | 5.786 s |
| 8K | 9.352 s | 9.397 s | 9.673 s |
| 32K | 37.246 s | 37.351 s | 38.207 s |

最终二进制 SHA-256：

```text
c0b3817d5029d9fef3de4edbb4083f570fac213c5c37b33d64130893db0bfe11
```

## 候选路径

所有实验路径默认关闭，旧实现仍是默认路径。

- `STRATA_DS4_MMAC928=1`：IQ1_M、IQ2_XXS、IQ2_XS 的 16x16x16 BF16
  专家 MMAC；权重 tile 使用 64+8 padding。
- `STRATA_DS4_QSA_MMAC928=1`：32-head indexer，BF16 key 和 query
  高低拆分；64-key tile 会在 32 个 head 之间复用。
- `STRATA_DS4_DENSE_MMAC928=1`：F16 64x64 dense GEMM；A/B 行跨度使用
  64+8 padding。
- `STRATA_DS4_ATTN_MMAC928=1`：16-head、16-key 分块的稀疏注意力。
  四个 wave 分别计算 128 个 QK 维，并负责 128 个输出维。query 默认使用
  高低拆分，概率值保持内存中的 BF16 路径。
- candidate 服务配置：
  [`candidate-mmac928-dense-attn-config.json`](candidate-mmac928-dense-attn-config.json)

## Profile

32K mixed prompt 的四卡墙钟等价值：

| 阶段 | dense+attention | 再加入专家 padding |
|---|---:|---:|
| experts，含等待 | 614.7 us/token | 344.5 us/token |
| attn rest | 181.0 us/token | 181.0 us/token |
| compressor + indexer | 154.8 us/token | 158.6 us/token |
| between layers | 81.2 us/token | 76.4 us/token |
| sparse attention | 74.9 us/token | 74.9 us/token |
| q_b + rope | 51.2 us/token | 51.2 us/token |
| shared expert | 31.7 us/token | 31.8 us/token |
| 总 prefill | 1,464.0 us/token | 1,163.3 us/token |

相对于只用原始专家、原始 QSA 时的 32K 阶段，sparse attention 从
164.7 降到 74.9 us/token；dense 相关阶段合计从约 449 us/token
降到约 316 us/token；专家则从 614.7 降到 344.5 us/token。

## 质量门

teacher-forced 质量集由模型自带 tokenizer 生成，5 个 segment 各 999
个计分位置。同一二进制分别关闭和开启全部实验路径：

| Segment | legacy PPL / top-1 | candidate PPL / top-1 |
|---|---:|---:|
| code | 4.5317 / 75.58% | 4.5317 / 75.58% |
| docs | 23.3337 / 46.25% | 23.3337 / 46.25% |
| English | 5.2836 / 55.76% | 5.2836 / 55.76% |
| multilingual | 8.1017 / 69.67% | 8.1017 / 69.67% |
| agent | 2.8700 / 80.28% | 2.8700 / 80.28% |
| 全部 4,995 位置 | 6.649 / - | 6.649 / - |

64-token log-prob 的逐位置比较结果：

```text
positions 4995
top-1    100.00%
mean KL  0.00000
PPL      6.649
PPL ref  6.649
```

四个固定 API 任务（法国首都、17x23、Python add、1 千米换算）在 legacy
与 candidate 上都通过，最终 `content` 逐字相同。HTTP 非流式、流式、
usage、缓存和 metrics 也通过；流式请求 `cached_tokens=0`。

## 算子验证

- `ds4_gguf_test --kv`：`ALL OK`。sparse attention 的 rel L2 为
  1.545e-4（16 queries）和 1.095e-4（256 queries）；QSA 的最坏相对
  误差为 9.671e-7。
- `ds4_iq_test --devices 0 --type 29,16,17`：`ALL OK`。IQ1_M、
  IQ2_XXS、IQ2_XS 的 gate/up 和 down 路径均通过原有容差。
- `dense_gemm_mmac_test`：所有边界/累加测试通过；T=1024 的两个真实
  形状达到 13.17 和 13.75 TMAC/s。
- 专家 32-row、48-expert 微基准相对标量路径：IQ1_M
  4.5x/4.1x，IQ2_XXS 3.8x/3.9x，IQ2_XS 3.1x/3.6x
  （依次为 gate/up、down）。
- 64-head 算子微基准中的 sparse attention 与标量路径持平；真实模型每卡
  16 head 时，32K profile 的该阶段减少约 55%。因此该路径的收益按真实
  模型 profile 和端到端结果保留，不按 64-head 微基准外推。

## 边界

- 完整九提示三次矩阵在本轮只有 candidate。固定 baseline 的逐提示矩阵
  尚未采集，2x 结论的跨类别基线比较仍以 mixed 三档为准。
- sparse attention 的 candidate-restricted 和 chunked 路径从 64K 起可能
  仍回退到标量实现；1M context 尚未做本轮质量与速度验证。
- 本轮没有 push、publish 或修改既有 v0.2.2 release。
