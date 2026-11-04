# 第三轮：32K indexer 分段与 packed-key stride

本轮只研究已提交 MMAC928 候选上的剩余热点，不改变权重、专家数、路由规则或
已有的四卡组合。所有新路径继续由环境变量显式开启；没有收益或有回归的实验已
撤回。

## 结论

32K 的 `compressor + indexer` 原先约 `158.6 µs/token`，本轮拆成：

| 子阶段 | 四卡墙钟等效 µs/token | 占子阶段 |
|---|---:|---:|
| q/kv + compressor | 8.3 | 5.1% |
| indexer dense projections | 16.6 | 10.3% |
| indexer scoring | 125.5 | 77.8% |
| indexer top-k | 10.9 | 6.8% |
| 合计 | 161.3 | 100% |

因此优化目标是 indexer scoring，top-k 和 projection 不是主要瓶颈。

保留的 exact 改动只调整 QSA shared-memory row stride：

```text
STRATA_DS4_QSA_MMAC928_LD=132
```

默认为原来的 136。该参数只改变 shared 数组布局，不改变 MMAC、ReLU、head
加权或输出顺序。

## 32-query packed 微基准

测试使用真实 chunk 的 32-query slab，单个调用覆盖 16,385 个 key、32 个 head、
128 个维度。每个输出字节都与 BF16-key 路径一致，CPU 最坏相对误差
`1.091e-6`。

| stride | BF16 keys 三次 µs | packed keys 三次 µs |
|---|---:|---:|
| 130 | 308.2 | 258.3 |
| 132 | 282.9, 283.3, 283.4 | 235.5, 235.7, 236.4 |
| 134 | 307.1 | 258.8 |
| 136 | 295.6, 294.3, 294.7 | 243.1, 243.5, 243.6 |
| 144 | 347.9 | 298.1 |

132 同时满足 8-byte shared 向量对齐和较优的 bank 分布。最终测试为 packed
`235.2 µs`，全部 `ALL OK (0 failed)`。

按单次 packed 调用
`32 × 16,385 × 32 × 128 = 2.147G MAC`、即 `4.295 GFLOP` 计算，最终约
`18.3 TFLOP/s`。K100AI gfx928 的 BF16 标称峰值为 `196 TFLOP/s`，这条
微基准达到约 9%。这个数字只说明矩阵部分仍有空间；真实 indexer 还包含按
query 增长的三角前缀、mask、key 解码和 head reduction，不能直接把 196
当作端到端上限。

## 32K 整模型 A/B

同一最终二进制、同一四卡配置、同一 `mixed-32k` 提示，只切换 LD：

| 运行 | LD136 ms | LD132 ms |
|---|---:|---:|
| 1 | 38392.7 | 38366.1 |
| 2 | 38320.0 | 38318.8 |
| 3 | 38352.2 | 38330.7 |
| 中位 | 38352.2 | 38330.7 |

三次均 `cached_tokens_max=0`。中位改善 `21.5 ms`（`0.056%`）。

profile 中的 indexer scoring：

- LD136：`16459.4 / 16458.1 / 16457.6 ms`
- LD132：`16430.7 / 16437.5 / 16438.4 ms`

微基准收益没有按比例传到整模型，原因是 scoring 只占 32K prefill 约 10.7%，
且真实前缀较短、各阶段和设备等待会稀释 lane 级 bank-conflict 改善。收益较小
但方向稳定，故作为显式 candidate 配置保留；没有把它描述成 headline 加速。

## 失败或撤回的实验

| 实验 | 结果 | 处理 |
|---|---|---|
| 两个 16-query 组用 512-thread CTA | packed `405.6 µs`，8-wave CTA 占用退化 | 撤回 |
| 256-thread CTA 串行两组 query | packed `298.1/296.8/298.1 µs` | 撤回 |
| query tile 直接走 global/L1 | 数值错误，packed `529.3 µs` | 撤回 |
| 去掉 query 低位 split | packed `227.0 µs`，最坏相对误差 `7.416e-4` | 超 `1e-4`，撤回 |
| 两 head compact shared staging | packed `698.3/683.3/682.5 µs` | bank conflict/占用回归，撤回 |
| E2M1 scaled BF16 LUT | 逐位正确，但 packed `259.3 µs` | 无收益，撤回 |
| head 权重 shared staging | packed `250.5/251.2/251.1 µs` | 撤回 |
| `PSPLIT`、tiling 之外的近似 top-k | 未进入实现 | 缺少质量证据，未冒险 |

## 质量

LD132 是 exact layout 变化。最终固定 API 任务全部通过，并与已保存候选的
`content`、`reasoning`、`finish_reason`、`usage` 逐字相同：

- `france_capital_zh`：`巴黎。`
- `arithmetic`：`391`
- `python_add`：`def add(a, b):\n    return a + b`
- `unit_conversion_zh`：`1千米等于1000米。`

## 复现

候选：

```text
research/dsv41_opt/candidate-mmac928-dense-attn-config.json
```

关键环境变量：

```text
STRATA_DS4_MMAC928=1
STRATA_DS4_QSA_MMAC928=1
STRATA_DS4_QSA_MMAC928_LD=132
STRATA_DS4_DENSE_MMAC928=1
STRATA_DS4_ATTN_MMAC928=1
STRATA_DS4_ATTN_MMAC928_QSPLIT=1
STRATA_DS4_ATTN_MMAC928_PSPLIT=0
```

最终引擎 SHA-256：
`1b0610416b3cce179983f3e439c2d4dc173122c9dc432b485450107b9116d811`

最终 `ds4_gguf_test` SHA-256：
`6159e5e04a602f0362f23501194d63cce0bc47c2089f0b3be4dd088d486f9ee6`

原始记录：

- `profile-indexer-sample.json`：score/top-k 拆分后的冷 32K 请求
- `profile-split-sample.json`：q/kv、projection、score/top-k 拆分后的冷 32K 请求
- `round-003-summary.json`：本轮 profile、微基准、A/B 和最终哈希摘要
- `ld136-mixed32k.json`、`ld132-mixed32k.json`：各三次整模型 A/B
- `final-qsa-ld132-test.log`：最终算子测试
- `quality-api-final-ld132.json`：最终固定任务响应

## 边界

- 本轮整模型数字只覆盖 32K mixed；4K/8K 没有重新跑 A/B。
- LD132 是 0.056% 级别的局部优化，不足以单独宣称新的端到端倍数。
- 64K 以上 candidate-restricted/chunked attention 和 1M context 仍未验证。
- indexer scoring 仍是下一轮最大的单点；后续应优先研究能在不降低 CTA
  占用率的前提下跨 key tile 复用 query，或给出有完整 PPL/任务门禁的稀疏
  scoring。
