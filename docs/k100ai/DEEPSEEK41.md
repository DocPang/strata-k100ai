# DeepSeek-V4.1-Flash

[返回首页](../../README.md) · [文档索引](README.md) · [配置目录](../../configs/k100ai/README.md)

本报告冻结 R001-R179 的最终生产结论，对应 `v0.3.0`。实验探索到此结束；
后续发布只使用本报告列出的三套配置和生产二进制。

## 交付基线

生产 clean 二进制：

```text
bin/strata-dsv41
SHA-256 4fd967c1fa6769e60298357d55683c18e5522f6689441b12fe5a97489b3a35d6
```

最终源码以生产安装目录 `/data/dsv41-opt-v022-20261104/src` 为准。发布前已
对账 `src/`、`include/`、`cmake/`、`third_party/`、构建入口和相关运行脚本，
并发现在研究分支提交之外还有生产源修正，因此发布树直接采用远端生产源，
没有仅挑取 `ds4_model.cu`。关键哈希：

| 文件 | SHA-256 |
| --- | --- |
| `src/dsv41/ds4_model.cu` | `fa4644560c1fd3bd7e1aa259791b817a001a9d7ee48b1206ecd5cb228b0a548a` |
| `src/dsv41/ds4_kernels.cu` | `3734325b18913f6a533965cd8b274ff298f15ec676eeb7e07d791c9a4e3daa5a` |
| `src/dsv41/ds4_iq.cu` | `052524a03e65aa0139fe9e7b69bbbaf81c91a3ea78a6ab30f4aabc9b133a5bf8` |
| `src/prefill/wmma_gemm_mmac928.cuh` | `466cb00ec187199fc15970783ddedbfffad299a29e63358414e30ab3c0577458` |
| `src/dsv41/vendor/exl3.cu` | `257a97a4e7fe0d5c37a167fc7a8c4f463db8b381d742b9b8bf0ce69c88aa6e2c` |
| `include/strata/dsv41/exl3.hpp` | `4979877507a7a0ac73ca1b1d5f1e549fdf0fb4cbfaf5a1c3cdf365181a47a60b` |

## 三套生产配置

| 配置 | 设备 | Pack | Prefill chunk | Decode |
| --- | --- | --- | ---: | --- |
| `configs/k100ai/exl3-tp1-prod.json` | HCU0-3，EXL3 TP1 | EXL3 2.9 bpw | 1024 | Native |
| `configs/k100ai/exl3-tp4-prod.json` | HCU0-3，EXL3 TP4 | EXL3 2.9 bpw | 3072 | DSpark |
| `configs/k100ai/gguf-tp2-prod.json` | HCU0-1，GGUF TP2 | GGUF 2.00 bpw | 512 | DSpark |

所有配置只启用 4K、32K 和 Decode 已验证的路径。模型资产、tokenizer、profile
和日志路径可以在 JSON 中按部署位置修改。配置顶部的中文说明字段是发布元数据，
不会传给 `strata-dsv41`。

## 实测结果

### EXL3 四卡

R179 将 R178 的自适应 active-K Prefill 移植到 EXL3 TP1 和 TP4。每组使用
同一二进制进行 `off -> on -> on -> off` 两次冷启动，结果取中位数。

| 拓扑 | 4K Off | 4K On | 32K Off | 32K On | 32K 改善 | 目标 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP1 | 3.090 s | 3.114 s | 18.629 s | **17.953 s** | **3.63%** | `<14.5 s`，未达到 |
| TP4 | 4.094 s | 4.070 s | 30.321 s | **29.352 s** | **3.20%** | `<24 s`，未达到 |

生产安装后的正式 smoke：

| 配置 | 4K TTFT | 32K TTFT | Decode |
| --- | ---: | ---: | ---: |
| EXL3 TP1 | 3.142 s | 17.954 s | Native 18.7 tok/s |
| EXL3 TP4 | 4.042 s | 29.220 s | DSpark 43.9 tok/s |

TP1 的 4K 数字在重复 Decode 批量中位数为 `2.982 s`。单次冷启动 A/B 的
`3.114 s` 和 postinstall 的 `3.142 s` 反映冷启动噪声，发布表使用用户指定的
重复中位数 `2.98 s`。

### GGUF 双卡 TP2

R178 在 TP2 + DSpark 上采用：

- prompt 少于 4608 行时，非尾部、非最终 chunk 使用 active K=2。
- prompt 达到 4608 行时，非尾部、非最终 chunk 使用 active K=1。
- 最后 1024 行和最终 chunk 恢复完整 K。
- QSA candidate top-k=1024 仅用于至少 8192 行的请求。
- Decode、DSpark draft 和 verify 不进入 Prefill 近似路径。

| 指标 | Off | On | 变化 |
| --- | ---: | ---: | ---: |
| 4K TTFT | 8.604 s | **8.110 s** | **5.75%** |
| 32K TTFT | 65.228 s | **52.415 s** | **19.64%** |
| Native Decode | 22.5 tok/s | 22.4 tok/s | `-0.44%`，噪声内 |

DSpark Decode 四组结果为 `37.3 / 36.4 / 46.8 / 46.9 tok/s`，中位数
`42.05 tok/s`。样本呈双峰分布，因此该数值只作为中位数记录，不宣称为稳定
加速。

## 精度与质量

固定质量门为 PPL 相对基线不超过 `+2%`，并执行四题 API 测试。

| 范围 | Baseline PPL | 最终 PPL | 变化 | Top-1 | API |
| --- | ---: | ---: | ---: | ---: | ---: |
| GGUF TP2，window 1024 | 1.741926 | 1.742965 | `+0.060%` | 0.895195 | 4/4 |
| EXL3 TP1，window 1024 | 4.860520 | 4.847801 | `-0.262%` | 0.777489 | 4/4 |
| EXL3 TP4，window 1 | 1.598806 | 1.601385 | `+0.161%` | 0.908845 | 4/4 |

三套配置均通过 `+2%` PPL 门禁。EXL3 PPL 评测覆盖 19,999 positions。TP4 的
window 1 评分约耗时 726 秒。

## R001-R179 收敛记录

R001-R050 建立了 gfx928 MMAC928 dense、attention、QSA、Router 和 expert
路径，并加入 dense weight cache、低行 expert wave、QSA candidate、chunk
曲线和逐算子质量门。大量看似有利的 tile、persistent grid、寄存器压缩和
半精度路径因实际回归被拒绝。

R051-R067 收敛 attention finish/epilogue F16、wo_a 直接输出、四投影 3D
合并、gate/up 分组和 MoE active-K。R068-R090 点亮 EXL3 TP1/TP4：接入 RCCL
全 rank all-reduce，修复 P2P 一致性，融合 all-reduce 与 hc_post，分区 QSA
query，恢复 packed-layout attention，并对 Router、dense 和 prefill chunk
做生产级 A/B。

R091-R137 完成了四卡、双卡和单卡拓扑台账、专家驻留与 CPU lane 画像、并发
曲线、GGUF TP2 专用 profile、chunk 4096/copyram 和 DSpark 配置扫描。
并发 prefill 实际串行，因此没有把短请求的近线性 decode 扩展误写成长 prompt
吞吐。

R138-R161 完成 TP1 与 TP4 的交互延迟攻坚、chunk 1024/512、DSpark 分阶段
剖析、AllReduce 阈值和参数面收敛。多次 `M128`、chunk 极端值、FP32
AllReduce 与关闭 F16 的候选均未超过基线。

R162-R169 继续完成 EXL3 reconstruct、量化和 MMAC928 验证；R169 把双卡
GGUF TP2 的 4K TTFT 降至 8.60 秒并通过干净 A/B。R170-R176 处理 TP1 direct
attention epilogue、TP4 split-KV、plain-step K5 和 dense cache 扩容。R176
的 direct epilogue 原质量验证不完整，R179 已在最终配置下补做 PPL 和 API。

R177-R179 收敛有界近似 Prefill：先验证 active-K 路径，再加入 4608 行短/长
阈值和 1024 行尾部保护，最后移植到 EXL3 TP1/TP4。R179 的实测收益为
`3.63%` 和 `3.20%`，低于用户给出的 `14.5 s` / `24 s` 目标，因此发布文档
保留真实结果。

## 启用参数

R178/R179 的自适应 Prefill 只在下面的六项环境变量显式开启时生效：

```text
STRATA_DS4_MOE_PREFILL_ACTIVE_K=2
STRATA_DS4_MOE_PREFILL_ACTIVE_K_LONG=1
STRATA_DS4_MOE_PREFILL_LONG_MIN_ROWS=4608
STRATA_DS4_QSA_PREFILL_CAND_TOPK=1024
STRATA_DS4_QSA_PREFILL_CAND_MIN_ROWS=8192
STRATA_DS4_MOE_PREFILL_KEEP_TAIL_ROWS=1024
```

不是所有环境下都应启用这些参数。发布配置已经显式包含它们；其他拓扑需要
重新做 PPL、API、4K/32K 和 Decode A/B。

## 启动

启动命令与设备映射见
[`configs/k100ai/README.md`](../../configs/k100ai/README.md)。三套配置可任选一套：

```bash
python3 -m serve.server \
  --engine strata \
  --config configs/k100ai/exl3-tp4-prod.json \
  --host 127.0.0.1 \
  --port 18097 \
  --no-api-key
```

启动前执行设备占用检查，只允许 HCU0-3 / `renderD128-131`。HCU4-7 /
`renderD132-135` 不进入本版本的配置或测试脚本。

## 限制

- TP1 32K 实测 `17.953 s`，没有达到 `<14.5 s`。
- TP4 32K 实测 `29.352 s`，没有达到 `<24 s`。
- DSpark Decode 呈现明显 run-to-run 波动，不能承诺固定收益。
- 自适应 Prefill 是显式配置的有界近似路径；其他模型、量化格式和拓扑没有
  本报告的 PPL/API 覆盖。
- 1M context 可以配置，但完整 1M 窗口尚未做质量评测。
- 本机未重新编译出同 SHA-256 的二进制。Release 资产使用已经过 R178 clean
  rebuild 和 R179 postinstall 验证的生产二进制，并以逐文件源码对账记录
  provenance。
