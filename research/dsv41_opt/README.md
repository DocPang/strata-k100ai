# DeepSeek-V4.1-Flash K100AI optimization

This directory is the local research record for the first K100AI DeepSeek
optimization phase. The target is a 2x end-to-end speedup on representative
4K, 8K and 32K prompts without changing the IQ2_XXS weights, the four-card
placement, the expert count or the routing rule.

## Fixed baseline

- Platform: Hygon K100AI, gfx928, DTK 2604
- Devices: HCU0-3 only
- Source content: `strata-v0.2.2` source tree
- Engine source commit: `8544d2fa6cddf61fb47edc0c3bd8a5d7d638140e`
- Binary: `6543311f453bee2c6d61653a24f0b6282f9cb3568b17a8b889c2c5e073b12286`
- Pack: `/data/models/dsv41-2.00/maya-ds4-dspark`
- Expert profile: `data/dsv41/profile-2.00.bin`
- Expert profile SHA-256:
  `8db93e483b34b15dbf63a0826601f1db2107a8cfea73774e7d21d8559e0a9568`
- Split: `DS4_SPLIT=10,10,10,10`
- Context: 1,048,576
- Prefill chunk: 1024
- Experts routed per token: 6
- DSpark: on, confidence 0.65

## Measurement protocol

1. Generate the fixed Chinese, code and mixed prompts with
   `make_prompts.py`. Their raw user-content targets are 4096, 8192 and 32768
   tokens.
2. Start the engine with `STRATA_DS4_REUSE=0`. Every request must report zero
   cached prompt tokens.
3. Discard one short warmup request after the model has loaded.
4. Run every prompt three times. Report medians for:
   - external stream time to first generated token;
   - engine prompt milliseconds and prompt tokens/s;
   - generated decode tokens/s;
   - total request time.
5. Keep `DS4_PROF=1` and `STRATA_PREFILL_TIMING=1` enabled for profiling.
   Uninstrumented numbers are recorded separately if the events materially
   change timing.
6. A candidate is retained only after operator parity, full-model speed and a
   basic API request pass. Approximate paths remain opt-in and need a
   perplexity increase of at most 2% plus fixed task checks.

## 当前进度

精确 release 源码和二进制已在远端复核。HCU0–3 为本任务可用设备，HCU4–7
仍被其他任务占用；所有 GPU 测试只暴露 HCU0–3。

已保存以下基线：

- 原引擎二进制备份：
  `/data/dsv41-opt-v022-20261104/run/baseline-bin/strata-dsv41`
- 原算子测试二进制备份：
  `/data/dsv41-opt-v022-20261104/run/baseline-bin/ds4_iq_test`
- 原始 profile：
  `/data/dsv41-opt-v022-20261104/run/baseline-profile-engine.log`
- 解析后的逐 token 数据：
  [`baseline-profile-parsed.json`](baseline-profile-parsed.json)

首轮 mixed prompt profile 的关键成本如下。表中“四卡均值”是四个设备的
profiler 事件之和除以 4，再除以 prompt token 数；它用于和端到端墙钟比较。

| Prompt | 端到端 prefill | Experts | IQ1_M gate/up | IQ1_M down | Compressor/indexer |
|---|---:|---:|---:|---:|---:|
| 4K | 3,515 µs/token | 1,190 µs/token | 408 µs/token | 301 µs/token | 58 µs/token |
| 8K | 3,020 µs/token | 1,174 µs/token | 408 µs/token | 296 µs/token | 103 µs/token |
| 32K | 2,770 µs/token | 1,184 µs/token | 410 µs/token | 300 µs/token | 333 µs/token |

单卡微基准确认，IQ1_M 的 12–15 行轻量专家路径与 IQ2 接近；差距主要来自
超过 16 行后进入的旧 F32 64×64 tile。4K profile 中 IQ1_M 仍出现约
30,720 次 gate/up 和 53,760 次 down 专家处理，且其精确 2.8/3.3 TFLOP/s
与 tile 微基准一致。

首轮候选只替换这个高行数 IQ1_M tile：gfx928 BF16 16×16×16 MMAC。
候选由 `STRATA_DS4_MMAC928=1` 显式开启，旧的 F32 tile 仍是默认路径。
候选文件：

- [`../../src/dsv41/ds4_iq_mmac928.inc`](../../src/dsv41/ds4_iq_mmac928.inc)
- [`hbm_bench.hip`](hbm_bench.hip)
- [`parse_profile.py`](parse_profile.py)

公开规格给出的 K100AI 单卡上限为 896 GB/s、BF16/FP16 196 TFLOP/s、
INT8 392 TOPS。直读/写/拷贝探针仍用于测实际可持续上限，不能把公开峰值当作
可达值。下一步是完成候选编译、`ds4_iq_test --type 29` 数值检查、
32/64/128 行微基准，再做 4K/8K/32K 整模型 A/B。

## 第二轮结果

最终候选在 mixed 4K/8K/32K 上达到 2.47x/2.57x/2.37x，九提示三次矩阵、
4,995 位置 teacher-forced PPL、固定 API 任务和全部相关算子测试均通过。
完整数据、profile、质量门和已知边界见
[`round-002-final.md`](round-002-final.md)。
