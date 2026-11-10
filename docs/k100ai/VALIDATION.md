# K100AI 验证记录

[返回首页](../../README.md) · [文档索引](README.md) · [来源与校验](PROVENANCE.md)

本页保留 Qwen 在 v0.2.3 的完整验收数据，并汇总 v0.3.0 的验证边界。v0.3.0 的 DeepSeek 三套生产配置、PPL、32K 输入和 DSpark 结果见 [DeepSeek-V4.1 指南](DEEPSEEK41.md)。Qwen 的两个二进制未改变，因此继续使用相同验收记录。

所有 GPU 测试只使用 HCU0–3。每个 case 前执行 preflight，结束后确认没有测试容器或设备 lease。

## 构建和算子测试

v0.3.0 发布包包含三个生产二进制，运行环境均为 DTK 2604：

| 目标 | 大小 | SHA-256 |
| --- | ---: | --- |
| Qwen 默认快速档 `strata-0.1.47` | 45,674,152 B | `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9` |
| Qwen 当前对照档 `strata-0.1.55` | 49,496,856 B | `25d4ea83b883443d0a8bc68686c72f613cd0923a2897c2f7e9063939025e904e` |
| DeepSeek `strata-dsv41` | 23,941,864 B | `4fd967c1fa6769e60298357d55683c18e5522f6689441b12fe5a97489b3a35d6` |

v0.2.3 的算子测试还包括 `ds4_gguf_test`，覆盖 BF16、MXFP4、Q8_0、Q2_K、Q3_K、Q4_K、IQ4_XS、IQ3_S、IQ2_XXS 和 IQ2_S 的 expert、CPU lane、SwiGLU、combine 与 KV packed rows。该项属于历史构建记录。

## Qwen 精度和快速档验收

0.1.55 对照档保留 `8544d2fa` 的 gfx928 int8 expert MMAC 精确集合：

- gate/up 安全格式：`IQ3_XXS,IQ3_S,IQ4_XS`
- down 安全格式：`Q2_0`
- 未设置、空值和 `all` 都只选择上述安全集合

这些格式逐项验证保持输出和 first logits。`IQ2_XXS`、`IQ2_XS` gate
精确路径会出现漂移，不能用于生产；`IQ4_NL` down 与 `Q2_0` 同时启用也
会出现漂移，因此该对照档只启用 `Q2_0` down。

0.1.55 对照档的 canonical 结果：

| 检查 | SHA-256 |
| --- | --- |
| Qwen 64-token 输出 | `853f3ceb7f86361934939409a5e0ff5753c34cb2c5990365b5d21b7ca898142b` |
| Qwen 256-token 输出 | `30a8df09679b1960f58a94a430136030654db248308952662784e8b79f578d93` |
| Qwen 64-token first logits | `432cb823f8e7e0fc1d774fc3bd541dbded93e7617627999406410fd05e7a5125` |
| Qwen 256-token first logits | `432cb823f8e7e0fc1d774fc3bd541dbded93e7617627999406410fd05e7a5125` |
| Qwen 4,095 prompt + 64 output | `36001b836010a5e19d9b22003c09c4c7d844bd290029decbb8d0d1438aa9c5ed` |
| Qwen long first logits | `fb4d985a7046a08778e43e517108a6e9ba2283a5104112b1c9022d9726eed76d` |

输出哈希是去掉日志行末尾空白后，对空格分隔 token ID 再补一个 newline
计算；first logits 对引擎 dump 文件本身计算 SHA-256。

0.1.55 对照档配合外置 MTP runtime 的实测：

| 输入 | MTP loaded | Draft 接受 | Decode | 输出 / first logits |
| --- | --- | --- | ---: | --- |
| 18-token，64 输出 | 798 MiB | 42 / 66 | 61.70 tok/s | 均保持 canonical |
| 4,095-token，64 输出 | 808 MiB | 48 / 48 | 75.83 tok/s | 均保持 canonical |

长输入这次 GPU timeline 为 7,524 ms，引擎 prefill 7,564.9 ms，CLI
prefill 8,034.2 ms，TTFT 9,932.8 ms。draft layer 明确打印
`strata mtp: draft layer loaded`，接受数在引擎日志和 API metrics 中均为
非零，因此可以声称本版本实际启用了 MTP；该结论不覆盖未加载 draft layer
的旧 run。

0.1.55 对照档的 64-token、256-token、4,095-token、MTP 64-token 和 MTP
4,095-token 运行日志保留在构建主机，未打进发布归档；归档检查要求模型
权重、外置 MTP runtime 和研究构建路径均为零。

默认 `bin/strata` 快速档采用 K100AI 原生 gfx928 prompt MMAC、fused
gate/up、Q2_0 down、HC upmix 和 FP16 prompt GEMM。验收门槛是：first-logit
top-1 保持、余弦相似度高于 0.98、短输出语义及 greedy 前缀保持。实测：

| 输入 | Top-1 | Max abs | Mean abs | 相对 L2 | 余弦 |
| --- | --- | ---: | ---: | ---: | ---: |
| 8,191 token | 112853 = 112853 | 2.00396 | 0.27595 | 13.04% | 0.991468 |
| 40,058 token，对照 0.1.55 F16 | 12654 = 12654 | 4.22781 | 0.42068 | 16.71% | 0.987025 |

8,191-token 快速档运行保持 top-1，MTP draft layer 加载，draft 接受
56/57，64-token decode 为 79.37 tok/s。40,058-token 快速档的 8-token
输出与 0.1.55 对照完全相同：`12654 220 136723 104302 97318 3709 96317
98153`。该漂移高于纯 FP16 档，所以快速档不冒充 bitwise 或 canonical
profile。

## Qwen 性能

相同 pack、profile、expert cache、设备、KV int8、prefill chunk 和
`--no-prefill-borrow` 的配对测试：

| 输入 | 引擎 | GPU timeline | 引擎 prefill | 吞吐 |
| --- | --- | ---: | ---: | ---: |
| 8,190 | 0.1.47 快速档 | 4,375 ms | 4,436.2 ms | 1,846.2 tok/s |
| 8,190 | 0.1.55 F16 | 7,860 ms | 7,923.4 ms | 1,033.6 tok/s |
| 40,057 | 0.1.47 快速档 | 19,066 ms | 19,202.2 ms | 2,086.1 tok/s |
| 40,057 | 0.1.55 F16，`--no-prefill-borrow` | 34,361 ms | 34,492.4 ms | 1,161.3 tok/s |

0.1.55 在 0.1.47 原环境重跑时为 1,218.5 tok/s，说明旧环境开关只贡献约
5%；关闭 exact MMAC 反而降到 1,042.0 tok/s，启用 HC upmix 也只到
1,266.2 tok/s。阶段日志确认 0.1.55 主要慢在 prompt attention、gate/up
和 down；不是 KV、专家借用或测量口径。当前实现没有达到旧版
1.8x–2.1x 的路径，因此默认包使用经验证的 0.1.47 快速档，而不是声称
0.1.55 已修复。

## 历史记录：DeepSeek 四卡、DSpark 和长输入（v0.2.3）

最终包的 `strata-dsv41` 在 HCU0-3 上完整放置四张卡，CPU lane 为 0。

### 单并发十档长输出

服务使用四张 K100AI、IQ2_XXS、FP4 KV、1,048,576-token 配置上下文、
DSpark Q8_0、6 experts 和 `--dspark-conf 0.65`。请求使用 OpenAI 兼容
`/v1/chat/completions`，单并发、`temperature=0`、`top_p=1`、目标输出
256 tokens，逐档冷启动，所有档 `cached_tokens=0`。输出速度使用服务端
timings；TTFT 与总请求时间使用客户端墙钟。

| 目标输入 | 实际 prompt | TTFT | 预填充 | 输出 | 输出 tokens | 说明 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 556 | 2.079 s | 121.6 tok/s | 26.8 tok/s | 256 | 正常 |
| 2,048 | 2,092 | 3.685 s | 274.9 tok/s | 14.5 tok/s | 256 | 正常 |
| 4,096 | 4,141 | 5.768 s | 349.7 tok/s | 16.9 tok/s | 256 | 正常 |
| 8,192 | 8,237 | 9.948 s | 415.0 tok/s | 22.0 tok/s | 160 | 自然 EOS |
| 12,288 | 12,333 | 13.632 s | 452.4 tok/s | 25.8 tok/s | 256 | 正常 |
| 16,384 | 16,429 | 37.289 s | 201.8 tok/s | **3.0 tok/s** | 256 | **异常，排除出正常区间统计** |
| 32,768 | 32,813 | 38.409 s | 426.9 tok/s | 27.7 tok/s | 256 | 正常 |
| 65,536 | 65,581 | 81.431 s | 399.8 tok/s | 26.0 tok/s | 256 | 正常 |
| 131,072 | 131,117 | 191.209 s | 333.8 tok/s | 24.2 tok/s | 256 | 正常 |
| 257,901 | 257,945 | 355.155 s | 363.5 tok/s | 26.3 tok/s | 256 | 正常 |

正常档的输出范围是 14.5–27.7 tok/s，中位数约 25.8 tok/s。32K 及以上没有
出现随上下文单调下降，说明这一组 mixed-content 请求的主要限制不是 KV
长度。DSpark 每轮能接受多少 draft、输出本身是否可预测，仍是主要变量。

16K 档是异常点：引擎日志前半段约 128 tokens / 75,664 ms，即 1.69 tok/s；
后半段约 111 tokens / 4,037 ms，即 27.50 tok/s。两段合并得到报告中的
3.0 tok/s。相邻的 12K 与 32K 档正常，因此该点不作为长上下文退化证据。

### 69.19 tok/s 的准确口径

发布 smoke 使用固定 25-token 输入和 64-token 输出。输出是以下 8-token
序列连续重复 8 次：

```text
455,14083,2645,344,260,1688,2645,16
```

这种高度可预测的输出让 DSpark 在 11 轮内从 54 个 offered drafts 中接受
52 个，命中率
96.3%，最终 decode 为 64 tokens / 911 ms，即 69.19 tok/s。该次 smoke 中：

- CUDA0 layers 0..9：4,224 / 4,224 experts，37.30 GB
- CUDA1 layers 10..19：3,840 / 3,840 experts，33.12 GB
- CUDA2 layers 20..29：3,840 / 3,840 experts，34.01 GB
- CUDA3 layers 30..39 + head：3,840 / 3,840 experts，34.86 GB
- DSpark：11 rounds，54 drafts offered，52 accepted
- decode：64 tokens / 911 ms，69.19 tok/s

固定 25-token 输入按 DSpark off/on/off 运行：

| 运行 | Decode | DSpark |
| --- | ---: | --- |
| off A | 24.32 tok/s | off |
| on B | 68.53 tok/s | 54 offered / 52 accepted |
| off C | 24.25 tok/s | off |
| on，最终包短输出峰值 | 69.19 tok/s | 54 offered / 52 accepted |

on/off 平均加速为 `68.53 / ((24.32 + 24.25) / 2) = 2.82x`。off/on/off
和最终包 smoke 的四组 64-token token ID 完全相同。profile 测试前后均为：

```text
8db93e483b34b15dbf63a0826601f1db2107a8cfea73774e7d21d8559e0a9568
```

服务使用冻结的 `data/dsv41/profile-2.00.bin` 和 `--profile-readonly`。

因此 69.19 tok/s 和 2.82× 只描述上述重复短序列。它们不能写成任意内容、
任意输出长度或长上下文的持续吞吐。较早日志中的 52 offered / 52 accepted
属于不同的 21-token 调参运行，decode 为 62.75–65.00 tok/s；发布 smoke
的最终证据是 25-token 输入与 54 offered / 52 accepted。

有效 4,095-token DSV 长输入使用重复且已验证的 25-token DSV 序列：

- prompt：4,095 tokens / 13,629 ms，300.45 tok/s
- DSpark：2 rounds，5 offered / 5 accepted
- decode：8 tokens / 145 ms，48.32 tok/s
- 四卡加载正常，CPU lane 为 0 / 1,725 experts
- profile read-only，运行后不变

对应的 off A、on B、off C、最终包 smoke 和有效 4,095-token 日志保留在
构建主机，不进入归档。

## Python、静态和发布包测试

最终改动后的本地结果：

- `python3 -m unittest tests.terminal.test_drafting serve.test_dsv41 serve.test_server`：
  464 tests，OK；其中 `serve.test_dsv41` 的 1 项因 `STRATA_DSV41_PACK`
  未设置而跳过
- `bash -n scripts/k100ai/*.sh`
- 所有 `scripts/k100ai/*.py` 通过 `py_compile`
- 两个 server template 通过 `json.tool`
- `git diff --check`
- launcher、preflight、setup 脚本保持 executable

本版发布归档：

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `strata-v0.2.9-k100ai-dual.tar.gz` | 70,790,149 B | `e318437e0bd5aff676dd022a737267c933a5f0df9c393fb83f1b8603073e2871` |
| `strata-v0.2.9-k100ai-dual-source.tar.gz` | 76,881,571 B | `deeffa4bc7086656b84ab8ec0507623a9fe78c44a46fdfff5c1a12038968e5b3` |
| `validation-evidence.tar.gz` | 59,981 B | `702c54ebd44d40de2dc40b0dda6ab2ab7d02712761511f28a829a5c585c9c764` |

运行包在全新目录验证：

- 解压后 `sha256sum -c SHA256SUMS`：5,812 个文件条目全部 OK
- 固定 DTK 2604 容器内 `--version`：`strata 0.1.47`、`strata 0.1.55`、
  `strata-dsv41 0.1.55 build unknown`

以下 API 和启动器 smoke 是 v0.2.2 发布包的历史验证记录，用于说明同源
runtime 路径；本次 v0.2.3 没有连接 HCU，未重跑 GPU/API 验证，因此不把
这些历史结果表述为本版最终包的实测：

- DTK Python 3.10 中首次 runtime setup 和重复 setup/import marker
- preflight 的 HCU0-3 占用检查
- launcher 同名容器拒绝、SIGTERM 清理和研究容器保持
- Qwen 与 DeepSeek OpenAI-compatible API smoke
- 模型权重没有进入包；嵌套源码树中的 `.gguf`、`.safetensors`、`.onnx`、
  `.pt` 和 `.pth` 会被打包器排除
- 嵌套源码树中的 `experts.bin`、`dense.bin`、`dense.txt` 和
  `draft_vocab*.bin` 会被打包器排除；外置 MTP runtime 不进入归档
- 嵌套源码树中的研究构建脚本和旧 qwen-server 配置会被排除，避免把主机
  特定 build 路径带进发布归档
- 部署脚本、配置和 `BUILD.json` 不引用研究/构建绝对路径

## OpenAI 兼容 API

双模型 API smoke 都使用 Greedy：

- Qwen `/health` 和 `/v1/models` 报告模型已加载、`max_context=262144`
- Qwen 非流式：62 prompt / 16 completion tokens，`finish_reason=length`
- Qwen 流式：cache hit 正常，16 completion tokens，`finish_reason=length`
- Qwen `/metrics` 报告 `drafting=mtp`；两次请求合计 drafts
  `46 offered / 28 accepted`，Prometheus 对应计数均非零
- DeepSeek `/health` 和 `/v1/models` 报告模型已加载、
  `max_context=1048576`
- DeepSeek 非流式：39 prompt / 8 completion tokens，
  `finish_reason=length`
- DeepSeek 流式：8 completion tokens；两次请求合计
  `14 offered / 14 accepted` drafts
- DeepSeek `/metrics` 报告 KV FP4、`kv_resident=1048576`、
  `expert_slots=15360`、`engram_layers=2`、`drafting=dspark`

先前最终二进制长请求已用 120,029 个独特 prompt tokens 完成，服务保持
稳定。该结果只证明该深度可用，不证明 1,048,576-token 窗口已全部灌满。

## 明确限制

- 原生 NVIDIA/CUDA 编译未完成。当前 DTK/HIP 构建不能作为 nvcc 编译
  证据；发布包只声明并验证 gfx928 + DTK 2604。
- Qwen 配置上限为 262,144 tokens。本轮快速档在 8,191 和 40,058 tokens
  重测，没有单独重跑 20K；更早的 0.1.47 服务记录到 100,058 tokens。没有
  重新灌满完整 262,144-token 窗口，也不声称整个窗口的语义质量。
- 40K 数值对照使用快速 0.1.47 与本轮 0.1.55 F16，并非 canonical 0.1.55
  first-logits oracle；证据是同 top-1、各完整性统计和相同的 8-token 输出
  前缀。
- DeepSeek 分配了 1,048,576-token KV，但独特内容验证到 120,029 tokens，
  有效长 prompt 实测到 4,095 tokens；未做完整 1M 灌满、长时间流式或
  长 prompt 答案质量评测。
- `--vision off`，本版本不宣称图片输入。
- `IQ2_XXS`、`IQ2_XS` gate exact MMAC 和双 down-format 组合仍是诊断
  路径，不属于生产支持。
- Qwen 默认快速档的 HC upmix、prompt MMAC 和 fused expert 输出不
  bitwise；它按本文的 top-1、余弦和输出前缀门槛验收。0.1.55 对照档保留
  在 `bin/strata-0.1.55`。
