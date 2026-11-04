# K100AI 双模型验证

发布包采用逐模型二进制分层。Qwen 默认快速档是本机已验证的 K100AI
0.1.47；DeepSeek 和 Qwen 对照档来自
`8544d2fa6cddf61fb47edc0c3bd8a5d7d638140e` 的 0.1.55。所有 GPU 测试只用
HCU0-3；每个 case 前执行 preflight，结束后确认没有测试容器和 lease，三个
研究容器保持 `exited`。

## 构建和算子测试

包内有四个 gfx928 二进制，运行环境均为同一 DTK 2604 镜像：

| 目标 | 大小 | SHA-256 |
| --- | ---: | --- |
| Qwen 默认快速档 `strata`（0.1.47） | 45,674,152 B | `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9` |
| Qwen 当前对照档 `strata-0.1.55` | 49,484,096 B | `437308792b7ec08724aa90e4b11736b4f40f41c10613faa9688fbdc1723ba7ab` |
| DeepSeek `strata-dsv41` | 22,576,544 B | `6543311f453bee2c6d61653a24f0b6282f9cb3568b17a8b889c2c5e073b12286` |
| DeepSeek `ds4_gguf_test` | 18,816,624 B | `0d6c8455d76815b810a38c5529bc868087e71641008e0c86843ecd8b34dd815c` |

最终包内 `ds4_gguf_test` 输出 `ALL OK (0 failed)`。覆盖 BF16、MXFP4、
Q8_0、Q2_K、Q3_K、Q4_K、IQ4_XS、IQ3_S、IQ2_XXS、IQ2_S 的
token/window expert、CPU lane、SwiGLU、combine 和 KV packed rows。

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

用户质疑此前 4K “无回归”结论是正确的。相同 pack、profile、expert cache、
设备、KV int8、prefill chunk 和 `--no-prefill-borrow` 的 matched A/B：

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

## DeepSeek 四卡、DSpark 和长输入

最终包的 `strata-dsv41` 在 HCU0-3 上完整放置四张卡，CPU lane 为 0。
64-token smoke 中：

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
| on，最终包 smoke | 69.19 tok/s | 54 offered / 52 accepted |

on/off 平均加速为 `68.53 / ((24.32 + 24.25) / 2) = 2.82x`。off/on/off
和最终包 smoke 的四组 64-token token ID 完全相同。profile 测试前后均为：

```text
8db93e483b34b15dbf63a0826601f1db2107a8cfea73774e7d21d8559e0a9568
```

服务使用冻结的 `data/dsv41/profile-2.00.bin` 和 `--profile-readonly`。

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

- `python3 -m unittest tests.terminal.test_drafting -v`：40 tests，OK
- `python3 -m unittest serve.test_dsv41`：64 tests，OK，1 skipped
- `python3 -m unittest serve.test_server`：360 tests，OK
- `bash -n scripts/k100ai/*.sh`
- 所有 `scripts/k100ai/*.py` 通过 `py_compile`
- 两个 server template 通过 `json.tool`
- `git diff --check`
- launcher、preflight、setup 脚本保持 executable

发布包还在全新目录验证：

- `sha256sum -c SHA256SUMS`
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
- 公开 push、公开分支和 release 仍处于冻结状态。
