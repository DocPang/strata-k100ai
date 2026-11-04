# 更新记录

## v0.2.2 — Qwen + DeepSeek 双模型与可复现发布

本版把 `strata-k100ai` 从单一 Qwen Docker 部署整理包，升级为面向 Hygon K100AI / gfx928 的双模型运行包：

- Qwen3.8-Flash-Next：默认 0.1.47 快速引擎，保留 0.1.55 对照引擎。
- DeepSeek-V4.1-Flash：独立 0.1.55 引擎，四卡、1M context、FP4 KV 与 DSpark。
- 两个模型都通过同一套 OpenAI / Anthropic 兼容 API 提供服务。
- 运行包与源码包分别发布，并加入确定性归档、固定镜像、固定模型 revision、二进制哈希断言和隔离安装验证。

上游基线为 **Strata v0.2.9**。本版不是 v0.2.1 的增量补丁；目录结构、启动入口和兼容范围都有变化。

## 成绩与速度

Qwen 在单卡 K100AI 64 GiB、gfx928、IQ3_XXS、INT8 KV、24,576 expert slots、prefill cap 20,480、MTP `spec=6` 下完成 matched A/B：

| 输入 | 0.1.47 快速档 | 快速档吞吐 | 0.1.55 | 0.1.55 吞吐 | 快速档倍数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,190 tokens | **4,375 ms** | **1,846.2 tok/s** | 7,860 ms | 1,033.6 tok/s | **1.80×** |
| 40,057 tokens | **19,066 ms** | **2,086.1 tok/s** | 34,361 ms | 1,161.3 tok/s | **1.80×** |

- Qwen 0.1.47 decode：**79.37 tok/s**，MTP accepted **56** / offered **57**。
- Qwen 0.1.55 在 4,095-token prompt + 64-token decode：**75.83 tok/s**，MTP accepted **48** / offered **48**。
- DeepSeek 条件是 4× K100AI、IQ2_XXS、FP4 KV、DSpark：64-token decode **69.19 tok/s**，accepted **52** / offered **54**。
- DeepSeek 4,095-token prompt：**300.45 tok/s** prefill；后续 8-token decode 为 **48.32 tok/s**。

Qwen 快速档不是 bitwise：

| 输入 | Top-1 | Max abs | Mean abs | Relative L2 | Cosine |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8,191 | 一致 | 2.00396 | 0.27595 | 13.04% | 0.991468 |
| 40,058 | 一致 | 4.22781 | 0.42068 | 16.71% | 0.987025 |

40,058-token 快速档和 0.1.55 的前 8 个 greedy token 完全相同。本版接受数值漂移以换取约 1.8× 的 prompt 吞吐，同时保留 0.1.55 作为 canonical 对照。

## 主要变化

### 双模型运行包

`strata-v0.2.9-k100ai-dual.tar.gz` 包含四个 gfx928 二进制：

| 文件 | 角色 | 大小 | SHA-256 |
| --- | --- | ---: | --- |
| `bin/strata` | Qwen 0.1.47 默认快速档 | 45,674,152 B | `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9` |
| `bin/strata-0.1.55` | Qwen 0.1.55 对照档 | 49,484,096 B | `437308792b7ec08724aa90e4b11736b4f40f41c10613faa9688fbdc1723ba7ab` |
| `bin/strata-dsv41` | DeepSeek-V4.1-Flash | 22,576,544 B | `6543311f453bee2c6d61653a24f0b6282f9cb3568b17a8b889c2c5e073b12286` |
| `bin/ds4_gguf_test` | DeepSeek GGUF 算子自检 | 18,816,624 B | `0d6c8455d76815b810a38c5529bc868087e71641008e0c86843ecd8b34dd815c` |

发布页提供运行包的 `.sha256` sidecar，包内另有 `SHA256SUMS`。模型权重、MTP runtime、DeepSeek pack 和 `draft_vocab.bin` 均未进入归档。

### Qwen 使用双引擎策略

默认 `bin/strata` 继续使用已经验证的 0.1.47 K100AI 快速路径，并启用 MTP：

- gfx928 native prompt MMAC
- F16 prompt GEMM
- fused gate/up
- Q2_0 down
- HC upmix

包内同时提供 `bin/strata-0.1.55`。该引擎包含 `8544d2fa` 的 gfx928 int8 expert MMAC，可用于 canonical 对照：

- gate/up：`IQ3_XXS,IQ3_S,IQ4_XS`
- down：`Q2_0`

这不是简单“保留两个文件”：两条引擎路径分别从固定源码构建，并各自带哈希断言。

### DeepSeek-V4.1-Flash

新增 `bin/strata-dsv41` 和 `bin/ds4_gguf_test`：

- 4× K100AI
- 1,048,576-token context
- FP4 KV
- 15,360 expert slots
- Engram 2 layers / 48 lookups
- DSpark drafting
- `cpu_experts=0`、`ram_experts=0`
- 冻结 placement profile，服务使用 `--profile-readonly`

DeepSeek pack 由固定 backbone shard、Engram 和 DSpark 生成。所有下载使用逐文件 ModelScope revision，并在生成后逐文件比对 SHA-256。

### 优化内容

| 范围 | 本版采用的优化 |
| --- | --- |
| K100AI 平台 | gfx928 HIP 兼容层、DTK 2604 编译链接、rocBLAS/hipBLASLt 接口和 ggml/llama.cpp MMQ 衔接 |
| Qwen prompt attention | 默认 0.1.47 使用 native gfx928 prompt MMAC；0.1.55 暂未恢复这条路径 |
| gate/up | fused gate/up、exact IQ2 展开、wsg16 tile；0.1.55 exact 集合覆盖 IQ3_XXS、IQ3_S、IQ4_XS |
| down | Q2_0 展开与 int8 MMAC、IQ4_NL exact pre-expansion、staging 和行 tile 调整 |
| Prompt GEMM | F16 prompt GEMM、形状相关 GEMM tile、stride padding 和验证过的 Tensile solution |
| Hyper-connection | HC upmix、split/staged read、down prefetch 和单 LDS plane 分块 |
| GDN 与路由 | 八 lane GDN 递归、去掉无用 FP32 写回、常驻 expert 排序和 native/MMQ/fused 分派 |
| MTP | 外置 Q2_0 MTP runtime、`spec=6`、接受阈值 0.7、draft graph 预热和会话缓存 |
| DeepSeek 四卡 | 每卡 10 层的 `10,10,10,10` 切分、全部 routed experts 驻留 HCU、1024-token prefill chunk |
| DeepSeek KV 与 Engram | 1M FP4 resident KV、2 层 Engram、48 lookups，CPU/RAM expert lane 为 0 |
| DeepSeek drafting | DSpark Q8_0、置信度 0.65，并在 API metrics 中记录 offered/accepted drafts |
| 服务与安全 | OpenAI/Anthropic API、metrics、127.0.0.1 绑定、单次容器清理和 HCU 预检查 |
| 可复现性 | 固定镜像 digest、固定模型 revision、逐文件 SHA-256、canonical 构建路径和确定性归档 |

Qwen 0.1.55 的 prompt 回归没有被写成“已优化”：最终单卡 prompt 仍比 0.1.47 慢约 1.8 倍，本版选择把 0.1.47 作为默认快速档。

### 可复现发布

新增 `strata-v0.2.9-k100ai-dual-source.tar.gz`，发布页提供对应的 `.sha256` sidecar。

源码包包含：

```text
main/
qwen-fast-0.1.47/
llama.cpp/
BUILD_INPUTS.json
```

固定输入：

- 打包源码：v0.2.2 tag commit；准确值同时记录在 `BUILD.json` 和 `BUILD_INPUTS.json`
- 当前引擎源码：`8544d2fa6cddf61fb47edc0c3bd8a5d7d638140e`
- Qwen 快速引擎源码：`2150b1acdb48c386a9afcd64e63e2369b5a5e108`
- 上游基线：`c62d5e3a15b40ea51c96aae9e8d36928965f53ab`

两个归档均为确定性 tar/gzip：排序路径、零时间戳、空 owner/group。连续生成两次，运行包与源码包均得到相同哈希。

### 安装与启动

本版 README 以**人工步骤**为默认安装路径，不要求调用全自动下载或启动脚本。文档逐项列出：

- Python 依赖的完整 `pip --target` 安装命令
- Qwen 两个 GGUF 的精确文件名、ModelScope snapshot、URL、大小和 SHA-256
- Qwen pack 的独立转换命令与六个输出文件摘要
- MTP 所需的 31 个 tensor 名称、来源 shard 和 SHA-256
- DeepSeek 五个 backbone shard、Engram 与 DSpark 的精确下载 URL
- DeepSeek pack 的独立生成命令和八个输出摘要
- HCU 设备、容器占用、Docker 挂载、端口、启动、检查和停止命令

上述步骤见新版 README 的“手动安装与运行”。只有在用户已经理解并核对人工步骤后，才可选择使用发布包内脚本减少重复输入：

```bash
bash scripts/k100ai/setup-runtime.sh
bash scripts/k100ai/preflight.sh
```

脚本不是唯一入口，也不是复现证据。launcher 仍会检查外部资产和 HCU0-3 占用；它不会自动停止其他研究容器，只清理本次创建的容器。

## 验证结果

最终候选包在独立空目录完成解包和 runtime 安装：

- 运行包内部 `SHA256SUMS`：全部通过
- runtime 依赖安装：首次安装通过
- runtime 依赖安装：重复执行通过
- preflight：每轮 GPU 测试前后均报告 `HCU0-3 are free`

真实 GPU API：

| 测试 | 结果 |
| --- | --- |
| Qwen 默认 0.1.47 | 262,144 context，MTP，46 offered / 28 accepted |
| Qwen 0.1.55 | 262,144 context，MTP，46 offered / 28 accepted |
| DeepSeek | 1,048,576 context，FP4 KV，DSpark，14 offered / 14 accepted |
| 人工直接 Docker 启动 | Qwen 与 DeepSeek 均加载成功；DeepSeek 请求返回 2/2 接受的 drafts |

DeepSeek `ds4_gguf_test` 完成：

- KV packing
- sparse attention
- BF16 / MXFP4 / Q8_0 MMV
- token/window experts
- CPU lanes
- SwiGLU
- combine
- expertise sets
- 量化格式

最终结果：

```text
ALL OK (0 failed)
```

DeepSeek profile 测试前后均为：

```text
8db93e483b34b15dbf63a0826601f1db2107a8cfea73774e7d21d8559e0a9568
```

源码测试：

- `tests.terminal.test_drafting`：40/40
- `serve.test_dsv41`：64 通过，1 个依赖模型 pack 的测试跳过
- `serve.test_server`：360/360

## 下载

发布页应上传：

- `strata-v0.2.9-k100ai-dual.tar.gz`
- `strata-v0.2.9-k100ai-dual.tar.gz.sha256`
- `strata-v0.2.9-k100ai-dual-source.tar.gz`
- `strata-v0.2.9-k100ai-dual-source.tar.gz.sha256`
- `RELEASE.json`
- `RELEASE.json.sha256`
- 验证日志归档

模型权重和模型派生运行资产不在发布附件中。下载与生成命令见新版 README。

## 升级与兼容性

### 从 v0.2.1 升级

请安装到新的空目录，不要覆盖旧部署：

1. 下载本版运行包和源码包。
2. 校验归档与内部 `SHA256SUMS`。
3. 运行 `setup-runtime.sh`。
4. 准备新的 Qwen pack/MTP 和 DeepSeek pack。
5. 使用 `serve-k100ai.sh` 启动。

旧版目录中的 `install.sh`、`start.sh`、`compose.yaml` 和 `deployment.env` 不属于本版运行契约。

### 视觉功能

v0.2.1 的 GPU 图片编码器不包含在本版本中。Qwen 和 DeepSeek 都以 `--vision off` 运行，`/health` 会报告 `images=false`。需要视觉能力的部署应继续使用 v0.2.1，或等待单独的视觉发布。

### API 模型名

- Qwen：`qwen3.8-flash-next`
- DeepSeek：`deepseek-v4.1-flash`

旧的 `qwen3.8-flash-next-strata` 名称不再由本版配置使用。

## 已知限制

- 仅支持 Hygon K100AI `gfx928` + DTK 2604；原生 NVIDIA/CUDA 未构建。
- Qwen 0.1.55 的 prompt attention 性能回归尚未修复。
- Qwen 默认 0.1.47 是有实测边界的 drift-accepted 快速档。
- 未灌满 262,144-token Qwen 或 1,048,576-token DeepSeek 窗口做语义质量评测。
- 本版视觉关闭。
- 模型资产许可证需要使用者自行确认；运行包不重新分发模型权重。

## 文档

- [README](https://github.com/DocPang/strata-k100ai/blob/v0.2.2/README.md)
- [运行与模型资产](https://github.com/DocPang/strata-k100ai/blob/v0.2.2/docs/k100ai/README.md)
- [独立复现](https://github.com/DocPang/strata-k100ai/blob/v0.2.2/docs/k100ai/REPRODUCIBILITY.md)
- [验证结果](https://github.com/DocPang/strata-k100ai/blob/v0.2.2/docs/k100ai/VALIDATION.md)
- [来源与提交](https://github.com/DocPang/strata-k100ai/blob/v0.2.2/docs/k100ai/PROVENANCE.md)

## 致谢与许可

本项目基于 [Strata](https://github.com/Niko1221/Strata)、[llama.cpp](https://github.com/ggml-org/llama.cpp)、ISTA-DASLab 的 Qwen/DeepSeek GGUF、Unsloth 的 DSpark，以及 Hygon DTK/HyHAL 构建。

项目代码遵循 MIT License。模型、量化文件、DTK/HyHAL 和上游依赖分别遵循各自许可证。

## 历史版本

- [v0.2.1](https://github.com/DocPang/strata-k100ai/releases/tag/v0.2.1)：Qwen3.8-Flash-Next GPU 多模态视觉。
- [v0.2.0](https://github.com/DocPang/strata-k100ai/releases/tag/v0.2.0)：K100AI 优化集重基到 Strata 0.2.2。
- [v0.1.0-rc.7](https://github.com/DocPang/strata-k100ai/releases/tag/v0.1.0-rc.7)：rocBLAS Tensile solution 按形状锁定。
- [更早版本](https://github.com/DocPang/strata-k100ai/releases)：rc.1–rc.6 和首个公开部署包。
