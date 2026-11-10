# K100AI 运行参数

K100AI 发布版本 **v0.3.0**。DeepSeek-V4.1-Flash 的 R001-R179
最终优化结果、三套生产配置、PPL/API 和启动说明见
[DeepSeek-V4.1 指南](../../DEEPSEEK41.md)。完整下载和资产准备步骤见[历史安装说明](INSTALL.md)。支持：

- DeepSeek-V4.1-Flash，服务模型名 `deepseek-v4.1-flash`
- Qwen3.8-Flash-Next，服务模型名 `qwen3.8-flash-next`

发布目录可从任意空目录解包运行，不依赖构建目录或研究目录中的绝对路径。
模型权重不进入发布包；嵌套 `source/` 树也排除 `.gguf`、`.safetensors`、
`.onnx`、`.pt` 和 `.pth`，包括可选 Qwen 实验控制向量。

## 版本

- 上游基础：Strata v0.2.9，`c62d5e3a15b40ea51c96aae9e8d36928965f53ab`
- 当前引擎源码：`8c413cae0f5e9f88df784642772d78dff7bef428`
- 打包源码：`ae0abbd3a8dc198d0983ed5d947ad554a52330a3`
- 当前引擎版本：`0.1.55`
- Qwen 默认快速引擎版本：`0.1.47`
- 发布标签：`v0.2.3-k100ai`
- 镜像：
  `harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68`
- 允许设备：HCU0-3，对应
  `/dev/dri/renderD128` 至 `/dev/dri/renderD131`

发布包按模型选择引擎。Qwen 默认使用经过 8K/40K
验证的 0.1.47 快速档，因为当前 0.1.55 的 prompt 实现仍有可复现的长上下文
回归；DeepSeek 使用 0.1.55。当前 Qwen 0.1.55 也保留在包内，便于切换和
对照。`BUILD.json` 分开记录各引擎与打包源码：

```json
{
  "engine_version": "0.1.55",
  "engine_source_commit": "8c413cae0f5e9f88df784642772d78dff7bef428",
  "package_source_commit": "ae0abbd3a8dc198d0983ed5d947ad554a52330a3",
  "container_image": "harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68",
  "platform": "linux-x86_64-gfx928-dtk2604",
  "models_included": false,
  "qwen_engines": {
    "default_fast": {
      "binary": "bin/strata",
      "engine_version": "0.1.47"
    },
    "current": {
      "binary": "bin/strata-0.1.55",
      "engine_version": "0.1.55"
    }
  }
}
```

`package_source_commit` 以包内 `BUILD.json` 为准。它包含运行时、launcher、
文档和打包修复；不把这些提交误报为新的引擎源码提交。

最终二进制：

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `bin/strata`（Qwen 默认快速档，0.1.47） | 45,674,152 B | `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9` |
| `bin/strata-0.1.55`（Qwen 当前对照档） | 49,496,856 B | `25d4ea83b883443d0a8bc68686c72f613cd0923a2897c2f7e9063939025e904e` |
| `bin/strata-dsv41` | 22,812,816 B | `1b0610416b3cce179983f3e439c2d4dc173122c9dc432b485450107b9116d811` |
| `bin/ds4_gguf_test` | 19,044,176 B | `6159e5e04a602f0362f23501194d63cce0bc47c2089f0b3be4dd088d486f9ee6` |

最终归档：

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `strata-v0.2.9-k100ai-dual.tar.gz` | 70,790,149 B | `e318437e0bd5aff676dd022a737267c933a5f0df9c393fb83f1b8603073e2871` |
| `strata-v0.2.9-k100ai-dual-source.tar.gz` | 76,881,571 B | `deeffa4bc7086656b84ab8ec0507623a9fe78c44a46fdfff5c1a12038968e5b3` |
| `validation-evidence.tar.gz` | 59,981 B | `702c54ebd44d40de2dc40b0dda6ab2ab7d02712761511f28a829a5c585c9c764` |

配套源码归档、资产下载和从零构建步骤见
[REPRODUCIBILITY.md](../../REPRODUCIBILITY.md)。源码归档包含主源码树、
Qwen 0.1.47 快速源码树和固定 llama.cpp 快照；运行包和源码归档都采用
固定顺序、零时间戳的 tar/gzip 元数据生成。

## 解包和安装

先把归档解到空目录，再从该目录安装 Python 运行时依赖：

```bash
mkdir -p /data/strata-v029-k100ai
tar -xzf strata-v0.2.9-k100ai-dual.tar.gz \
  -C /data/strata-v029-k100ai --strip-components=1
cd /data/strata-v029-k100ai
sha256sum -c SHA256SUMS
bash scripts/k100ai/setup-runtime.sh
```

`setup-runtime.sh` 使用 DTK 镜像内的 Python 3.10 和 pip `--target` 安装到
包内 `.python-packages`。重复执行时会再次在容器 Python 3.10 中验证导入，
成功后写入 `.python-packages/.k100ai-ready`。

## 启动

每次 GPU 工作前执行：

```bash
cd /data/strata-v029-k100ai
bash scripts/k100ai/preflight.sh
```

preflight 会拒绝：

- 缺失的 HCU0-3 render 设备
- `fuser` 可见的设备占用
- 任意运行中容器映射 `/dev/dri/renderD128` 至 `renderD131`

第三项修复了此前“不同容器名可以把 HCU0-3 租给服务，而 preflight 仍报告
free”的盲点。HCU4-7 使用 `renderD132-135`，不会被误判。

启动 Qwen：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 bash scripts/k100ai/serve-k100ai.sh qwen
```

默认选择 `bin/strata` 快速档。需要切到包内的当前 Qwen 引擎时：

```bash
K100AI_QWEN_ENGINE=bin/strata-0.1.55 \
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 bash scripts/k100ai/serve-k100ai.sh qwen
```

启动 DeepSeek V4.1：

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18097 bash scripts/k100ai/serve-k100ai.sh dsv41
```

launcher 固定以下边界：

- 主机只监听 `127.0.0.1:<PORT>`；容器内监听 `0.0.0.0`
- Qwen 只映射 `/dev/kfd` 和 `renderD128`
- DeepSeek 只映射 `/dev/kfd` 和 `renderD128-131`
- 已存在同名容器时以退出码 4 拒绝，不替换原容器
- 运行中的研究容器会使启动以退出码 3 拒绝
- 退出时只停止并删除本次 launcher 创建的容器 ID
- 不再自动停止或恢复研究容器

前台 Ctrl-C 和发送给 launcher 的 `SIGTERM` 均已实测。INT 返回 130，
TERM 返回 143；两种路径都会停止、删除本次服务容器，预置容器和研究容器
保持原状态。服务退出后再次执行 preflight。

## API 验收

Qwen：

```bash
BASE_URL=http://127.0.0.1:18096 \
MODEL=qwen3.8-flash-next \
EXPECT_DRAFT_METRICS=1 \
OUT=api-qwen \
  bash scripts/k100ai/validate-api.sh
```

DeepSeek：

```bash
BASE_URL=http://127.0.0.1:18097 \
MODEL=deepseek-v4.1-flash \
OUT=api-dsv41 \
  bash scripts/k100ai/validate-api.sh
```

两个配置都不使用 API key，因此主机监听必须保持 `127.0.0.1`。

## 模型和精度

Qwen 使用现有 IQ3_XXS pack、native GGUF、PLE GGUF、24,576-rank expert
profile，`--expert-cache 24576`、`--prefill 20480`、`--spec 6`、
`--spec-min-p 0.7`、int8 KV 和 262,144-token 上限，并使用外置 787 MiB
MTP runtime 的 `--mtp` draft layer。快速档启用 gfx928 prompt MMAC、
F16 prompt GEMM、fused expert 和 HC upmix，因此不是 bitwise profile。
MTP runtime 不进入发布包；launcher 默认从
`$K100AI_ROOT/assets/qwen3.8-flash-next-iq3-xxs/mtp/rt` 读取，也可通过
`K100AI_QWEN_MTP_RT` 指向同一份已验证 runtime。runtime 的四个文件及
SHA-256 见
[PROVENANCE.md](PROVENANCE.md)。

当前 0.1.55 引擎包含 `8544d2fa` 的 gfx928 int8 expert MMAC 和
`8c413cae` 的显式 gfx928 DeepSeek MMAC/QSA 路径，并保留为
Qwen 对照二进制：

- gate/up：`IQ3_XXS,IQ3_S,IQ4_XS`
- down：`Q2_0`

空值、未设置和 `all` 都映射到上述安全集合；`IQ2_XXS`、`IQ2_XS` gate
精确路径仍可显式用于诊断，但不是生产默认值。BF16 HC upmix MMAC 会改变
归约顺序和 logits。默认快速档明确按“允许轻微精度漂移”验收，不用
bitwise 门槛掩盖这一差异。

DeepSeek 使用 `--experts 6`、`--vision off`、`--tp 4`、
`--max-context 1048576`、`--prefill-chunk 1024`、
`--dspark on --dspark-conf 0.65`。它使用包内冻结的
`data/dsv41/profile-2.00.bin` 和 `--profile-readonly`，不会在请求后改写
placement profile。

完整正确性、性能、API、机器可复现证据和限制见
[VALIDATION.md](../../VALIDATION.md)。构建与模型 provenance 见
[PROVENANCE.md](PROVENANCE.md)。

模型和 pack 可由发布内脚本重新生成：

```bash
bash scripts/k100ai/fetch-qwen-iq3xxs.sh /data/k100ai-assets/qwen3.8-flash-next-iq3-xxs
bash scripts/k100ai/fetch-dsv41-iq2xxs.sh /data/models/dsv41-2.00
bash scripts/k100ai/prepare-dsv41-pack.sh \
  /data/models/dsv41-2.00 \
  /data/models/dsv41-2.00/maya-ds4-dspark
```

脚本固定 ModelScope 文件 revision，并使用大小与 SHA-256 双重校验。模型
权重和 MTP runtime 始终位于发布归档之外。
