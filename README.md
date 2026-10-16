# Strata for K100AI

面向海光 K100AI（gfx928、64 GiB 显存）的 **Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS** 推理优化与部署配置。

当前版本为 **[v0.1.0-rc.5](https://github.com/DocPang/strata-k100ai/releases/tag/v0.1.0-rc.5)**，分发 2026-10-16 已完成分档性能测试的固定引擎。引擎 SHA-256 为 `d87a4678a21b3c482658e53e6e766f7e54c2033853e6367bbf39e8e088ca4d6b`。后续平台修复和 16 位 topk 实验构建未纳入本版本。

运行包约 8.5 MiB，包含推理程序、HTTP 服务与模型转换工具；基础镜像、驱动和模型权重单独准备。支持直接复用本机 DTK 镜像，也可将运行包加入基础镜像生成本地派生镜像。

## 累计优化内容

当前运行包包含早期 K100AI 适配与内核优化，以及本次 IQ3_XXS 长上下文优化。下表列出发布版本采用的累计内容；部分属于已有内核的 gfx928 适配、分派或参数选择，不代表每一项算法均由本项目从零实现。

| 范围 | 发布版本采用的优化 |
| --- | --- |
| K100AI 平台适配 | gfx928 HIP 兼容层、DTK 编译链接、指令与架构分派适配，以及 ggml/llama.cpp MMQ 衔接 |
| 量化稠密计算 | Q6_K/Q8_0 packed 布局、Q6 wave packing 与准备路径；按实际张量类型命中，不将 Q6 优化归到 XXS 的 Q5_K 输出头 |
| MoE 专家布局与分派 | 常驻专家排序、native/MMQ/fused 路径协同，减少不必要的数据重排 |
| MoE gate/up | exact 展开和 MMAC 路径，增加 IQ2_XS/IQ2_XXS 支持，使该 XXS 权重的 48 层 gate/up 全部具备 exact 路径 |
| MoE down 投影 | IQ4_NL exact preexpand、Q2_0 展开与 int8 MMAC、staging 及行 tile 调整；按格式和形状分派 |
| GEMM 形状与布局 | 按专家组行数选择计算 tile，采用 rocBLAS 步长 padding，并保留 DTK BLAS 接口适配 |
| GDN 递归与写回 | 采用八 lane 协作递归路径，关闭无消费者的 FP32 中间结果写回 |
| Hyper-connection 计算 | gfx928 upmix 融合、split/staged read 与 down prefetch，减少分离计算和数据访问开销 |
| 激活量化复用 | 对唯一 token 行量化一次，再按专家索引 gather；融合 SwiGLU 与量化，减少重复量化和中间读写 |
| Prefill 工作区 | 条件分配 scratch、复用 embedding 缓冲、共享 gated 工作区和暂存 ring；显存记账与实际布局协同 |
| QSA scores 访问与计算 | 调整网格轴序改善复用，使用 gfx928 MMAC，并通过 LDS 合并装载 query/key |
| QSA topk 与调度 | lane-major 访问；selection 批次按最大上下文缓冲尺寸分档，不使用固定 1024 覆盖 |
| Selection 缓冲 | FP16 scores 将分数缓冲从 4 字节缩为 2 字节；该步骤改变中间精度，正确性范围见成绩说明 |
| Attention 数据路径 | INT8 V 暂存与合并读取、K 暂存模板化、页索引地址计算和 lane/cell 路径调整 |
| PLE 访问 | prelaunch 调度与 mmap 数据访问配合 |
| Decode 稠密计算 | MMVQ wave/packed 路径，并启用已测有收益的 MMVQ LDS 激活暂存 |

**本次发布的主要新增项**是 MMAC scores 的 LDS 合并装载、IQ2 exact 扩展、FP16 scores 缓冲、K 暂存模板化及 MMVQ LDS 的启用与验证。其余内容从此前版本沿用，构成本版本测试的基础。

部署层面另采用 **IQ3_XXS、24576 专家槽全部驻留、256K INT8 KV 全显存驻留、prefill=30602、MTP spec=4、单槽 FIFO**。多卡通过独立实例分发请求。KV、MTP、会话缓存和 API 服务基于上游能力，这里提供 K100AI 的配置与验证记录；默认关闭会话缓存以复现冷 prefill。

[完整优化清单](docs/OPTIMIZATIONS.md)逐项说明实现位置、启用方式和测试范围。未启用的实验路径、否定结果及后续研究构建不计入上述清单；单项收益不能相加作为累计加速比。

## 实测性能

环境：K100AI 单卡、IQ3_XXS、最大上下文 262144、INT8 KV 全显存驻留、expert-cache=24576、prefill=30602、spec=4。每个测试臂运行两次完整 prefill，并在同一批次运行前后参考配置。会话缓存关闭。

| 输入档位 | 实际输入 token | 优化配置平均耗时 | 有效吞吐 | 相对同批参考配置的耗时下降 |
| --- | ---: | ---: | ---: | ---: |
| 32K | 30603 | **15.71 秒** | **1948 tok/s** | 约 11.5% |
| 128K | 126857 | **70.83 秒** | **1791 tok/s** | 约 17.8% |
| 250K | 249986 | **146.79 秒** | **1703 tok/s** | 约 23.3% |

250K 第二批复测平均为 **146.86 秒**。同一输入改用 prefill=20480 时平均为 **149.75 秒**。这些数字均来自本版本分发的二进制；它们不是不同长度历史请求间的加速比。

Decode 的独立对照使用 32K 输入、128-token 生成：启用 MMVQ LDS 后，窗口平均耗时由 41.98 ms 降为 41.04 ms（约 −2.2%）；四个优化样本的生成速率为 77.33、81.94、77.65、82.23 tok/s。提示内容和草稿接受率影响生成速度，不将最高样本作为通用吞吐保证。

250K 的已测对照输出 SHA 一致。短输入记录存在一个 EOS 后 token 差异；已有结果不构成所有提示或不同量化权重的输出等价证明。多任务质量、长时间稳定性和干净源码重建尚未完成。详见 [成绩与原始汇总](docs/BENCHMARKS.md)、[发布验证](docs/VALIDATION.md)。

## 安装与运行

以下步骤在 **Linux x86_64 算力服务器**执行，采用原生 Docker 命令。按顺序在同一个 Bash 会话中执行；换终端后需重新设置 `PROJECT_DIR`、`DATA_DIR`、`BASE_IMAGE` 和 API key。

### 1. 检查基础环境

宿主机应已安装 Docker Engine、与 K100AI 匹配的驱动和 HyHAL。需要 `git`、`curl`、`openssl`、`tar`、`sha256sum`；Ubuntu 缺少这些工具时可安装：

```bash
sudo apt-get update
sudo apt-get install -y git curl openssl tar coreutils

docker info
ls -l /dev/kfd /dev/dri/renderD*
ls -ld /opt/hyhal /opt/hyhal/lib
df -h .
free -h
```

两个 GGUF 分片合计约 75.84 GB；生成专家 pack 还需约 40 GiB，以及 MTP 和转换中间文件的空间。建议为整个工作目录预留至少 200 GiB 可用磁盘；转换与运行还需要充足的主机内存。若数据盘位于其他路径，下一步将 `DATA_DIR` 改到该盘。

### 2. 获取本版本部署文件并建立目录

```bash
mkdir -p "$HOME/strata-deploy"
cd "$HOME/strata-deploy"
git clone --depth 1 --branch v0.1.0-rc.5 \
  https://github.com/DocPang/strata-k100ai.git strata-k100ai-rc5
cd strata-k100ai-rc5

export PROJECT_DIR="$PWD"
export DATA_DIR="$PROJECT_DIR/data"
mkdir -p "$DATA_DIR/models/IQ3_XXS" "$DATA_DIR/packs" "$DATA_DIR/mtp"
```

也可从 [Release](https://github.com/DocPang/strata-k100ai/releases/tag/v0.1.0-rc.5) 下载 `deployment-v0.1.0-rc.5.tar.gz` 解压，进入解压目录后设置上述两个变量。

### 3. 准备国内 DTK 基础镜像

```bash
export BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
docker pull "$BASE_IMAGE"
```

如果本机已经有相同内容的镜像，可直接指定本地名称：

```bash
# 将下面的名称替换为本机已有的 DTK 镜像标签。
export BASE_IMAGE=my-local-dtk:2604
docker image inspect "$BASE_IMAGE" --format '{{.Id}}'
```

以上为两种选择，使用本地镜像时才执行第二段。经过验证的镜像配置 ID 为 `sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad`。镜像来自海光国内仓库；下述模型权重和 MTP 从魔塔下载。

### 4. 下载并校验优化运行包

```bash
cd "$PROJECT_DIR"
curl -fL --retry 3 -o strata-xxs-20261016-linux-amd64.tar.gz \
  https://github.com/DocPang/strata-k100ai/releases/download/v0.1.0-rc.5/strata-xxs-20261016-linux-amd64.tar.gz
sha256sum -c engines/strata/xxs-20261016/artifact.sha256
tar -xzf strata-xxs-20261016-linux-amd64.tar.gz
(cd runtime && sha256sum -c FILES.sha256)
sha256sum runtime/bin/strata
```

最后一条应显示本页开头的 `d87a4678a21b…` 完整摘要。所有校验通过后再继续。

### 5. 从魔塔下载两个 IQ3_XXS 权重分片

模型页面：[ISTA-DASLab / Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://modelscope.cn/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/files)。以下命令支持断点续传，下载中断后可重新执行同一条命令。

```bash
curl -fL --retry 5 -C - \
  -o "$DATA_DIR/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf" \
  'https://modelscope.cn/api/v1/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/repo?Revision=master&FilePath=IQ3_XXS%2FQwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf'

curl -fL --retry 5 -C - \
  -o "$DATA_DIR/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf" \
  'https://modelscope.cn/api/v1/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/repo?Revision=master&FilePath=IQ3_XXS%2FQwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf'

(cd "$DATA_DIR/models/IQ3_XXS" && \
  sha256sum -c "$PROJECT_DIR/profiles/next-iq3xxs-single/weights.sha256")
```

两分片必须在同一目录，文件名保持不变。校验表固定了本版本测试使用的文件；魔塔仓库后续更新导致校验失败时，应先确认文件版本。

### 6. 生成与权重匹配的 native pack

转换在 DTK 容器中执行，不需要宿主 Python，也不需要 GPU。此步骤从权重生成索引、专家数据及 tokenizer，不能直接复用 IQ3_S 的 pack。

```bash
docker run --rm --network none \
  --env-file "$PROJECT_DIR/engines/strata/xxs-20261016/environment.env" \
  --mount "type=bind,src=$PROJECT_DIR/runtime,dst=/opt/k100ai,readonly" \
  --mount "type=bind,src=$DATA_DIR/models/IQ3_XXS,dst=/models,readonly" \
  --mount "type=bind,src=$DATA_DIR/packs,dst=/out" \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" tools/iq_pack.py \
  --gguf /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --out /out/next-iq3xxs --experts-bin

ls -lh "$DATA_DIR/packs/next-iq3xxs"
test -f "$DATA_DIR/packs/next-iq3xxs/native_experts.txt"
test -d "$DATA_DIR/packs/next-iq3xxs/tokenizer"
```

`native_experts.txt` 在转换完成时写入。目录不完整或转换退出非零时，不继续启动服务。

### 7. 从魔塔准备 MTP 草稿模型

MTP 来源：[Qwen / Qwen3.8-Flash-Next](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/files)。提供的工具通过 HTTP Range 只下载所需 MTP 张量，无需下载整个 BF16 主模型。以下四条 Docker 命令依次执行。

下载所需张量：

```bash
docker run --rm --env STRATA_SOURCE=modelscope \
  --env-file "$PROJECT_DIR/engines/strata/xxs-20261016/environment.env" \
  --mount "type=bind,src=$PROJECT_DIR/runtime,dst=/opt/k100ai,readonly" \
  --mount "type=bind,src=$DATA_DIR/mtp,dst=/out" \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" tools/mtp_fetch.py fetch --out /out/bf16
```

验证张量的固定 SHA-256：

```bash
docker run --rm --network none \
  --env-file "$PROJECT_DIR/engines/strata/xxs-20261016/environment.env" \
  --mount "type=bind,src=$PROJECT_DIR/runtime,dst=/opt/k100ai,readonly" \
  --mount "type=bind,src=$DATA_DIR/mtp,dst=/out" \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" tools/mtp_fetch.py verify --out /out/bf16
```

将 MTP 专家打包为 Q2_0：

```bash
docker run --rm --network none \
  --env-file "$PROJECT_DIR/engines/strata/xxs-20261016/environment.env" \
  --mount "type=bind,src=$PROJECT_DIR/runtime,dst=/opt/k100ai,readonly" \
  --mount "type=bind,src=$DATA_DIR/mtp,dst=/out" \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" tools/mtp_pack.py --src /out/bf16 --experts q2_0 --out /out/mtp-q2_0.gguf
```

转换为推理引擎使用的 MTP runtime：

```bash
docker run --rm --network none \
  --env-file "$PROJECT_DIR/engines/strata/xxs-20261016/environment.env" \
  --mount "type=bind,src=$PROJECT_DIR/runtime,dst=/opt/k100ai,readonly" \
  --mount "type=bind,src=$DATA_DIR/mtp,dst=/out" \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" tools/mtp_rt.py --gguf /out/mtp-q2_0.gguf --out /out/runtime

ls -lh "$DATA_DIR/mtp/runtime"
```

若网络代理不支持 Range，下载工具会报错并停止；应更换可正确返回分段内容的网络路径，不能跳过张量校验。已有与上述流程匹配的 MTP runtime 时，可复用该目录。

### 8. 选择显卡并准备 API key

```bash
ls -l /dev/dri/by-path/
ls -l /dev/dri/renderD*

# 根据 PCI 地址与供应方设备工具确认目标卡，并检查是否已有任务占用。
export RENDER_DEVICE=/dev/dri/renderD128
export STRATA_API_KEY="$(openssl rand -hex 24)"
(umask 077; printf '%s\n' "$STRATA_API_KEY" > "$PROJECT_DIR/.strata-api-key")
```

只映射目标 render 节点，并保留宿主与容器中的相同路径。容器内部使用 `HIP_VISIBLE_DEVICES=0` 和 `ROCR_VISIBLE_DEVICES=0`。`renderD128` 是示例，不能据此假定所有服务器的设备编号相同。

### 9. 启动原生 Strata HTTP 服务

```bash
docker run --detach --name strata-next-iq3xxs \
  --device /dev/kfd:/dev/kfd --device "$RENDER_DEVICE:$RENDER_DEVICE" \
  --ulimit memlock=-1:-1 --shm-size=2g \
  --security-opt seccomp=unconfined \
  --tmpfs /var/log/hylog:rw,nosuid,nodev,mode=0755 \
  --publish 127.0.0.1:18640:8080 --env STRATA_API_KEY \
  --env-file "$PROJECT_DIR/engines/strata/xxs-20261016/environment.env" \
  --mount "type=bind,src=$PROJECT_DIR/runtime,dst=/opt/k100ai,readonly" \
  --mount "type=bind,src=$DATA_DIR/models/IQ3_XXS,dst=/models,readonly" \
  --mount "type=bind,src=$DATA_DIR/packs/next-iq3xxs,dst=/pack,readonly" \
  --mount "type=bind,src=$DATA_DIR/mtp/runtime,dst=/mtp,readonly" \
  --mount type=bind,src=/opt/hyhal,dst=/opt/hyhal,readonly \
  --mount "type=bind,src=$PROJECT_DIR/profiles/next-iq3xxs-single/strata.json,dst=/config/strata.json,readonly" \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" -m serve.server --engine strata \
  --config /config/strata.json --host 0.0.0.0 --port 8080
```

初次加载需要读取权重并建立专家缓存。服务仅绑定宿主 `127.0.0.1:18640`；跨机器访问应通过已有网关、反向代理或 SSH 转发接入。配置中的模型标识为 `qwen3.8-flash-next-strata`。

### 10. 检查加载并发送请求

```bash
docker logs --tail 100 strata-next-iq3xxs
curl --fail http://127.0.0.1:18640/health

# 新终端可从权限受限的本地文件恢复 key。
export STRATA_API_KEY="$(cat "$PROJECT_DIR/.strata-api-key")"
curl --fail http://127.0.0.1:18640/v1/models \
  -H "Authorization: Bearer $STRATA_API_KEY"

curl --fail http://127.0.0.1:18640/v1/chat/completions \
  -H "Authorization: Bearer $STRATA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-strata","messages":[{"role":"user","content":"请用一句话介绍你自己。"}],"max_tokens":128,"stream":true}'
```

`/health` 应显示 `loaded=true`。查看引擎日志可执行 `docker exec strata-next-iq3xxs tail -n 100 /tmp/strata-engine.log`。排查步骤、前台启动、停止容器和直接调用 C++ 引擎见 [运行与调试](docs/RUNNING.md)。

## 默认参数与调整

| 参数 | 本版本默认值 | 说明 |
| --- | --- | --- |
| `--max-context` | 262144 | 原生 256K 上限；输入、历史与输出共同占用上下文 |
| `--expert-cache` | 24576 | 保持全部专家驻留，避免回退到主机内存影响生成速度 |
| `--kv` / `--kv-resident` | int8 / 省略 | 完整 KV 驻留显存；不是 32K KV streaming 配置 |
| `--prefill` | 30602 | 本页三档成绩所用值；显存不足时先降到 20480，再降到 16384 |
| `--spec` / `--spec-min-p` | 4 / 0.70 | 本页测试使用的 MTP 参数；接受率随提示而变化 |
| `--vram-reserve-mib` | 2048 | 显存规划预留量；实际峰值仍需看日志与设备工具 |
| `--batch` / `--batch-mtp` | 省略 | 单槽 FIFO；多卡采用独立实例，不以单卡多槽增加并发 |
| `--prompt-cache` / `--conversation-cache-mib` | 0 / 0 | 默认关闭以便复现冷 prefill；按业务需要在验证后开启 |
| `STRATA_SEL_BATCH` | 不设置 | 使用引擎按上下文缓冲尺寸选择的默认值；不沿用旧版 1024 覆盖 |

修改原生配置时编辑 `profiles/next-iq3xxs-single/strata.json` 的 `args`；修改优化开关时编辑 `engines/strata/xxs-20261016/environment.env`。停止并删除自己的容器后，重新执行启动命令使参数生效。优先保留全部专家驻留，通过降低 prefill、上下文或关闭额外缓存平衡显存。

需要将运行文件固定到镜像中时，见 [手动构建派生镜像与离线准备](docs/INSTALL.md)。`install.sh`、`start.sh` 和 Compose 保留为可选入口，原生命令不依赖这些脚本。

## 文档与来源

[模型准备](docs/MODELS.md) · [参数参考](docs/PARAMETERS.md) · [显存说明](docs/MEMORY.md) · [完整成绩](docs/BENCHMARKS.md) · [优化说明](docs/OPTIMIZATIONS.md) · [源码恢复](docs/SOURCE_BUILD.md) · [验证范围](docs/VALIDATION.md)

本项目由独立维护者发布，基于 [Strata](https://github.com/Niko1221/Strata) 和 [llama.cpp](https://github.com/ggml-org/llama.cpp)，遵循 [MIT](LICENSE) 并保留[上游许可](engines/strata/LICENSE)。DTK/HyHAL 与模型权重遵循各自许可。
