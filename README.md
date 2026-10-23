# Strata for K100AI

面向海光 K100AI（gfx928、64 GiB 显存）的 **Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS** 推理优化与部署配置。

当前版本 **[v0.2.0](https://github.com/DocPang/strata-k100ai/releases/tag/v0.2.0)**，引擎 SHA-256 为 `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9`（45,674,152 B）。

本版把优化集**重基到上游 strata 0.2.2（`011c9e60`）**。rc.1–rc.7 的引擎谱系基于上游 `fb58e0db`，与本版不是同一条基线（两棵树有 333 个文件不同），两版基线不同，所以下面的差值既包含本版的优化，也包含上游基线本身的变化——这是"上一版对上一版"的口径，与 CHANGELOG 的既有做法一致；对照组是**同一张卡上的上一版 rc.7 引擎**。

## 本版成绩

测试条件：单卡 gfx928（renderD132）、IQ3_XXS、max-context=262144、expert-cache=24576、INT8 KV、prefill=30602、spec=6；对照组为同卡上一版 rc.7 引擎（`ec7021b0…`），**同会话交替、单请求串行**；43 个开关完整列在 [`engines/strata/xxs-20261023-k100ai022/manifest.json`](engines/strata/xxs-20261023-k100ai022/manifest.json) 的 `environment_required`。

### 预填（TTFT 的主体）

| 套件 | prompt token | 本版 | rc.7 | 比值 | 本版吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ctx32 | 30,603 | **14,232 / 13,947 ms** | 14,411 / 14,160 ms | **0.988 / 0.985** | 2,150 / 2,194 tok/s |
| lt | 7,556 | 4,497 / **3,851 ms** | 4,482 / 3,956 ms | 1.003 / **0.974** | 1,680 / 1,962 tok/s |
| ctx128 | 126,857 | **67,401 / 67,350 ms** | 68,356 / 68,280 ms | **0.986 / 0.986** | 1,882 / 1,884 tok/s |

每档两个数字是两个不同 prompt 的实测。**5/6 格领先，1 格 1.003**——lt 的第二个 prompt 是抖动格（该 prompt 上两引擎的 run-to-run 散布约 1%，三轮均值 0.998），按单次如实列出，不取最好样本。

### decode（端到端，每请求）

| 套件 | 本版 | rc.7 | 比值 |
| --- | ---: | ---: | ---: |
| ctx32 | **84.8 / 149.7 / 135.1 ms** | 89.8 / 152.1 / 136.3 ms | **0.944 / 0.984 / 0.991** |
| lt | **83.6 / 140.5 / 129.2 ms** | 87.5 / 141.2 / 129.4 ms | **0.955 / 0.995 / 0.998** |
| ctx128 | **85.5 / 369.5 / 321.7 ms** | 88.5 / 379.0 / 332.4 ms | **0.966 / 0.975 / 0.968** |

**9/9 格持平或领先**，首个窗口差距最大（0.944–0.966）：decode 图在 prompt 阶段就预热好了，对照侧是首次使用时才捕获。

同 T 对照（spec=2、43–45 个窗口、avg T 3.16 对 3.13）：**11.78 对 12.24 ms/token，约 85 对 82 tok/s（0.963）**。生产配置 spec=6 下窗口更大（avg T 4.75），每窗口 48.83 对 49.57 ms。

### 模块级

预填逐阶段 GPU 计时，5 次本版 / 4 次对照采样并做区间重叠检验：

- **领先**：hc 计算块 **0.89**（快 11%，预取路径的直接结果）、dequant **0.83**、combine+写回 **0.971**、qsa select 0.975、gate/up 0.995、gdn 递推 0.995
- **持平**（采样区间重叠）：qsa attn 1.000、qsa proj 1.001、gdn 出投影 1.002、down GEMM 0.998、router 1.004
- **唯一略高**：`gdn conv+gates` 1.012（+4 ms），其 kernel 与上游逐字节相同，属块边界归因

decode 在同 T 对照下：verify **0.993**、draft **0.96**、commit **0.19**。

### 两处如实记录的代价

1. **首个请求的预填更慢**：约 500–541 ms，对照组 303–307 ms。本版在 prompt 阶段预热 MTP 的 11 个 decode 图（11 × 16.4 ms），每进程只付一次、落在不计分的首个请求上；换来每个首 decode 窗口快约 86 ms。`STRATA_MTP_WARM_AT_START=0` 可关掉。
2. **ctx128 输出与上游不同**：`84f393bf…` 对 `a6bcf7fd…`。原因是 hc read 的 `[wd|wi]` 宽 GEMM（`STRATA_K100AI_HC_FUSE`，默认开）改变了该投影的累加序；置 `STRATA_K100AI_HC_FUSE=0` 后 30K logits 逐字节相同、ctx128 输出回到 `a6bcf7fd…`。这是两棵树之间**唯一**已知的数值差异，五个长度档的 top-1 全部一致。

## 累计优化内容

本版（v0.2.0）新增：

- **预填 down 投影的 prefetch 路径**：单 LDS plane、TH=128，配合单 plane 分块，使 5–8 token 窗口不再分两趟 launch。这是本版最大的一项，hc read 阶段从表观落后 38% 变为快 11%。
- **hc read 的 `[wd|wi]` 宽 GEMM**（`STRATA_K100AI_HC_FUSE`，默认开）：down 与 up/inject 两个投影合成一次宽 GEMM。
- **`moe_combine` 的 `SD` 特化**：按 `scale_down_rows` 是否存在做模板分派，去掉内层循环的每元素分支与乘法，combine 411 → 371 ms。
- **gate/up exact IQ2 路径**、wsg16 tile、down GEMM、IQ4 预展开。
- **decode 图预热**（`STRATA_MTP_WARM_AT_START`，默认开）：把 MTP 的 round/step 图捕获挪到 prompt 阶段。

rc.1–rc.7 累积（沿用上一版说明，未在本版重测）：`GEMM_Y_F16` 六族 16 位累加器、`TOPK_H16` / `TOPK_REG_MAX`、`Q6_WAVE_PACK_LDS`、`MMVQ_WV2_GROUP`、`SCORES_MMAC`、GDN/HC 流水线、专家缓存与工作区复用、rocBLAS Tensile solution 按形状锁定（`STRATA_GEMM_SOL`）等。逐项归属与已否决项见 [CHANGELOG](CHANGELOG.md)。

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
git clone --depth 1 --branch v0.2.0 \
  https://github.com/DocPang/strata-k100ai.git strata-k100ai-020
cd strata-k100ai-020

export PROJECT_DIR="$PWD"
export DATA_DIR="$PROJECT_DIR/data"
mkdir -p "$DATA_DIR/models/IQ3_XXS" "$DATA_DIR/packs" "$DATA_DIR/mtp"
```

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
curl -fL --retry 3 -o strata-xxs-20261023-k100ai022-linux-amd64.tar.gz \
  https://github.com/DocPang/strata-k100ai/releases/download/v0.2.0/strata-xxs-20261023-k100ai022-linux-amd64.tar.gz
sha256sum -c engines/strata/xxs-20261023-k100ai022/artifact.sha256
tar -xzf strata-xxs-20261023-k100ai022-linux-amd64.tar.gz
(cd runtime && sha256sum -c FILES.sha256)
sha256sum runtime/bin/strata
```

最后一条应显示本页开头的 `bdbfd934…` 完整摘要。所有校验通过后再继续。

也可以从源码自己编（可验证路径）：clone `v0.2.0` 与上游 `011c9e60`，覆盖 [`sources/xxs-20261023-k100ai022/`](sources/xxs-20261023-k100ai022/) 的 36 个文件，按 [引擎清单](engines/strata/xxs-20261023-k100ai022/manifest.json) 记录的配方编译，产出摘要应为 `bdbfd934…`。注意同一构建目录内两次干净编译逐字节一致（`da43917a…`），换目录后 SHA-256 会变而字节数不变，因此 `clean_rebuild_verified` 记为 `false`，细节见 [复现报告](sources/xxs-20261023-k100ai022/REBUILD_REPORT.md)。

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
  --env-file "$PROJECT_DIR/engines/strata/xxs-20261016-gold/environment.env" \
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
  --env-file "$PROJECT_DIR/engines/strata/xxs-20261016-gold/environment.env" \
  --mount "type=bind,src=$PROJECT_DIR/runtime,dst=/opt/k100ai,readonly" \
  --mount "type=bind,src=$DATA_DIR/mtp,dst=/out" \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" tools/mtp_fetch.py fetch --out /out/bf16
```

验证张量的固定 SHA-256：

```bash
docker run --rm --network none \
  --env-file "$PROJECT_DIR/engines/strata/xxs-20261016-gold/environment.env" \
  --mount "type=bind,src=$PROJECT_DIR/runtime,dst=/opt/k100ai,readonly" \
  --mount "type=bind,src=$DATA_DIR/mtp,dst=/out" \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" tools/mtp_fetch.py verify --out /out/bf16
```

将 MTP 专家打包为 Q2_0：

```bash
docker run --rm --network none \
  --env-file "$PROJECT_DIR/engines/strata/xxs-20261016-gold/environment.env" \
  --mount "type=bind,src=$PROJECT_DIR/runtime,dst=/opt/k100ai,readonly" \
  --mount "type=bind,src=$DATA_DIR/mtp,dst=/out" \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" tools/mtp_pack.py --src /out/bf16 --experts q2_0 --out /out/mtp-q2_0.gguf
```

转换为推理引擎使用的 MTP runtime：

```bash
docker run --rm --network none \
  --env-file "$PROJECT_DIR/engines/strata/xxs-20261016-gold/environment.env" \
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
  --env-file "$PROJECT_DIR/engines/strata/xxs-20261016-gold/environment.env" \
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
| `--spec` / `--spec-min-p` | 4 / 0.70 | 部署档默认值；本页成绩表用的是 `--spec 6`（同卡对照同参数），接受率随提示而变化 |
| `--vram-reserve-mib` | 2048 | 显存规划预留量；实际峰值仍需看日志与设备工具 |
| `--batch` / `--batch-mtp` | 省略 | 单槽 FIFO；多卡采用独立实例，不以单卡多槽增加并发 |
| `--prompt-cache` / `--conversation-cache-mib` | 0 / 0 | 默认关闭以便复现冷 prefill；按业务需要在验证后开启 |
| `STRATA_SEL_BATCH` | 不设置 | 使用引擎按上下文缓冲尺寸选择的默认值；不沿用旧版 1024 覆盖 |

修改原生配置时编辑 `profiles/next-iq3xxs-single/strata.json` 的 `args`；修改优化开关时按 `engines/strata/xxs-20261023-k100ai022/manifest.json` 的 `environment_required` 设置环境变量（本版实测用的 43 个键即该表）。停止并删除自己的容器后，重新执行启动命令使参数生效。优先保留全部专家驻留，通过降低 prefill、上下文或关闭额外缓存平衡显存。

需要将运行文件固定到镜像中时，见 [手动构建派生镜像与离线准备](docs/INSTALL.md)。`install.sh`、`start.sh` 和 Compose 保留为可选入口，原生命令不依赖这些脚本。

## 文档与来源

[模型准备](docs/MODELS.md) · [参数参考](docs/PARAMETERS.md) · [显存说明](docs/MEMORY.md) · [完整成绩](docs/BENCHMARKS.md) · [优化说明](docs/OPTIMIZATIONS.md) · [源码恢复](docs/SOURCE_BUILD.md) · [验证范围](docs/VALIDATION.md) · [v0.2.0 复现报告](sources/xxs-20261023-k100ai022/REBUILD_REPORT.md)

本项目由独立维护者发布，基于 [Strata](https://github.com/Niko1221/Strata) 和 [llama.cpp](https://github.com/ggml-org/llama.cpp)，遵循 [MIT](LICENSE) 并保留[上游许可](engines/strata/LICENSE)。DTK/HyHAL 与模型权重遵循各自许可。
