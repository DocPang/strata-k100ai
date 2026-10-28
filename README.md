# Strata for K100AI

面向海光 K100AI（gfx928、64 GiB 显存）的 **Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS** 推理优化与部署配置。

当前版本 **[v0.2.1](https://github.com/DocPang/strata-k100ai/releases/tag/v0.2.1)**，文本引擎 SHA-256 为 `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9`（45,674,152 B），运行包 SHA-256 为 `dc6a0e969e3d92c1ba54de5545f6a97f37cd62492297e50ad3d3b02356ef9aba`（29,472,859 B）。

v0.2.0 把优化集**重基到上游 strata 0.2.2（`011c9e60`）**；rc.1–rc.7 的引擎谱系基于 `fb58e0db`，与该版不是同一条基线。下面的文本成绩是 v0.2.0 的同卡对照结果，v0.2.1 未改文本路径，因此继续适用。

v0.2.1 在 v0.2.0 的文本引擎之上增加 **GPU 多模态视觉**：新增 HIP 图片编码器 `runtime/bin/strata-vision`，图片先编码成 embeddings 再进入文本模型。启用方式、性能与验证见 [多模态视觉](docs/MULTIMODAL.md)。

## 文本成绩（v0.2.0 引擎，v0.2.1 同）

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

## 多模态（v0.2.1）

v0.2.1 新增 GPU 图片编码器 `runtime/bin/strata-vision`（HIP，gfx928），文本引擎 `runtime/bin/strata` 不变。编码器是独立常驻进程，约 2.1 GiB 显存；单卡加载后仍余约 5.27 GiB。

| 项目 | 实测 |
| --- | ---: |
| 760×520 图（475 image tokens）GPU 编码 | **1.72 s** |
| 同图 CPU 编码（8 线程对照） | 18.0 s |
| 同图二次请求（编码缓存） | 约 1.8 s 端到端 |
| 1800×1200 图（1024 tokens） | 12.5 s 端到端（含 309 tokens 生成） |
| GPU vs CPU embeddings | cosine mean 0.999293 / rel-L2 3.24% |

启用：

```bash
# 1) 把 mmproj 放进 MODEL_DIR（与两个 GGUF 分片同目录）
#    mmproj-Qwen3.8-Flash-Next-BF16.gguf（866 MiB）
# 2) deployment.env
VISION_ENABLED=1
VISION_MAX_TOKENS=1024
# 3) 启动
./start.sh
```

`/health` 出现 `"images": true` 即加载成功。原生 Docker 启动时，在 `strata.json` 中加 `"vision"` 段即可；完整字段、API 示例、性能与限制见 [docs/MULTIMODAL.md](docs/MULTIMODAL.md)，编码器源码与补丁见 [sources/vision-20261028](sources/vision-20261028/)。

## 累计优化内容

当前运行包包含早期 K100AI 适配与内核优化，以及本次 IQ3_XXS 长上下文优化。下表列出发布版本采用的累计内容；部分属于已有内核的 gfx928 适配、分派或参数选择，不代表每一项算法均由本项目从零实现。标注版本号的是该项的引入版本，未标注的是 rc.1 之前既已存在。

| 范围 | 发布版本采用的优化 |
| --- | --- |
| K100AI 平台适配 | gfx928 HIP 兼容层、DTK 编译链接、指令与架构分派适配，以及 ggml/llama.cpp MMQ 衔接 |
| 量化稠密计算 | Q6_K/Q8_0 packed 布局、Q6 wave packing 与准备路径；按实际张量类型命中，不将 Q6 优化归到 XXS 的 Q5_K 输出头 |
| MoE 专家布局与分派 | 常驻专家排序、native/MMQ/fused 路径协同，减少不必要的数据重排 |
| MoE gate/up | exact 展开和 MMAC 路径，增加 IQ2_XS/IQ2_XXS 支持，使该 XXS 权重的 48 层 gate/up 全部具备 exact 路径；**v0.2.0** 归入 wsg16 tile 分派 |
| MoE down 投影 | IQ4_NL exact preexpand、Q2_0 展开与 int8 MMAC、staging 及行 tile 调整；按格式和形状分派；**v0.2.0** 改为按单 LDS plane 分块（5–8 token 窗口不再分两趟 launch） |
| MoE 结果合并 | **v0.2.0**：`moe_combine` 按 `scale_down_rows` 是否存在做模板特化，内层循环去掉每元素分支与乘法 |
| GEMM 形状与布局 | 按专家组行数选择计算 tile，采用 rocBLAS 步长 padding，并保留 DTK BLAS 接口适配；**v0.1.0-rc.7** 增加按形状锁定 Tensile solution（`STRATA_GEMM_SOL`） |
| GDN 递归与写回 | 采用八 lane 协作递归路径，关闭无消费者的 FP32 中间结果写回 |
| Hyper-connection 计算 | gfx928 upmix 融合、split/staged read 与 down prefetch，减少分离计算和数据访问开销；**v0.2.0** 增加 down 投影的 prefetch kernel（单 plane、TH=128）与 hc read 的 `[wd|wi]` 宽 GEMM |
| 激活量化复用 | 对唯一 token 行量化一次，再按专家索引 gather；融合 SwiGLU 与量化，减少重复量化和中间读写 |
| Prefill 工作区 | 条件分配 scratch、复用 embedding 缓冲、共享 gated 工作区和暂存 ring；显存记账与实际布局协同 |
| QSA scores 访问与计算 | 调整网格轴序改善复用，使用 gfx928 MMAC，并通过 LDS 合并装载 query/key；合并批次由 1 提至 2（rc.6） |
| QSA topk 与调度 | lane-major 访问；selection 批次按最大上下文缓冲尺寸分档；topk 走 2-pass h16 核并用寄存器缓存最大值（rc.6） |
| Selection 缓冲 | FP16 scores 将分数缓冲从 4 字节缩为 2 字节；该步骤改变中间精度，正确性范围见成绩说明 |
| Attention 数据路径 | INT8 V 暂存与合并读取、K 暂存模板化、页索引地址计算和 lane/cell 路径调整 |
| PLE 访问 | prelaunch 调度与 mmap 数据访问配合 |
| Decode 稠密计算 | MMVQ wave/packed 路径、MMVQ LDS 激活暂存与 WV2 分组；Q6K packed-wave 走 LDS |
| 稠密 GEMM 累加器 | GEMM_Y_F16：qkv / qf / sg / su / sd / z 六个投影族的 GEMM Y 累加器改走 16 位路径（rc.6） |
| MTP 图调度 | **v0.2.0**：round/step 图在 prompt 阶段预热（`STRATA_MTP_WARM_AT_START`），把 11 次图上传挪出首个 decode 窗口 |
| 多模态视觉 | **v0.2.1**：HIP 图片编码器 `strata-vision`（llama.cpp `mtmd`），GPU 编码 475 tokens 1.72 s（CPU 18.0 s），最大 1024 image tokens；文本引擎不变 |

### 各版本新增项

- **v0.2.1（当前）**：新增 GPU 图片编码器 `strata-vision`（HIP/gfx928），支持 1024 image tokens；修正 DTK 下 norm launch bounds 导致的静默算错，并绕过 cooperative-groups 头文件兼容问题。文本引擎与 v0.2.0 相同。
- **v0.2.0**：down 投影 prefetch 路径与单 plane 分块、hc read 的 `[wd|wi]` 宽 GEMM、`moe_combine` 的 `SD` 特化、decode 图预热。前两项合计把 hc read 阶段从相对 rc.7 的表观落后 38% 变成快 11%。
- **v0.1.0-rc.7**：rocBLAS Tensile solution 按形状锁定（`STRATA_GEMM_SOL`），以及与 `STRATA_DENSE_TN` 的互斥约束。
- **v0.1.0-rc.6**：16 位 GEMM Y 累加器 `GEMM_Y_F16`、h16 topk 核与寄存器最大值缓存 `TOPK_H16`/`TOPK_REG_MAX`、scores 合并批次 `SCORES_MMAC_NB` 翻倍、Q6K packed-wave LDS、MMVQ WV2 分组。
- **v0.1.0-rc.5**：MMAC scores LDS 合并装载、IQ2 exact 扩展、FP16 scores 缓冲、K 暂存模板化、MMVQ LDS。
- **更早（c4 → rc.4）**：平台适配、量化布局、MoE 布局与分派、GDN/HC 递归、工作区与 attention 数据路径等，以上表未标注版本的行。

部署层面另采用 **IQ3_XXS、24576 专家槽全部驻留、256K INT8 KV 全显存驻留、prefill=30602、MTP spec=4、单槽 FIFO**。多卡通过独立实例分发请求。KV、MTP、会话缓存和 API 服务基于上游能力，这里提供 K100AI 的配置与验证记录；默认关闭会话缓存以复现冷 prefill。

### 历代成绩

各版本口径不同（prefill chunk、spec、测试协议都不一样），**不可跨行直接相减**；每行只与它自己的对照组比较。

| 版本 | 对照 | 负载 | 变化 | 口径 |
| --- | --- | --- | --- | --- |
| v0.1.0-rc.6 | rc.5 | 32K / 128K / 250K | −4.70% / −6.88% / −6.49% | 同批 A/B/A，各用自己的发布环境 |
| v0.1.0-rc.7 | rc.6 | ctx128（126,857 token） | −1.64% | 交叉 A/B，控制组极差 0.003% |
| **v0.2.0** | **rc.7** | 30,603 / 7,556 / 126,857 | 预填 **0.988 / 0.974–1.003 / 0.986**；decode **0.941–0.998** | 同卡同会话交替，单请求串行 |
| **v0.2.1** | v0.2.0 文本引擎 | 475 / 1024 image tokens | 编码 **1.72 s**（CPU 18.0 s）；余弦 0.999293；文本性能不变 | 单卡 gfx928，GPU HIP 编码器 |

[完整优化清单](docs/OPTIMIZATIONS.md)逐项说明实现位置、启用方式和测试范围，[完整成绩](docs/BENCHMARKS.md)保留各版本原始记录。未启用的实验路径、否定结果及后续研究构建不计入上述清单；单项收益不能相加作为累计加速比。

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

两个 GGUF 分片合计约 75.84 GB；开启视觉还需一个 866 MiB 的 mmproj 文件。生成专家 pack 还需约 40 GiB，以及 MTP 和转换中间文件的空间。建议为整个工作目录预留至少 200 GiB 可用磁盘；转换与运行还需要充足的主机内存。若数据盘位于其他路径，下一步将 `DATA_DIR` 改到该盘。

### 2. 获取本版本部署文件并建立目录

```bash
mkdir -p "$HOME/strata-deploy"
cd "$HOME/strata-deploy"
git clone --depth 1 --branch v0.2.1 \
  https://github.com/DocPang/strata-k100ai.git strata-k100ai-021
cd strata-k100ai-021

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
  https://github.com/DocPang/strata-k100ai/releases/download/v0.2.1/strata-xxs-20261023-k100ai022-linux-amd64.tar.gz
sha256sum -c engines/strata/xxs-20261023-k100ai022/artifact.sha256
tar -xzf strata-xxs-20261023-k100ai022-linux-amd64.tar.gz
(cd runtime && sha256sum -c FILES.sha256)
sha256sum runtime/bin/strata runtime/bin/strata-vision
```

文本引擎应显示本页开头的 `bdbfd934…`；图片编码器应显示 `9c4dc14303e508c0c45814d886d56456a0b0b70a8732e04f9555c0491fdb41d4`。所有校验通过后再继续。

文本引擎也可从源码自己编（可验证路径）：clone `v0.2.1` 与上游 `011c9e60`，覆盖 [`sources/xxs-20261023-k100ai022/`](sources/xxs-20261023-k100ai022/) 的 36 个文件，按 [引擎清单](engines/strata/xxs-20261023-k100ai022/manifest.json) 记录的配方编译，产出摘要应为 `bdbfd934…`。图片编码器的构建见 [sources/vision-20261028](sources/vision-20261028/)。注意同一构建目录内两次干净编译逐字节一致（`da43917a…`），换目录后 SHA-256 会变而字节数不变，因此 `clean_rebuild_verified` 记为 `false`，细节见 [复现报告](sources/xxs-20261023-k100ai022/REBUILD_REPORT.md)。

### 5. 从魔塔下载 IQ3_XXS 权重与视觉投影

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

多模态需要同一目录下的视觉投影文件（仅开启 `VISION_ENABLED=1` 时需要）：

```bash
curl -fL --retry 5 -C - \
  -o "$DATA_DIR/models/IQ3_XXS/mmproj-Qwen3.8-Flash-Next-BF16.gguf" \
  'https://modelscope.cn/api/v1/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/repo?Revision=master&FilePath=mmproj-Qwen3.8-Flash-Next-BF16.gguf'

printf '%s  %s\n' \
  b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0 \
  "$DATA_DIR/models/IQ3_XXS/mmproj-Qwen3.8-Flash-Next-BF16.gguf" | sha256sum -c -
```

### 6. 生成与权重匹配的 native pack

转换在 DTK 容器中执行，不需要宿主 Python，也不需要 GPU。此步骤从权重生成索引、专家数据及 tokenizer，不能直接复用 IQ3_S 的 pack。

说明：第 6/7 步的转换工具容器沿用 `engines/strata/xxs-20261016-gold/environment.env`，因为它带有 `STRATA_GGUF_PY`；第 9 步的服务启动使用当前 `engines/strata/xxs-20261023-k100ai022/environment.env`。

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
  --env-file "$PROJECT_DIR/engines/strata/xxs-20261023-k100ai022/environment.env" \
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

视觉原生启动：确认上面的模型目录（`$DATA_DIR/models/IQ3_XXS`）内有 mmproj，把上面命令里的
`profiles/next-iq3xxs-single/strata.json` 换成 `profiles/next-iq3xxs-single/strata-vision.json`
（该文件已含 `vision` 段），其余参数不变。也可以改用下面的辅助入口流程。

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

### 11. 视觉启动（辅助入口完整流程）

如果使用 `install.sh` + `start.sh`，先确认 `deployment.env` 里的 `MODEL_DIR` / `PACK_DIR` / `MTP_DIR` 指向本项目的 `$DATA_DIR`，并已下载 mmproj：

```bash
cd "$PROJECT_DIR"
test -f "$DATA_DIR/models/IQ3_XXS/mmproj-Qwen3.8-Flash-Next-BF16.gguf"

./install.sh
sed -i 's/^VISION_ENABLED=.*/VISION_ENABLED=1/' deployment.env
grep -E '^(VISION_ENABLED|VISION_MAX_TOKENS)=' deployment.env

./start.sh
curl -s http://127.0.0.1:8080/health
```

`/health` 返回 `"images": true` 即编码器已加载；若改过 `PORT`，把地址换成对应端口。发一张图验证：

```bash
IMG_B64="$(base64 -w0 /path/to/image.png)"
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $(cat "$PROJECT_DIR/.strata-api-key")" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3.8-flash-next-strata\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"读出图片标题\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$IMG_B64\"}}]}],\"max_tokens\":256}"
```

更多字段与 Python 示例见 [MULTIMODAL.md](docs/MULTIMODAL.md)。

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

[多模态视觉](docs/MULTIMODAL.md) · [模型准备](docs/MODELS.md) · [参数参考](docs/PARAMETERS.md) · [显存说明](docs/MEMORY.md) · [完整成绩](docs/BENCHMARKS.md) · [优化说明](docs/OPTIMIZATIONS.md) · [源码恢复](docs/SOURCE_BUILD.md) · [验证范围](docs/VALIDATION.md) · [v0.2.0 文本引擎复现报告](sources/xxs-20261023-k100ai022/REBUILD_REPORT.md)

本项目由独立维护者发布，基于 [Strata](https://github.com/Niko1221/Strata) 和 [llama.cpp](https://github.com/ggml-org/llama.cpp)，遵循 [MIT](LICENSE) 并保留[上游许可](engines/strata/LICENSE)。DTK/HyHAL 与模型权重遵循各自许可。
