# 完整安装与从零复现指南（v0.4.0）

[返回首页](../../README.md) · [运行与多卡缓存池](RUNTIME.md) · [v0.4.0 更新说明](release-notes/v0.4.0.md)

本指南面向**从零开始部署与复现**的工程师。所有外部权重、视觉投影、投机解码草稿头均有明确出处与校验和；全程使用**原生命令**，不隐藏在一键黑盒脚本后，方便按实际机器路径与显存自由修改。

---

## 1. 环境与资源准备

| 项目 | Qwen3.8-Flash-Next（单卡） | DeepSeek-V4.1-Flash（双卡） |
| :--- | :--- | :--- |
| **加速卡** | 1× 海光 K100AI（64 GiB HBM） | 2× 海光 K100AI（128 GiB HBM） |
| **驱动与运行库** | 宿主机已安装海光驱动与 `/opt/hyhal` | 宿主机已安装海光驱动与 `/opt/hyhal` |
| **主机内存** | 建议 ≥ 64 GiB（开启多卡缓存池建议 ≥ 128 GiB） | 建议 ≥ 128 GiB |
| **可用磁盘** | 约 150 GB（GGUF 83.6 GB + Pack 52 GB + MTP/mmproj） | 约 260 GB（NVMe 推荐） |

先在宿主机确认设备与 HyHAL 路径正常：

```bash
ls -l /dev/kfd /dev/dri/renderD*
ls -ld /opt/hyhal /opt/hyhal/lib
docker info >/dev/null && echo "Docker OK"
```

---

## 2. 选择海光 DTK Docker 镜像（不挑镜像）

本项目只依赖容器内的海光 DTK 运行时（HIP / rocBLAS / RCCL）与 Python 3.10+，**对具体镜像版本没有强绑定**。如果历史文档中的 `harbor.sourcefind.cn:5443` 不可达，直接改用以下任一方式即可：

1. **优先使用宿主机或内网已有的任意海光 DTK 镜像**（`docker images` 查看）；
2. **或从光源公开镜像站拉取基础镜像**（[sourcefind.cn](https://www.sourcefind.cn/)）：

```bash
# 示例（可替换为本机已有的任意 DTK 25.04 / 26.04 镜像）
export DTK_IMAGE="image.sourcefind.cn:5000/dcu/admin/base/pytorch:2.4.1-ubuntu22.04-dtk25.04-py3.10"
docker pull "$DTK_IMAGE"
```

后续所有命令都只引用 `$DTK_IMAGE`，换镜像无需改动任何脚本。

---

## 3. 获取 v0.4.0 发布包

```bash
export REL_DIR=/data/strata-k100ai-v0.4.0
mkdir -p "$REL_DIR" && cd "$REL_DIR"

# 从 GitHub Release 下载 v0.4.0 发布包并校验（具体文件名以 Release 页为准）
# https://github.com/DocPang/strata-k100ai/releases
```

解压后核心目录包含：
- `bin/`：引擎二进制（`strata`、`strata-vision-hip`、`strata-dsv41`、`dflash2.q4.pack`）
- `configs/k100ai/`：Qwen 与 DeepSeek 的原生 JSON 配置模板
- `serve/`、`tools/`：OpenAI/Anthropic API 服务与资产转换工具
- `session-manager/`、`sticky-router/`：多卡共享缓存池组件

---

## 4. 准备 Qwen3.8-Flash-Next（IQ3_S）资产

下面按步骤从官方仓库下载并转换，支持断点续传（`curl -C -`）。

```bash
export ASSETS_DIR=/data/k100ai-assets/qwen3.8-flash-next-iq3-s
mkdir -p "$ASSETS_DIR"/{model/IQ3_S,pack/iq3s,mtp/source,mtp/rt}
```

### 4.1 下载主权重与视觉投影 `mmproj`

- **出处**：[ModelScope · ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://www.modelscope.cn/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)（固定 revision `6ac9406afe74db316b11a3865a3382c7bedbcf34`）

```bash
MS_BASE="https://www.modelscope.cn/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/6ac9406afe74db316b11a3865a3382c7bedbcf34"

# 主模型分片 1（54.8 GB）
curl -fL --retry 5 -C - \
  -o "$ASSETS_DIR/model/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" \
  "$MS_BASE/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"

# 主模型分片 2（28.8 GB）
curl -fL --retry 5 -C - \
  -o "$ASSETS_DIR/model/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" \
  "$MS_BASE/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"

# 视觉投影 mmproj（866 MB，开启 GPU 视觉必需）
curl -fL --retry 5 -C - \
  -o "$ASSETS_DIR/model/IQ3_S/mmproj-Qwen3.8-Flash-Next-BF16.gguf" \
  "$MS_BASE/mmproj-Qwen3.8-Flash-Next-BF16.gguf"
```

校验 SHA-256：

```bash
cd "$ASSETS_DIR/model/IQ3_S"
cat <<'EOF' | sha256sum -c -
4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0  mmproj-Qwen3.8-Flash-Next-BF16.gguf
EOF
```

### 4.2 生成专家 Pack（纯 CPU，不需要 GPU）

```bash
docker run --rm \
  -v "$REL_DIR:/app:ro" \
  -v "$ASSETS_DIR:/assets" \
  -w /app \
  --entrypoint python3 "$DTK_IMAGE" \
  tools/iq_pack.py \
  --gguf /assets/model/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --out /assets/pack/iq3s \
  --experts-bin
```

### 4.3 准备投机解码资产（DFlash2 + MTP + Draft 词表）

- **DFlash2 草稿模型出处**：[Hugging Face · incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)（发布包已附带量化打包好的 `bin/dflash2.q4.pack`，也可从该仓库自行转换）
- **MTP 草稿层出处**：[ModelScope · Qwen/Qwen3.8-Flash-Next](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/files)（通过 HTTP Range 仅下载 MTP 张量，无需下载完整 BF16 主模型）
- **Draft 词表出处**：
  - 通用词表：[Niko1221/Strata `draft_vocab.bin`](https://github.com/Niko1221/Strata/blob/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/data/draft_vocab.bin)
  - 中文 60K 词表：[nicejunjie/Strata-rdna3 `draft_vocab_zh60k.bin`](https://github.com/nicejunjie/Strata-rdna3/blob/7df6538d6c886093df39d74b36ed0a3396bf9467/rdna3/draft_vocab_zh60k.bin)（已内置于仓库 `data/`）

```bash
# 拉取并转换 MTP runtime
docker run --rm -e STRATA_SOURCE=modelscope \
  -v "$REL_DIR:/app:ro" -v "$ASSETS_DIR:/assets" -w /app \
  --entrypoint python3 "$DTK_IMAGE" \
  tools/mtp_fetch.py fetch --out /assets/mtp/source

docker run --rm \
  -v "$REL_DIR:/app:ro" -v "$ASSETS_DIR:/assets" -w /app \
  --entrypoint python3 "$DTK_IMAGE" \
  tools/mtp_fetch.py verify --out /assets/mtp/source

docker run --rm \
  -v "$REL_DIR:/app:ro" -v "$ASSETS_DIR:/assets" -w /app \
  --entrypoint python3 "$DTK_IMAGE" \
  tools/mtp_pack.py --src /assets/mtp/source --experts q2_0 --out /assets/mtp/mtp-q2_0.gguf

docker run --rm \
  -v "$REL_DIR:/app:ro" -v "$ASSETS_DIR:/assets" -w /app \
  --entrypoint python3 "$DTK_IMAGE" \
  tools/mtp_rt.py --gguf /assets/mtp/mtp-q2_0.gguf --out /assets/mtp/rt
```

---

## 5. 准备 DeepSeek-V4.1-Flash（双卡 GGUF 2.00bpw）资产

- **模型出处**：[Hugging Face · ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
- **包含内容**：IST-DAS `2.00bpw` 量化分片 + Engram 查表文件 + DSpark 投机草稿头（共约 257 GB，建议放在 NVMe 盘）
- **国内加速**：下载前设置 `export HF_ENDPOINT=https://hf-mirror.com`

```bash
export DS_DIR=/data/models/dsv41-2.00
mkdir -p "$DS_DIR"

# 使用上游 Strata 工具自动拉取并校验 IST-DAS 2.00bpw + DSpark（约 257 GB）
# 如需国内镜像加速，附加 -e HF_ENDPOINT=https://hf-mirror.com
docker run --rm -e HF_ENDPOINT=https://hf-mirror.com \
  -v "$REL_DIR:/app:ro" -v "$DS_DIR:/out" -w /app \
  --entrypoint bash "$DTK_IMAGE" \
  -lc 'python3 setup.py --family deepseek41 --model IST-DAS-2.00bpw --dir /out'
```

完成后 `/data/models/dsv41-2.00/maya-ds4-dspark` 内即包含可直接挂载启动的完整 pack 与 tokenizer。

---

## 6. 原生 Docker 启动命令

### 6.1 启动 Qwen3.8-Flash-Next（单卡默认全功能）

默认启用：**GPU 视觉 + 262,144 上下文（64K KV 驻留）+ 单卡缓存复用 + 共享会话目录 + `--prefill 8192`**（全 `24,576` 专家 100% 驻留 VRAM，整卡约 `61.1 / 65.5 GiB`）：

```bash
export HOTCACHE_DIR=/data/strata-hotcache/v040
mkdir -p "$REL_DIR/logs" "$HOTCACHE_DIR"

docker run -d --name strata-qwen-hcu0 \
  --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add video --ipc=host --shm-size 16g \
  -p 127.0.0.1:18740:18740 \
  -w /app \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$REL_DIR:/app:ro" \
  -v "$REL_DIR/logs:/logs" \
  -v "$ASSETS_DIR:/assets:ro" \
  -v "$HOTCACHE_DIR:/data/sessions" \
  --entrypoint /bin/bash \
  "$DTK_IMAGE" \
  -lc 'exec python3 -m serve.server --engine strata --config /app/configs/k100ai/qwen-iq3s-v040.json --host 0.0.0.0 --port 18740 --slot-save-path /data/sessions'
```

> 如需多卡共享缓存池，按同样命令再起 `renderD129`（端口 `18741`）并挂载同一个 `$HOTCACHE_DIR`，再启动宿主机 `session_manager.py` 与 `sticky_router.py`，完整步骤见 [运行与多卡缓存池指南](RUNTIME.md)。

### 6.2 启动 DeepSeek-V4.1-Flash（双卡交错 YOCO 最终版）

双卡 `renderD128 + renderD129`，默认启用无损交错 YOCO 流水线（32K Prefill `1,393.3 tok/s` / Decode `51.23 tok/s`，128/128 token 零差异）：

```bash
docker run -d --name strata-dsv41-tp2 \
  --device /dev/kfd \
  --device /dev/dri/renderD128 \
  --device /dev/dri/renderD129 \
  --group-add video --ipc=host --shm-size 32g \
  -p 127.0.0.1:18097:18097 \
  -w /app \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$REL_DIR:/app:ro" \
  -v "$DS_DIR:/models:ro" \
  -e DS4_PROF=0 \
  -e DS4_YOCO=1 \
  -e DS4_YOCO_INTERLEAVE=1 \
  -e DS4_ATTN_SLAB=1024 \
  -e DS4_ATTN_SPLIT=1 \
  -e DS4_DSPARK_CONF=0.2 \
  -e STRATA_DS4_ATTN_MMAC928_QSPLIT=0 \
  -e STRATA_DS4_ATTN_EPILOGUE_DIRECT=1 \
  -e DS4_DQ_OLD=1 \
  -e STRATA_DS4_IQ1M_PRECOMP_TABLE=1 \
  -e STRATA_DS4_QSA_MMAC928_LD=136 \
  -e STRATA_DS4_DSPARK_FALLBACK_VERIFY=1 \
  -e STRATA_DS4_SPARSE_ROPE_FUSE=1 \
  -e STRATA_DS4_Q5_ROPE_FUSE=1 \
  -e STRATA_DS4_ROUTER_F16=1 \
  --entrypoint /bin/bash \
  "$DTK_IMAGE" \
  -lc 'exec python3 -m serve.server --engine strata --config /app/configs/k100ai/gguf-tp2-v040.json --host 0.0.0.0 --port 18097 --no-api-key'
```

> **切至极速吞吐档**（32K Prefill `1,629.7 tok/s`，4K Prefill 最高 `831.8 tok/s`）：只需把上面命令中的 `-e DS4_YOCO=1` 改为 `-e DS4_YOCO=2`（短提示场景可同时把配置内 `--prefill-chunk` 从 `2048` 改为 `1024`）。

---

## 7. 启动验收

```bash
# 1. 查看加载日志
docker logs -f strata-qwen-hcu0

# 2. 健康检查（Qwen 返回 loaded=true, images=true, max_context=262144）
curl --fail http://127.0.0.1:18740/health

# 3. 发送文本请求
curl --fail http://127.0.0.1:18740/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"你好，请简短自我介绍"}],"max_tokens":64,"temperature":0}'
```
