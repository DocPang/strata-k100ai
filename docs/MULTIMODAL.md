# 多模态视觉（v0.2.1）

v0.2.1 在文本引擎不变的前提下，为 K100AI 增加 GPU 图片编码器 `strata-vision`。图片先由 `strata-vision`（HIP + llama.cpp `mtmd`）编码成 embeddings，再以 `<|image_pad|>` token 进入 Qwen3.8-Flash-Next 文本模型。

## 组件

| 组件 | 路径 | 说明 |
| --- | --- | --- |
| 文本引擎 | `/opt/k100ai/bin/strata` | 与 v0.2.0 相同，SHA-256 `bdbfd934…` |
| 图片编码器 | `/opt/k100ai/bin/strata-vision` | 新增，HIP/GPU，SHA-256 `9c4dc143…`（119,439,216 B） |
| 视觉投影 | `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 放在 `MODEL_DIR`，866 MiB，SHA-256 `b1a82259…` |

编码器是独立常驻进程。服务启动时加载，约占 **2.1 GiB 显存**；单卡 64 GiB 下文本引擎加载后仍余约 **5.27 GiB**。

## 前置

- K100AI（gfx928）+ 已验证的 DTK 25.10 基础镜像；
- v0.2.1 运行包（已包含 `runtime/bin/strata-vision`）；
- 模型目录中有 `mmproj-Qwen3.8-Flash-Next-BF16.gguf`；
- 文本侧安装、权重、pack、MTP 已按 [INSTALL.md](INSTALL.md) 准备好。

下载 mmproj（与两个 IQ3_XXS 分片同一模型仓库）：

```bash
curl -fL --retry 5 -C - \
  -o "$DATA_DIR/models/IQ3_XXS/mmproj-Qwen3.8-Flash-Next-BF16.gguf" \
  'https://modelscope.cn/api/v1/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/repo?Revision=master&FilePath=mmproj-Qwen3.8-Flash-Next-BF16.gguf'

printf '%s  %s\n' \
  b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0 \
  "$DATA_DIR/models/IQ3_XXS/mmproj-Qwen3.8-Flash-Next-BF16.gguf" | sha256sum -c -
```

## 启用

1. 安装 v0.2.1：

```bash
./install.sh
```

2. 在 `deployment.env` 中打开：

```bash
VISION_ENABLED=1
VISION_MAX_TOKENS=1024
```

如需自定义路径：

```bash
# 默认 /opt/k100ai/bin/strata-vision
VISION_EXE=/opt/k100ai/bin/strata-vision
# 默认 mmproj-Qwen3.8-Flash-Next-BF16.gguf（相对 /models）
VISION_MMPROJ=mmproj-Qwen3.8-Flash-Next-BF16.gguf
```

3. 启动：

```bash
./start.sh
docker logs -f "k100ai-$INSTANCE" | grep -E 'vision|images'
```

`/health` 返回 `"images": true` 表示编码器已加载。启动失败时，entrypoint 会明确报缺少 `strata-vision` 或 mmproj。

## 调用

端口与文本服务一致（默认 `127.0.0.1:8080/v1`），使用同一个 `STRATA_API_KEY`。

```bash
IMG_B64="$(base64 -w0 /path/to/image.png)"

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $STRATA_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"qwen3.8-flash-next-strata\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"读出图片里的标题和表格内容\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$IMG_B64\"}}
      ]
    }],
    \"max_tokens\": 1000
  }"
```

Python 客户端：

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="<STRATA_API_KEY>")
b64 = base64.b64encode(open("image.png", "rb").read()).decode()

r = client.chat.completions.create(
    model="qwen3.8-flash-next-strata",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "读出图片里的标题和表格内容"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ],
    }],
    max_tokens=1000,
)
print(r.choices[0].message.content)
```

也支持公网 `https://` 图片 URL。本地文件请用 base64 data URL；服务端不直接读取客户端本地路径。

## 性能与验证

单卡 gfx928，760×520 图（475 image tokens）：

| 项目 | 结果 |
| --- | --- |
| GPU 编码 | **1.72 s** |
| CPU 编码（8 线程，对照） | 18.0 s |
| 同图二次请求（编码缓存） | 约 1.8 s 端到端 |
| 1800×1200 图（1024 tokens） | 12.5 s 端到端（含 309 tokens 生成） |
| GPU vs CPU embeddings | cosine mean 0.999293 / min 0.947855 / rel-L2 3.24% |
| 服务启动后剩余显存 | 约 5,274 MiB |

长上下文压测（40K / 100K / 230K token）期间显存峰值不变，KV 不额外占用显存。

## 限制与回滚

- 图片上限按 GPU 默认 **1024 tokens**；更大值未验证，耗时和显存随 tokens 增长。
- `VISION_MAX_TOKENS` 允许 1–4096，但建议保持 1024。
- 编码器常驻约 2.1 GiB 显存；与文本引擎共享同一张卡。
- 关闭视觉：`VISION_ENABLED=0` 后重启容器；文本路径不受影响。
- 编码器源码、补丁和构建步骤见 [sources/vision-20261028](../sources/vision-20261028/)。
