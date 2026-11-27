# 运行、调优与多卡共享缓存池指南（v0.4.0）

[返回首页](../../README.md) · [安装与复现](INSTALL.md) · [v0.4.0 更新说明](release-notes/v0.4.0.md)

---

## 1. Qwen 核心参数速查与显存策略

在单张 64 GiB K100AI 上同时开启 **GPU 视觉 + 256K 上下文 + 缓存复用 + 多卡共享缓存池** 时，显存非常讲究平衡。下面是实机对照 3 套方案得出的结论：

| 方案 | 关键参数 | 常驻专家 | Graph 后余量 | 整卡占用 | Decode 速度 | 结论 |
| :--- | :--- | :---: | ---: | ---: | ---: | :--- |
| 大 Prefill 缓冲 | `--prefill 20480 --expert-cache 24576` | 24,576 (100%) | 185 MiB | 64,436 MiB | 78.3 tok/s | 显存过紧（98.5%），高分辨率视觉易 OOM |
| 卸载冷专家到内存 | `--prefill 20480 --expert-cache auto --vram-reserve-mib 7680` | 17,549 (71.4%) | 13,975 MiB | ~51,500 MiB | **35.5 tok/s** (-54.7%) | 一旦有专家不在 VRAM，引擎会退出零门铃图，解码直接腰斩 |
| **v0.4.0 默认方案** | **`--prefill 8192 --expert-cache 24576 --vram-reserve-mib 2048`** | **24,576 (100%)** | **3,589 MiB** | **61,085 MiB** | **78.3 tok/s（满速）** | **不卸载专家即释放 3.4 GiB 显存，1080p 视觉稳定零 OOM** |

### 常用参数调整对照

| 参数 | 默认值 | 调大/调小的影响 |
| :--- | :--- | :--- |
| `--max-context` | `262144` | 最大上下文窗口（256K），无需轻易缩减 |
| `--kv` / `--kv-resident` | `int8` / `65536` | 64K KV 常驻显存，超出部分从 pinned RAM 流式换入 |
| `--prefill` | `8192` | 显存紧张时优先调此项（`8192` 省下 3.4 GiB）；不开启视觉时可调回 `20480` |
| `--expert-cache` | `24576` | **强烈建议保持 `24576` 全驻留**，避免退出零门铃图导致 Decode 腰斩 |
| `--prompt-cache` / `--conversation-cache-mib` | `8` / `32768` | 控制主机内存中的提示词与多轮会话缓存容量 |
| `--vision` | 开启 | 配合配置内 `vision.gpu: true` 启用 `strata-vision-hip` GPU 视觉编码 |

---

## 2. 多卡共享缓存池架构与原生启动

多张 K100AI 卡各自跑一个独立容器（互不抢显存），宿主机通过每卡一个 `session_manager.py` 和统一入口 `sticky_router.py` 共享同一个 **tmpfs 热缓存目录 + SSD 冷归档目录**，实现跨卡秒级会话恢复：

```text
Client (OpenAI / Anthropic API)
   │
   ▼  :18768
sticky_router.py  (按会话粘性优先路由回原卡，原卡忙时自动漂移)
   ├──► session_manager (HCU0, :18760) ──► strata-qwen-hcu0 (:18740)
   └──► session_manager (HCU1, :18761) ──► strata-qwen-hcu1 (:18741)
                 │                                  │
                 └──── 共享 tmpfs 热缓存 + SSD 归档 ──┘
```

### 原生启动步骤（双卡示例：`HCU0` + `HCU1`）

```bash
export REL_DIR=/data/strata-k100ai-v0.4.0
export HOTCACHE_DIR=/data/strata-hotcache/v040     # 建议挂载在 tmpfs / 内存盘
export SESSION_DIR=/data/strata-sessions/v040      # SSD 冷归档目录
mkdir -p "$REL_DIR/logs" "$HOTCACHE_DIR" "$SESSION_DIR"

# 1. 先按 INSTALL.md 启动两个引擎容器：
#    strata-qwen-hcu0 (renderD128 -> 127.0.0.1:18740)
#    strata-qwen-hcu1 (renderD129 -> 127.0.0.1:18741)
#    注意：两个容器都挂载同一个 "$HOTCACHE_DIR:/data/sessions"

# 2. 启动两张卡各自的 session_manager
python3 "$REL_DIR/session-manager/session_manager.py" \
  --host 127.0.0.1 --port 18760 \
  --upstream http://127.0.0.1:18740 \
  --session-dir "$SESSION_DIR" --hotcache-dir "$HOTCACHE_DIR" \
  --max-hotcache-gb 50 --max-disk-gb 500 \
  > "$REL_DIR/logs/mgr-hcu0.log" 2>&1 &

python3 "$REL_DIR/session-manager/session_manager.py" \
  --host 127.0.0.1 --port 18761 \
  --upstream http://127.0.0.1:18741 \
  --session-dir "$SESSION_DIR" --hotcache-dir "$HOTCACHE_DIR" \
  --max-hotcache-gb 50 --max-disk-gb 500 \
  > "$REL_DIR/logs/mgr-hcu1.log" 2>&1 &

# 3. 启动统一粘性路由入口（对外暴露 18768）
python3 "$REL_DIR/sticky-router/sticky_router.py" \
  --host 0.0.0.0 --port 18768 \
  --backends hcu0=http://127.0.0.1:18760,hcu1=http://127.0.0.1:18761 \
  > "$REL_DIR/logs/sticky.log" 2>&1 &
```

实测效果：`HCU0` 落盘 `257 MB` 会话快照后，请求漂移到 `HCU1` 仅需 **`0.99 s`** 即可跨卡恢复 `2,241 / 2,267` tokens（解码 `83.2 tok/s`）；命中粘性路由仅 **`0.42 s`**（`87.0 tok/s`）。

---

## 3. API 调用与多模态视觉示例

所有服务同时支持 OpenAI（`/v1/chat/completions`）与 Anthropic（`/v1/messages`）协议。

### 3.1 多模态视觉请求（Base64 图片）

```bash
IMG_B64="$(base64 -w0 /path/to/test.png)"
curl --fail http://127.0.0.1:18768/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"qwen3.8-flash-next\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"用一句话描述这张图片的内容\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$IMG_B64\"}}
      ]
    }],
    \"max_tokens\": 128,
    \"temperature\": 0
  }"
```

### 3.2 Anthropic 协议（Claude Code 等客户端）

将 `ANTHROPIC_BASE_URL` 指向 `http://127.0.0.1:18768`（或单卡端口 `18740`），模型名填 `qwen3.8-flash-next` 或 `deepseek-v4.1-flash` 即可直接接入。

---

## 4. 常见问题排障

- **容器内报 `libhydmi.so: cannot open shared object file`**：检查 `docker run` 是否挂载了 `-v /opt/hyhal:/opt/hyhal:ro`。
- **`/health` 里 `"images": false`**：检查 `mmproj-Qwen3.8-Flash-Next-BF16.gguf` 路径是否与 JSON 配置中 `vision.mmproj` 一致，且 `bin/strata-vision-hip` 具备可执行权限。
- **DeepSeek 双卡 Decode 掉到 ~25 tok/s**：检查是否把 `--reserve-mib` 设得过大（建议保持 `600`），避免把常驻专家挤出显存。
