# strata-vision HIP port（gfx928）

日期：2026-10-28
适用版本：`v0.2.1`

本目录记录把 Strata 的图片编码器 `strata-vision` 从 CPU 迁到 K100AI GPU（HIP）所需的源码补丁、构建命令和验证数据。文本引擎 `runtime/bin/strata` 与 v0.2.0 完全相同；本版新增的是 `runtime/bin/strata-vision`。

## 背景

上游 strata 的 AMD 后端只提供 CPU 图片编码器：

```python
# setup.py
def hip_vision(asked):
    """... a HIP (GPU) encoder build is a later step, so `yes`/`gpu` leave images off ..."""
```

`strata-vision` 本身基于 llama.cpp 的 `mtmd/clip`，该库与后端无关。用 `GGML_HIP=ON` 重新构建 `strata-vision`，即可让 ViT 在 gfx928 上执行。文本模型与图片编码器是两个进程：图片先编码成 embeddings，再以 `<|image_pad|>` 进入文本模型。

## 补丁

以下补丁针对 llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`，全部从 llama.cpp 根目录用 `patch -p1` 应用：

1. `patches/0001-dtk-hip-version.patch`
   DTK 25.10 的 `hip-config-version.cmake` 报 `5.3.26113`，而 `hipconfig --version` 是 `6.3.26113dcc`。把 llama.cpp 的 `hip_VERSION >= 6.1` 硬失败改成警告。
2. `patches/0002-softmax-hip-cooperative-groups.patch`
   DTK 的 cooperative-groups 头文件与 clang 17 不兼容（`redefinition of __any`、`invalid input constraint 'l'`）。HIP 下编译掉 grid-wide softmax 路径；视觉编码器的 ncols 很小，不会走该路径。
3. `patches/0003-norm-hip-256-threads.patch`
   `norm_f32<1024>` / `group_norm_f32<1024>` 在 gfx928 上超过 launch bounds（256）。HIP 下改用 256 线程；不修会让 embeddings 静默算错（与 CPU 的 cosine 只有 0.05，但模型仍会给出看似合理的错误答案）。

## 构建

前置：DTK 25.10（`/opt/dtk`）、HyHAL（`/opt/hyhal`）、CMake ≥ 3.21、Ninja，以及上游 strata 0.2.2 源码中的 `tools/vision`。

建议在固定的 DTK 容器内构建，挂载源码和 `/opt/hyhal`：

```bash
STRATA_SRC=/work/strata-0.2.2 \
LLAMA_SRC=/work/llama.cpp-3cf03257 \
bash build-vision-hip.sh
```

脚本会：

- 校验 llama.cpp 源码目录；
- 依次应用三个补丁；
- 用 gfx928 + HIP 配置并编译；
- 输出并校验 `build-vision-hip/bin/strata-vision`。

链接必须带 `-no-pie`；否则 ggml-hip 的 non-PIE 目标文件会报：

```text
relocation R_X86_64_32 ... can not be used when making a PIE object
```

## 运行配置

容器内把编码器放在 `/opt/k100ai/bin/strata-vision`，配置片段：

```json
{
  "vision": {
    "exe": "/opt/k100ai/bin/strata-vision",
    "mmproj": "/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
    "model": "/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
    "gpu": true,
    "max_tokens": 1024
  }
}
```

`v0.2.1` 的 entrypoint 通过 `VISION_ENABLED=1` 自动注入该段；`VISION_MAX_TOKENS` 默认 1024。编码器常驻进程会在服务启动时加载，约占 2.1 GiB 显存；每卡剩余显存实测约 5.27 GiB。

## 验证

- 单张 760×520 图（475 image tokens）：GPU 编码 **1.72 s**；CPU 同图 18.0 s（8 线程）。
- GPU vs CPU embeddings：cosine mean `0.999293`，min `0.947855`，rel-L2 `3.24%`。
- 1800×1200 大图：1024 image tokens，端到端 12.5 s（含 309 tokens 生成）。
- 长上下文不占额外显存：40K / 100K / 230K token 请求期间 VRAM 峰值不变（60,246 MiB，剩余 5,274 MiB）。
- 端到端链路：DSH 桌面 `describe_image` 10 s、内置 `read_image` 6 s，均读对标题、字段与柱状图数值。

## 限制

- 编码器是独立进程；没有图片请求时不占 GPU 计算，但常驻显存约 2.1 GiB。
- `max_tokens` 上限按 GPU 默认 1024；更大值未验证，耗时和显存随 tokens 增长。
- 该二进制针对 gfx928 + DTK 25.10 构建；换 DTK/架构需重新编译。
- 文本引擎未改动，v0.2.0 的文本性能数据仍然适用。
