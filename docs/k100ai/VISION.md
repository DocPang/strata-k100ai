# K100AI 视觉输入（strata-vision）

[返回首页](../../README.md) · [安装与启动](INSTALL_V0.3.md) · [来源记录](PROVENANCE_V0.3.md)

v0.3.0 发布包未包含视觉编码器二进制。本文记录在 DTK 26.04（gfx928）上从源码启用
图片输入的完整步骤与实测结果：CPU 编码器（零源码改动）与 HIP 编码器（GPU 加速
8.3×）两条路径均已验证。

## 结论速览

| 路径 | 图像编码耗时（640×480 → 300 token） | 前提 |
| --- | ---: | --- |
| CPU（16 线程，Hygon C86-4G） | 6323 ms | 零源码改动，portable 构建可在任意 DTK 容器运行 |
| HIP（gfx928 GPU） | **762 ms（8.3×）** | 需先做 ggml-hip 的 DTK 5.2 最小适配（见 [issue #3](https://github.com/DocPang/strata-k100ai/issues/3)） |

端到端（图片输入 + 64 token 生成）：CPU 8.6 s，GPU 3.1 s。

## 1. CPU 编码器

上游 `tools/vision` 的 portable 构建即可在 DTK 容器内运行：

```sh
cmake -S tools/vision -B build-vision -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR=third_party/llama.cpp \
  -DSTRATA_PORTABLE=ON
ninja -C build-vision strata-vision
```

产物 `build-vision/strata-vision` 约 7.6 MB，AVX2 基线自包含。独立冒烟输出
`READY 2560` 后跟 `OK 300 20 15 6323`，末字段 6323 为编码总耗时（ms）。

## 2. HIP 编码器（GPU）

上游 HIP 构建面向 ROCm 6.1+；DTK 26.04 的运行时库版本口径为 5.2，需先按
[issue #3](https://github.com/DocPang/strata-k100ai/issues/3) 的配方完成
ggml-hip 最小适配（版本门、`hipStreamPerThread` 兜底、cooperative_groups 桩、
禁用 VMM，共约 10 行）。构建：

```sh
cmake -S tools/vision -B build-vision-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR=third_party/llama.cpp \
  -DSTRATA_VISION_HIP=ON \
  -DSTRATA_VISION_HIP_ARCHS=gfx928 \
  -DGGML_HIP_NO_VMM=ON \
  -DCMAKE_C_COMPILER=/opt/dtk/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/clang++ \
  -DCMAKE_HIP_FLAGS="--gpu-max-threads-per-block=1024"
ninja -C build-vision-hip strata-vision-hip
```

两个注意点：

- hipcc 不显式指定 `--offload-arch` 时默认报 gfx938（≠ 实际设备 gfx928），所以
  `STRATA_VISION_HIP_ARCHS=gfx928` 必须显式给出；`tools/vision/gpu_arch.h` 会
  读取 ggml-hip 启动时的设备行并校验架构在编译列表内，不在则拒绝启动、服务端
  回退 CPU 读取器。
- `--gpu-max-threads-per-block=1024` 是 DTK clang 对 launch bounds 的要求。

## 3. mmproj 资产与 serve 配置

mmproj 与 pack 同源，取自
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
的 `mmproj-Qwen3.8-Flash-Next-BF16.gguf`（866 MB）。模型运行配置增加
`vision` 块：

```json
{
  "vision": {
    "exe": "<strata-vision 或 strata-vision-hip 路径>",
    "mmproj": "<mmproj gguf 路径>",
    "model": "<主模型 gguf 路径>"
  }
}
```

引擎以 `--vision` 启动；`/health` 返回 `"images": true` 即视觉通道就绪。
视觉编码器显存预算约 1 GiB（mmproj 权重 + 中间激活），全驻留部署时计入
显存账本。

## 4. 端到端验证

- OpenAI 接口 `content` 数组携带 `image_url`（base64 / http / 本地路径均可）；
- 测试图含文字 "STRATA VISION TEST"，模型可逐字读出；
- 整请求耗时（编码 + 64 token 生成）：CPU 8.6 s，HIP 3.1 s。

## 5. 已知限制

- ggml-hip 的 DTK 5.2 适配尚未进入上游 llama.cpp / ggml（配方与归属讨论见
  [issue #3](https://github.com/DocPang/strata-k100ai/issues/3)），HIP 路径
  目前由使用者按配方自行构建；
- gfx928 不在 ggml 的厂商架构表中，按架构查表的位置发生静默回退时性能会显著
  下降且不报错，部署后建议对照本文的耗时数字核验。

验证环境：单张 K100AI（64 GiB HBM）、DTK 26.04、galaxyhip 5.2.26113.2065、
Hygon C86-4G 宿主机。
