# K100AI 视觉输入（strata-vision）

[返回首页](../../README.md) · [安装与启动](INSTALL_V0.3.md) · [来源记录](PROVENANCE_V0.3.md)

v0.3.0 发布包未包含视觉编码器二进制。本文记录在 DTK 26.04（gfx928）上从干净检出启用
图片输入的完整步骤与实测结果：CPU 编码器（零源码改动）与 HIP 编码器（GPU 加速
8.3×）两条路径均已验证。

## 结论速览

| 路径 | 图像编码耗时（640×480 → 300 token） | 前提 |
| --- | ---: | --- |
| CPU（16 线程，Hygon C86-4G） | 6323 ms | 零源码改动，portable 构建可在任意 DTK 容器运行 |
| HIP（gfx928 GPU） | **762 ms（8.3×）** | 需先做 ggml-hip 的 DTK 5.2 最小适配（见 [issue #3](https://github.com/DocPang/strata-k100ai/issues/3)） |

端到端（图片输入 + 64 token 生成）：CPU 8.6 s，GPU 3.1 s。

## 1. 准备源码与 llama.cpp 固定版本

本仓库不含 `third_party/llama.cpp`：视觉编码器按 `setup.py` 里的
`LLAMA_CPP_COMMIT`（当前 `3cf03257f219afbe7334045ff7c6a06ac68c627d`）引用外部
llama.cpp 检出。以 `k100ai/` 为源码根，从干净检出开始：

```sh
git clone https://github.com/DocPang/strata-k100ai
cd strata-k100ai/k100ai
mkdir -p third_party
curl -LO https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.zip
unzip -q 3cf03257f219afbe7334045ff7c6a06ac68c627d.zip -d third_party
mv third_party/llama.cpp-3cf0325* third_party/llama.cpp
```

（运行 `setup.py` 走完整安装时会自动完成同样的下载与解包；上面的手动步骤是为只构建
视觉编码器的读者准备的。）

**只有 HIP 路径**需要先按 [issue #3](https://github.com/DocPang/strata-k100ai/issues/3)
的配方给 llama.cpp 打 DTK 5.2 适配，共 **6 处编译期改动**——原配方 4 处（版本门
6.1→5.0、`hipStreamPerThread` 兜底、cooperative_groups 桩、softmax.cu 守卫）加社区
实测补充的 2 处（见
[#3 的评论](https://github.com/DocPang/strata-k100ai/issues/3)）：

- `ggml/src/ggml-cuda/topk-moe.cu` 的 `__ballot_sync` shim——DTK 的 galaxyhip 没有
  CUDA 9+ 的该内建，调用点 warp 本就收敛，`#define __ballot_sync(mask, pred) __ballot(pred)` 即可；
- DTK 版本守卫覆盖——`hipconfig` 报 6.3 而运行时库是 5.2（cmake 读到 5.2），导致
  `HIP_VERSION >= 60300000` 类守卫走进只有 6.3 才提供的路径而编译失败；定义
  `GGML_DTK_GALAXYHIP` 后将这些守卫强制按旧版处理。

原配方四个锚点在上面的固定 commit 上原样成立——`ggml/src/ggml-hip/CMakeLists.txt:54`、
`ggml/src/ggml-cuda/vendors/hip.h:122`、`ggml/src/ggml-cuda/softmax.cu:6`——
打补丁前请照配方做锚点唯一校验。CPU 路径无需任何补丁。

## 2. CPU 编码器（无源码改动）

```sh
cmake -S tools/vision -B build-vision -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR=third_party/llama.cpp \
  -DSTRATA_PORTABLE=ON
ninja -C build-vision strata-vision
```

产物在 **`build-vision/bin/strata-vision`**（约 7.6 MB，AVX2 基线自包含）。
独立冒烟输出 `READY 2560` 后跟 `OK 300 20 15 6323`，末字段 6323 为编码总耗时（ms）。

## 3. HIP 编码器（GPU，gfx928）

```sh
# hsa.h 藏在隐藏目录 /opt/dtk/.hyhal/include/hsa，且 -D 变量传不进 try_compile
# 子工程，容器内先补一条软链（一次即可）：
ln -sfn /opt/dtk/.hyhal/include/hsa /opt/dtk/hip/include/hsa

cmake -S tools/vision -B build-vision-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR=third_party/llama.cpp \
  -DSTRATA_VISION_HIP=ON \
  -DSTRATA_VISION_HIP_ARCHS=gfx928 \
  -DGGML_HIP_NO_VMM=ON \
  -DHSA_HEADER=/opt/dtk/.hyhal/include \
  -DCMAKE_PREFIX_PATH=/opt/dtk/.hyhal \
  -DCMAKE_C_COMPILER=/opt/dtk/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/clang++ \
  -DCMAKE_HIP_FLAGS="--gpu-max-threads-per-block=1024"
ninja -C build-vision-hip strata-vision-hip
```

产物在 **`build-vision-hip/bin/strata-vision-hip`**。

两个注意点：

- hipcc 不显式指定 `--offload-arch` 时默认报 gfx938（≠ 实际设备 gfx928），所以
  `STRATA_VISION_HIP_ARCHS=gfx928` 必须显式给出；`tools/vision/gpu_arch.h` 会
  读取 ggml-hip 启动时的设备行并校验架构在编译列表内，不在则拒绝启动、服务端
  回退 CPU 读取器。
- `--gpu-max-threads-per-block=1024` 是 DTK clang 对 launch bounds 的要求。

## 4. mmproj 资产与 serve 配置

mmproj 与 pack 同源，取自
[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
的 `mmproj-Qwen3.8-Flash-Next-BF16.gguf`（866 MB）。

**CPU-only 配置**（无 `gpu` 字段：服务端不传 `--gpu`，直接用 CPU 读取器；
CPU 读取器的默认 max tokens 为 300）：

```json
{
  "vision": {
    "exe": "<绝对路径>/build-vision/bin/strata-vision",
    "mmproj": "<mmproj gguf 路径>",
    "model": "<主模型 gguf 路径>"
  }
}
```

**HIP 配置**（`"gpu": true` 时服务端才给读取器传 `--gpu` 走 GPU；`cpu_exe` 指定
回退读取器——GPU 读取器因架构校验等原因拒启时，服务端自动落回它；GPU/CPU 读取器的
默认 max tokens 分别为 1024 / 300）：

```json
{
  "vision": {
    "exe": "<绝对路径>/build-vision-hip/bin/strata-vision-hip",
    "cpu_exe": "<绝对路径>/build-vision/bin/strata-vision",
    "gpu": true,
    "mmproj": "<mmproj gguf 路径>",
    "model": "<主模型 gguf 路径>"
  }
}
```

引擎以 `--vision` 启动；`/health` 返回 `"images": true` 即视觉通道就绪。
视觉编码器显存预算约 1 GiB（mmproj 权重 + 中间激活），全驻留部署时计入
显存账本。

**运行期四个必要条件**（不满足时全部表现为**静默失败**——能跑、只是变慢或回退
CPU，无告警；社区实测补充，见 [#3 评论](https://github.com/DocPang/strata-k100ai/issues/3)）：

| # | 条件 | 不满足时的表现 |
|---|---|---|
| 1 | 容器挂载 `-v /opt/hyhal:/opt/hyhal:ro` | `no ROCm-capable device is detected` |
| 2 | 视觉进程用干净的 `LD_LIBRARY_PATH`（`/opt/dtk/lib:/opt/dtk/hip/lib:/opt/hyhal/lib:/opt/hyhal/lib64`） | 直接继承引擎环境（含 `/opt/dtk/llvm/lib`）时 `open hsa lib:libhsa-runtime64.so error`，服务端回退 CPU |
| 3 | `LD_PRELOAD=/opt/dtk/llvm/lib/libomp.so` | `error while loading shared libraries: libomp.so` |
| 4 | serve 配置 `vision` 块写 `"gpu": true`（并给 `cpu_exe` 回退） | 只给 `exe` 时静默走 CPU，日志仅一行 `strata-vision: on the CPU, 16 threads` |

## 5. 端到端验证

- OpenAI 接口 `content` 数组携带 `image_url`（base64 / http / 本地路径均可）；
- 测试图含文字 "STRATA VISION TEST"，模型可逐字读出；
- 整请求耗时（编码 + 64 token 生成）：CPU 8.6 s，HIP 3.1 s。

## 6. 已知限制

- ggml-hip 的 DTK 5.2 适配尚未进入上游 llama.cpp / ggml（配方与归属讨论见
  [issue #3](https://github.com/DocPang/strata-k100ai/issues/3)），HIP 路径
  目前由使用者按配方自行构建；
- gfx928 不在 ggml 的厂商架构表中，按架构查表的位置发生静默回退时性能会显著
  下降且不报错，部署后建议对照本文的耗时数字核验。

验证环境：单张 K100AI（64 GiB HBM）、DTK 26.04、galaxyhip 5.2.26113.2065、
Hygon C86-4G 宿主机。
