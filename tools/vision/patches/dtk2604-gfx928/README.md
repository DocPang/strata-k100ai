# DTK 26.04 / gfx928 最小适配补丁（视觉 HIP 构建）

针对 `setup.py` 固定的 llama.cpp 提交 **`3cf03257f219afbe7334045ff7c6a06ac68c627d`**
（`LLAMA_CPP_COMMIT`）的最小补丁集：让 `strata-vision-hip`（`STRATA_VISION_HIP=ON`）
能在海光 K100AI（gfx928）+ DTK 26.04（galaxyhip 5.2 运行时）上编译与运行。
讨论背景见 [issue #3](https://github.com/DocPang/strata-k100ai/issues/3)，
使用说明见 [docs/k100ai/VISION.md](../../../docs/k100ai/VISION.md)。

## 应用方式

```sh
cd <llama.cpp 检出根目录>   # 固定 commit 3cf0325
git apply tools/vision/patches/dtk2604-gfx928/0001-dtk2604-gfx928-adaptations.patch
```

## 四处适配（3 个文件）

| # | 文件 | 改动 | 依据 |
|---|---|---|---|
| 1 | `ggml/src/ggml-hip/CMakeLists.txt` | 版本门 `VERSION_LESS 6.1` → `5.0` | 逐点核查：该 pin 上全部 6.x API 引用均带 `HIP_VERSION` 守卫与回退分支，版本门是保守基线而非真实依赖 |
| 2 | `ggml/src/ggml-cuda/vendors/hip.h` | `cudaStreamPerThread` 兜底（DTK 5.2 头文件无此符号，退回默认流 `(hipStream_t)0`）；`__gfx928__` → `HYGON_GFX928` 宏；`GGML_HIP_NO_COOP` + cooperative_groups 最小桩 | DTK 5.2 的 `hip_cooperative_groups.h` 在 gfx928 上有 asm 约束 bug；桩可编译且运行时不被调用（`supports_cooperative_launch=false` 走标准路径） |
| 3 | `ggml/src/ggml-cuda/softmax.cu` | HIP 路径的 `hip_cooperative_groups` include 用 `GGML_HIP_NO_COOP` 守卫 | 同上，只有此文件引入该头 |

## 构建（与 VISION.md §3 一致，含 HSA 隐藏路径处理）

```sh
# 容器内先补软链（hsa.h 藏在隐藏目录 /opt/dtk/.hyhal，且 -D 传不进 try_compile 子工程）
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
ninja -C build-vision-hip strata-vision-hip    # 产物 build-vision-hip/bin/strata-vision-hip
```

运行期四个必要条件（不满足时静默失败）见
[VISION.md §4](../../../docs/k100ai/VISION.md)。

## 验证记录（2026-11-18，干净检出）

```text
基线:    llama.cpp 3cf0325 官方 zip 解包（未做任何其他改动）
补丁:    git apply 0001-dtk2604-gfx928-adaptations.patch（一次干净应用）
构建:    DTK sglang 容器（digest 878d317d…），clang / gfx928，Ninja
结果:    275/275 目标全部通过，0 errors（warnings 为模板实例化的既有噪声）
产物:    build-vision-hip/bin/strata-vision-hip
         SHA-256 a7ddfc6a0ea17c21e8c0ce37a6699daee130a66f91937002c565267f38b411be
运行参考: 生产环境同配方构建的 strata-vision-hip：图像编码 6323 ms (CPU 16线程)
         → 762 ms (gfx928)，8.3×（见 issue #3 / PR #4 记录）
```

## 版本边界说明

社区在**其它** llama.cpp 版本上还遇到两处编译问题（`topk-moe.cu` 的 `__ballot_sync`
缺失、`HIP_VERSION >= 60300000` 守卫因 hipconfig 报 6.3 / 运行时库 5.2 而走错路径，
见 issue #3 评论）——经核对，**在本 pin（3cf0325）上均不存在**：该文件的
`__ballot_sync` 调用不在 topk-moe.cu，全树仅 `fattn.cu:48` 一处，且在 DTK 26.04 上
原样编译通过（如上 275/275）。若将来升级 `LLAMA_CPP_COMMIT`，需要按 issue #3 评论的
两处修法重新评估。
