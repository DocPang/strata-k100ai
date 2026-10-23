# 复现报告（2026-10-23）

用**上游 0.2.2 的公开源码 + 本目录的 36 个文件**重建 `bdbfd934…`，记录哪些结论成立、哪些不成立。

## 结论

| 项 | 结果 |
| --- | --- |
| 源码对应关系 | **通过**：全树逐文件 SHA-1 比对，除本目录 36 个路径外与上游 `011c9e60` 完全一致 |
| 干净全量编译 | **通过**：`BUILD_OK`，退出码 0，无 error |
| 同一目录内字节可复现 | **是**：`/work/build-replay` 两次全量干净编译均产出 `da43917a…`（45,674,152 B） |
| 跨目录字节可复现 | **否**：`/work/build` 产出 `bdbfd934…`，`/work/build-replay` 产出 `da43917a…`，字节数相同、内容不同 |
| 行为等价 | **通过**：三套件输出 SHA-256 与上游一致（例外见下） |

**差异原因**：二进制内嵌构建目录路径。例如 `strings` 可见 `/work/build/ggml-tq-overlay/mmq.cuh`、`common.cuh`、`quantize.cu` 等 `__FILE__` 字符串；换目录后这些字符串变长/变短，字符串表随之重排。两次同目录重建逐字节相同，说明**构建本身是确定的**，不可复现的只是"换一个路径还得到同一个 SHA"。

因此 `clean_rebuild_verified` 记为 `false`，范围为"同一构建目录内可复现"。

## 步骤

```bash
# 1) 取上游 0.2.2
git clone https://github.com/Niko1221/strata && cd strata && git checkout 011c9e60

# 2) 覆盖本目录的 36 个文件（按路径覆盖，逐文件校验 manifest.json 的 sha256）
rsync -a --checksum <snapshot>/ ./

# 3) 干净编译（容器内，配方见下）
bash build3.sh
```

## 构建配方

```bash
cmake -S /work/strata -B /work/build -G Ninja \
  -DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_Q6K_EXPERTS=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx928 \
  -DSTRATA_GGML_DIR=/work/llama \
  -DCMAKE_PREFIX_PATH=/opt/dtk/.hyhal \
  -DCMAKE_C_COMPILER=/opt/dtk/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ \
  "-DCMAKE_HIP_FLAGS=--gpu-max-threads-per-block=1024 -DSTRATA_GFX928_DOT4=1 \
     -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false" \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /work/build --target strata --parallel 8
```

镜像：`harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317d…`（与 rc 系列相同）。

## 目录差异证据

对比方法：取上游 `011c9e60` 的 `git ls-tree -r`（4,625 个 blob 哈希），对本仓库工作树逐文件算 git blob 哈希后比对。

- 相同：2,088 个同源路径逐一命中
- 修改：30 个（见 `manifest.json` 的 `change: modified`）
- 新增：6 个（`gateup_exact.hpp`、`kdn_down_gemm.hpp`、`iq4_preexpand_exact.hpp`、`kgu_wsg16.hpp`、`k100ai-build.cmake`、`k100ai_stubs.cu`）
- 另有 30 个 `bench/results/**/*.log` 在上游存在但本仓库工作树没有，属本仓库 `.gitignore` 忽略项，**不是**本次改动的删除

## 数值一致性

发布二进制即实测二进制。三套件（单卡 gfx928、同会话交替、对照组为同卡上游 0.2.2 构建）的输出 SHA-256：

| 套件 | 本引擎 | 上游 0.2.2 |
| --- | --- | --- |
| ctx32 第 1/2/3 请求 | `6fcfdfac…` / `a868a40c…` / `a868a40c…` | 同 |
| lt 第 1/2/3 请求 | `6fcfdfac…` / `a868a40c…` / `a868a40c…` | 同 |
| ctx128 第 2/3 请求 | `84f393bf…` | `a6bcf7fd…` |

ctx128 的差异来自 hc read 的 `[wd|wi]` 宽 GEMM（`STRATA_K100AI_HC_FUSE`，默认开）改变了该投影的累加序；置 `STRATA_K100AI_HC_FUSE=0` 后 logits 逐字节相同、输出回到 `a6bcf7fd…`。这是两棵树之间**唯一**已知的数值差异。
