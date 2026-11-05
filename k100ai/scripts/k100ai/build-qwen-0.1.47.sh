#!/usr/bin/env bash
set -euo pipefail

# Rebuild the default fast Qwen engine from the preserved source line.
# Usage: build-qwen-0.1.47.sh FAST_SOURCE LLAMA_SOURCE [BUILD_DIR]

fast_src=$(cd -- "${1:?FAST_SOURCE is required}" && pwd -P)
llama_src=$(cd -- "${2:?LLAMA_SOURCE is required}" && pwd -P)
build_dir=${3:-"$PWD/build-qwen-0.1.47"}
image=${BASE_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68}
jobs=${JOBS:-64}
expected=bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9

mkdir -p "$build_dir"
docker run --rm --name "strata-build-qwen0147-$$" \
  --shm-size=8g \
  -v "$fast_src:/work/strata:ro" \
  -v "$llama_src:/work/llama:ro" \
  -v "$build_dir:/work/build" \
  -v /opt/hyhal:/opt/hyhal:ro \
  -e ROCM_PATH=/opt/dtk \
  -e HIP_PATH=/opt/dtk \
  -e HIP_PLATFORM=amd \
  -e "JOBS=$jobs" \
  -e "LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib:" \
  -w /work \
  --entrypoint /bin/bash \
  "$image" -lc '
    set -euo pipefail
    cmake -G Ninja -S /work/strata -B /work/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=/opt/dtk/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/clang++ \
      -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ \
      -DCMAKE_HIP_ARCHITECTURES=gfx928 \
      -DCMAKE_HIP_FLAGS="--gpu-max-threads-per-block=1024 -DSTRATA_GFX928_DOT4=1 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false" \
      -DSTRATA_GGML_DIR=/work/llama \
      -DSTRATA_ENABLE_CUDA=ON \
      -DSTRATA_ENABLE_HIP=OFF \
      -DSTRATA_HIP_GFX906=ON \
      -DSTRATA_PREFILL_MMQ=ON \
      -DSTRATA_MMQ_KQUANTS=ON \
      -DSTRATA_Q6K_EXPERTS=ON \
      -DSTRATA_NATIVE_EXPERTS=ON \
      -DSTRATA_MMQ_TQ_OVERLAY=ON \
      -DSTRATA_BUILD_TESTS=OFF \
      -DSTRATA_BUILD_CONVERSATION_TESTS=OFF \
      -DGGML_BUILD_TESTS=OFF \
      -DGGML_BUILD_EXAMPLES=OFF \
      -DSTRATA_PLE_FIXTURE_DIR=/work/strata/bench/micro
    cmake --build /work/build --target strata --parallel "$JOBS"
  '

actual=$(sha256sum "$build_dir/strata" | awk '{print $1}')
if [[ "$actual" != "$expected" ]]; then
  echo "binary mismatch: expected $expected, got $actual" >&2
  exit 1
fi
printf 'QWEN_0.1.47_REBUILD_OK\nsha256=%s\npath=%s\n' "$actual" "$build_dir/strata"
