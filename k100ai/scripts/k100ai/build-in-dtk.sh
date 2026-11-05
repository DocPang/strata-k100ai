#!/usr/bin/env bash
set -euo pipefail

root=${1:-/data/strata-dsv41-k100ai}
llama=${STRATA_GGML_DIR:-/data/experiments/strata_replay_c4_20261015/llama}
build=${BUILD_DIR:-"$root/build-gfx928-qwen"}
jobs=${JOBS:-64}
dsv41=${DSV41:-0}

if [[ ! -x /opt/dtk/llvm/bin/clang++ ]]; then
  echo "DTK clang++ is unavailable" >&2
  exit 1
fi

# The DTK image exposes HSA headers in .hyhal, while hip-lang expects the
# conventional include path. This symlink exists only in the disposable
# container.
if [[ -d /opt/dtk/.hyhal/include/hsa ]]; then
  ln -sfn /opt/dtk/.hyhal/include/hsa /opt/dtk/include/hsa
fi

export ROCM_PATH=/opt/dtk
export HIP_PATH=/opt/dtk
export HSA_HEADER=/opt/dtk/.hyhal/include

cmake -S "$root" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP_GFX906=ON \
  -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_PREFILL_MMQ=ON \
  -DSTRATA_MMQ_KQUANTS=ON \
  -DSTRATA_Q6K_EXPERTS=ON \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_BUILD_DSV41="$dsv41" \
  -DCMAKE_HIP_ARCHITECTURES=gfx928 \
  -DSTRATA_GGML_DIR="$llama" \
  -DCMAKE_PREFIX_PATH=/opt/dtk/.hyhal \
  -DCMAKE_C_COMPILER=/opt/dtk/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ \
  '-DCMAKE_HIP_FLAGS=--gpu-max-threads-per-block=1024 -DSTRATA_GFX928_DOT4=1 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false'

cmake --build "$build" --target "${TARGET:-strata}" -j "$jobs"
