#!/usr/bin/env bash
set -euo pipefail

# Rebuild the current Qwen 0.1.55 and DeepSeek 0.1.55 engines from a source
# tree and the pinned llama.cpp source. The build uses fixed container paths
# so the compiler's embedded source paths do not depend on the checkout name.

src=$(cd -- "${1:?SOURCE is required}" && pwd -P)
llama_src=$(cd -- "${2:?LLAMA_SOURCE is required}" && pwd -P)
out=${3:-"$PWD/build-current-k100ai"}
image=${BASE_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68}
jobs=${JOBS:-64}
qwen_expected=${QWEN_CURRENT_SHA256-437308792b7ec08724aa90e4b11736b4f40f41c10613faa9688fbdc1723ba7ab}
dsv41_expected=${DSV41_SHA256-6543311f453bee2c6d61653a24f0b6282f9cb3568b17a8b889c2c5e073b12286}
test_expected=${DS4_TEST_SHA256-0d6c8455d76815b810a38c5529bc868087e71641008e0c86843ecd8b34dd815c}

mkdir -p "$out"
docker run --rm --name "strata-build-current-$$" \
  --shm-size=8g \
  -v "$src:/work/strata:ro" \
  -v "$llama_src:/work/llama:ro" \
  -v "$out:/work/out" \
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
    common=(
      -G Ninja
      -DCMAKE_BUILD_TYPE=Release
      -DSTRATA_HIP_GFX906=ON
      -DSTRATA_ENABLE_HIP=OFF
      -DSTRATA_ENABLE_CUDA=ON
      -DSTRATA_PREFILL_MMQ=ON
      -DSTRATA_MMQ_KQUANTS=ON
      -DSTRATA_Q6K_EXPERTS=ON
      -DSTRATA_NATIVE_EXPERTS=ON
      -DSTRATA_BUILD_TESTS=OFF
      -DCMAKE_HIP_ARCHITECTURES=gfx928
      -DSTRATA_GGML_DIR=/work/llama
      -DCMAKE_PREFIX_PATH=/opt/dtk/.hyhal
      -DCMAKE_C_COMPILER=/opt/dtk/llvm/bin/clang
      -DCMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/clang++
      -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++
      "-DCMAKE_HIP_FLAGS=--gpu-max-threads-per-block=1024 -DSTRATA_GFX928_DOT4=1 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false"
    )
    cmake -S /work/strata -B /work/out/qwen "${common[@]}"
    cmake --build /work/out/qwen --target strata --parallel "$JOBS"
    cmake -S /work/strata -B /work/out/dsv41 "${common[@]}" -DSTRATA_BUILD_DSV41=ON
    cmake --build /work/out/dsv41 --target strata-dsv41 ds4_gguf_test --parallel "$JOBS"
  '

check_hash() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ -z "$expected" ]]; then
    printf '%s  %s\n' "$actual" "$path"
    return 0
  fi
  if [[ "$actual" != "$expected" ]]; then
    echo "binary mismatch: $path expected $expected, got $actual" >&2
    return 1
  fi
  printf '%s  %s\n' "$actual" "$path"
}

check_hash "$out/qwen/strata" "$qwen_expected"
check_hash "$out/dsv41/strata-dsv41" "$dsv41_expected"
check_hash "$out/dsv41/ds4_gguf_test" "$test_expected"
printf 'CURRENT_REBUILD_OK\n'
