#!/usr/bin/env bash
set -Eeuo pipefail
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
work_dir=${1:-/work}
jobs=${K100AI_BUILD_JOBS:-8}
[[ -f "$work_dir/SOURCE_REPLAY_VERIFIED.json" && ! -e "$work_dir/build" ]]
[[ -x /opt/dtk/llvm/bin/clang++ ]]

# These changes are confined to the disposable container and prepared source tree.
if [[ ! -e /opt/dtk/include/hsa && -d /opt/dtk/.hyhal/include/hsa ]]; then
  ln -sfn /opt/dtk/.hyhal/include/hsa /opt/dtk/include/hsa
fi
mkdir -p "$work_dir/strata/cmake"
cp "$recipe_dir/cmake/k100ai-build.cmake" "$work_dir/strata/cmake/k100ai-build.cmake"
printf '\ninclude("${CMAKE_CURRENT_LIST_DIR}/cmake/k100ai-build.cmake")\n' >> "$work_dir/strata/CMakeLists.txt"

cmake -S "$work_dir/strata" -B "$work_dir/build" -G Ninja \
  -DSTRATA_HIP_GFX906=ON -DSTRATA_ENABLE_HIP=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx928 \
  -DSTRATA_GGML_DIR="$work_dir/llama" \
  -DCMAKE_PREFIX_PATH=/opt/dtk/.hyhal \
  -DCMAKE_C_COMPILER=/opt/dtk/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++ \
  '-DCMAKE_HIP_FLAGS=--gpu-max-threads-per-block=1024 -DSTRATA_GFX928_DOT4=1 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false' \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
if [[ "${K100AI_CONFIGURE_ONLY:-0}" == 1 ]]; then
  printf 'CONFIGURE_ONLY_COMPLETE\n'
  exit 0
fi
cmake --build "$work_dir/build" --target strata --parallel "$jobs"
sha256sum "$work_dir/build/strata" > "$work_dir/REBUILT_SHA256SUMS"
printf 'CLEAN_BUILD_COMPLETE\n'
