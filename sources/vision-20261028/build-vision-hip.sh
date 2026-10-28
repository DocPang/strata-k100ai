#!/usr/bin/env bash
# Build strata-vision with the HIP backend (K100AI gfx928).
# Run inside the pinned DTK image with /opt/hyhal mounted.
set -euo pipefail

STRATA_SRC="${STRATA_SRC:?Set STRATA_SRC to the upstream strata 0.2.2 source tree}"
LLAMA_SRC="${LLAMA_SRC:?Set LLAMA_SRC to llama.cpp 3cf03257}"
DTK_PATH="${DTK_PATH:-/opt/dtk}"
HYHAL_PATH="${HYHAL_PATH:-/opt/hyhal}"
BUILD="${BUILD:-build-vision-hip}"
ARCH="${ARCH:-gfx928}"
JOBS="${JOBS:-24}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ -f "$LLAMA_SRC/ggml/src/ggml-hip/CMakeLists.txt" ]] || {
  echo "LLAMA_SRC does not look like llama.cpp: $LLAMA_SRC" >&2
  exit 1
}
[[ -f "$STRATA_SRC/tools/vision/CMakeLists.txt" ]] || {
  echo "STRATA_SRC does not contain tools/vision: $STRATA_SRC" >&2
  exit 1
}
[[ -d "$HYHAL_PATH/include" ]] || {
  echo "HyHAL headers not found under $HYHAL_PATH/include" >&2
  exit 1
}

for patch in "$HERE"/patches/000*.patch; do
  echo "applying $(basename "$patch")"
  patch -p1 -d "$LLAMA_SRC" < "$patch"
done

export PATH="$DTK_PATH/bin:$DTK_PATH/llvm/bin:$PATH"
export ROCM_PATH="$DTK_PATH"
export HIP_PATH="$DTK_PATH"

cmake -S "$STRATA_SRC/tools/vision" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_DIR="$LLAMA_SRC" \
  -DSTRATA_VISION_CUDA=OFF \
  -DSTRATA_PORTABLE=OFF \
  -DGGML_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES="$ARCH" \
  -DCMAKE_HIP_COMPILER="$DTK_PATH/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$DTK_PATH" \
  -DCMAKE_PREFIX_PATH="$DTK_PATH" \
  -DCMAKE_HIP_FLAGS="--rocm-path=$DTK_PATH" \
  -DCMAKE_EXE_LINKER_FLAGS=-no-pie

cmake --build "$BUILD" --target strata-vision -j "$JOBS"

OUT="$BUILD/bin/strata-vision"
[[ -x "$OUT" ]] || { echo "missing $OUT" >&2; exit 1; }
sha256sum "$OUT"
echo "built: $OUT"
