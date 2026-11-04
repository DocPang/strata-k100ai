#!/usr/bin/env bash
set -euo pipefail

# Build both gfx928 engines and produce a runtime tree plus tar.gz. The
# package owner may be outside this checkout; nothing in the package needs the
# source or build directory at run time.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd -- "$script_dir/../.." && pwd -P)
src=${SRC:-$root}
out=${OUT:-"$root/dist-release-$(date +%Y%m%d-%H%M%S)"}
image=${BASE_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68}
ggml=${STRATA_GGML_DIR:-}
jobs=${JOBS:-64}
qwen_fast_bin=${QWEN_FAST_BIN:-}
qwen_fast_source_commit=${QWEN_FAST_SOURCE_COMMIT:-2150b1acdb48c386a9afcd64e63e2369b5a5e108}
engine_source_commit=${ENGINE_SOURCE_COMMIT:-$(git -C "$src" log -1 --format=%H -- CMakeLists.txt src 2>/dev/null || cat "$src/BUILD_ID" 2>/dev/null || printf 'unknown')}
package_source_commit=${PACKAGE_SOURCE_COMMIT:-$(git -C "$src" rev-parse HEAD 2>/dev/null || cat "$src/BUILD_ID" 2>/dev/null || printf 'unknown')}
qwen_current_expected=437308792b7ec08724aa90e4b11736b4f40f41c10613faa9688fbdc1723ba7ab
dsv41_expected=6543311f453bee2c6d61653a24f0b6282f9cb3568b17a8b889c2c5e073b12286
test_expected=0d6c8455d76815b810a38c5529bc868087e71641008e0c86843ecd8b34dd815c

test -n "$ggml" || { echo "set STRATA_GGML_DIR to the fixed llama.cpp source snapshot" >&2; exit 2; }
test -n "$qwen_fast_bin" || { echo "set QWEN_FAST_BIN to the verified 0.1.47 engine" >&2; exit 2; }
for path in "$src/CMakeLists.txt" "$ggml" "$src/scripts/k100ai/package-release.py" "$qwen_fast_bin"; do
  test -e "$path" || { echo "missing build input: $path" >&2; exit 2; }
done
test ! -e "$out" || { echo "output already exists: $out" >&2; exit 2; }
mkdir -p "$out"

log="$out/build.log"
set +e
docker run --rm --name "strata-k100ai-release-build-$$" \
  -v "$src:/work/strata:ro" \
  -v "$out:/work/out" \
  -v "$ggml:/work/llama:ro" \
  -v /opt/hyhal/include:/opt/rocm/include:ro \
  -e ROCM_PATH=/opt/dtk \
  -e HIP_PATH=/opt/dtk \
  -e HIP_PLATFORM=amd \
  -e BUILD_SRC=/work/strata \
  -e BUILD_OUT=/work/out \
  -e BUILD_GGML=/work/llama \
  -e BUILD_JOBS="$jobs" \
  -w /work/strata \
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
      "-DSTRATA_GGML_DIR=$BUILD_GGML"
      -DCMAKE_PREFIX_PATH=/opt/dtk/.hyhal
      -DCMAKE_C_COMPILER=/opt/dtk/llvm/bin/clang
      -DCMAKE_CXX_COMPILER=/opt/dtk/llvm/bin/clang++
      -DCMAKE_HIP_COMPILER=/opt/dtk/llvm/bin/clang++
      "-DCMAKE_HIP_FLAGS=--gpu-max-threads-per-block=1024 -DSTRATA_GFX928_DOT4=1 -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false"
    )
    cmake -S "$BUILD_SRC" -B "$BUILD_OUT/qwen" "${common[@]}"
    cmake --build "$BUILD_OUT/qwen" --target strata --parallel "$BUILD_JOBS"
    cmake -S "$BUILD_SRC" -B "$BUILD_OUT/dsv41" "${common[@]}" -DSTRATA_BUILD_DSV41=ON
    cmake --build "$BUILD_OUT/dsv41" --target strata-dsv41 ds4_gguf_test --parallel "$BUILD_JOBS"
  ' 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e
if [[ $rc -ne 0 ]]; then
  echo "engine build failed; log: $log" >&2
  exit "$rc"
fi

actual=$(sha256sum "$out/qwen/strata" | awk '{print $1}')
[[ "$actual" == "$qwen_current_expected" ]] || {
  echo "Qwen current binary mismatch: expected $qwen_current_expected, got $actual" >&2
  exit 1
}
actual=$(sha256sum "$out/dsv41/strata-dsv41" | awk '{print $1}')
[[ "$actual" == "$dsv41_expected" ]] || {
  echo "DeepSeek binary mismatch: expected $dsv41_expected, got $actual" >&2
  exit 1
}
actual=$(sha256sum "$out/dsv41/ds4_gguf_test" | awk '{print $1}')
[[ "$actual" == "$test_expected" ]] || {
  echo "ds4_gguf_test binary mismatch: expected $test_expected, got $actual" >&2
  exit 1
}

python3 "$src/scripts/k100ai/package-release.py" \
  --source "$src" \
  --qwen-bin "$qwen_fast_bin" \
  --qwen-current-bin "$out/qwen/strata" \
  --qwen-fast-source-commit "$qwen_fast_source_commit" \
  --dsv41-bin "$out/dsv41/strata-dsv41" \
  --ds4-test-bin "$out/dsv41/ds4_gguf_test" \
  --ggml-dir "$ggml" \
  --base-image "$image" \
  --output "$out/strata-v0.2.9-k100ai-dual" \
  --engine-source-commit "$engine_source_commit" \
  --package-source-commit "$package_source_commit" \
  --engine-version 0.1.55
