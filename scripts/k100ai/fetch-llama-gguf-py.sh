#!/usr/bin/env bash
set -euo pipefail

# Provide the pinned llama.cpp Python package used by the Qwen asset builders.
# The release package normally embeds this directory. Source checkouts and
# older archives can call this helper to restore it before downloading models.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=${1:-$(cd -- "$script_dir/../.." && pwd -P)}
llama_root="$root/third_party/llama.cpp"
gguf_py="$llama_root/gguf-py"
commit=3cf03257f219afbe7334045ff7c6a06ac68c627d
zip_sha256=cbe23c594282ead2937abb3f008e51fcec4609d9256652fff42c7cc1c21ea47b
zip_url="https://github.com/ggml-org/llama.cpp/archive/$commit.zip"

gguf_py_ready() {
  [[ -f "$gguf_py/gguf/__init__.py" ]] &&
    [[ -f "$gguf_py/gguf/constants.py" ]] &&
    grep -Eq '^[[:space:]]*Q2_0[[:space:]]*=[[:space:]]*42([[:space:]]|$)' \
      "$gguf_py/gguf/constants.py"
}

if gguf_py_ready; then
  echo "gguf-py ready: $gguf_py"
  exit 0
fi

for command in curl unzip sha256sum; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "missing command required to fetch gguf-py: $command" >&2
    exit 2
  }
done

mkdir -p "$root/third_party"
tmp=$(mktemp -d "$root/third_party/.llama-gguf-py.XXXXXX")
cleanup() {
  rm -rf -- "$tmp"
}
trap cleanup EXIT

archive=${STRATA_LLAMA_CPP_ZIP:-"$tmp/llama.cpp-$commit.zip"}
expected_sha=${STRATA_LLAMA_CPP_ZIP_SHA256:-$zip_sha256}
if [[ -n "${STRATA_LLAMA_CPP_ZIP:-}" ]]; then
  [[ -f "$archive" ]] || {
    echo "missing llama.cpp archive: $archive" >&2
    exit 2
  }
  echo "using llama.cpp archive: $archive"
else
  echo "downloading pinned llama.cpp gguf-py source"
  curl -fL --retry 8 --retry-delay 5 -o "$archive.part" "$zip_url"
  mv "$archive.part" "$archive"
fi

actual_sha=$(sha256sum "$archive" | awk '{print $1}')
[[ "$actual_sha" == "$expected_sha" ]] || {
  echo "llama.cpp archive SHA-256 mismatch" >&2
  echo "expected: $expected_sha" >&2
  echo "actual:   $actual_sha" >&2
  exit 2
}

extract="$tmp/extract"
mkdir -p "$extract"
unzip -q "$archive" \
  "llama.cpp-$commit/gguf-py/*" \
  "llama.cpp-$commit/LICENSE" \
  -d "$extract"
source_root="$extract/llama.cpp-$commit"

[[ -f "$source_root/gguf-py/gguf/__init__.py" ]] || {
  echo "llama.cpp archive does not contain gguf-py" >&2
  exit 2
}
mkdir -p "$gguf_py"
cp -R "$source_root/gguf-py/." "$gguf_py/"
if [[ -f "$source_root/LICENSE" ]]; then
  cp "$source_root/LICENSE" "$llama_root/LICENSE"
fi

gguf_py_ready || {
  echo "restored gguf-py does not provide Q2_0=42: $gguf_py" >&2
  exit 2
}
echo "gguf-py ready: $gguf_py"
