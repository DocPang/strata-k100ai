#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=${K100AI_ROOT:-$(cd -- "$script_dir/../.." && pwd -P)}
dest=${1:-"$root/assets/dsv41-2.00"}
repo=ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF
dspark_repo=unsloth/DeepSeek-V4.1-Flash-1017-GGUF

mkdir -p "$dest"

# Current repository layout, updated 2026-10-24. It uses five backbone shards;
# the earlier four-shard names are stale and resolve to HTTP 404.
files=(
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf
  DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
)
sizes=(
  5780704
  39535781184
  39736712000
  39036908384
  19923784832
  110595290816
)
revisions=(
  40d2af57944f5a4a5d7cb17f921f30db8c45ecf3
  40d2af57944f5a4a5d7cb17f921f30db8c45ecf3
  08d153046911aec47936988a9897a8b9598adffd
  08d153046911aec47936988a9897a8b9598adffd
  20577e0adcacb9ce9c225debe076fbd4b974fffe
  20577e0adcacb9ce9c225debe076fbd4b974fffe
)

for i in "${!files[@]}"; do
  file=${files[i]}
  expected=${sizes[i]}
  revision=${revisions[i]}
  actual=$(stat -c%s "$dest/$file" 2>/dev/null || echo 0)
  if [[ "$actual" == "$expected" ]]; then
    echo "[skip] $file $(date --iso-8601=seconds) $actual"
    continue
  fi
  if (( actual > expected )); then
    stale="$dest/$file.oversize-$(date +%Y%m%dT%H%M%S)"
    mv "$dest/$file" "$stale"
    echo "[quarantine] $file $actual -> $stale"
  fi
  echo "[start] $file $(date --iso-8601=seconds)"
  curl -fL --retry 8 --retry-delay 5 -C - \
    -o "$dest/$file" \
    "https://www.modelscope.cn/models/$repo/resolve/$revision/$file"
  actual=$(stat -c%s "$dest/$file")
  if [[ "$actual" != "$expected" ]]; then
    echo "[size mismatch] $file expected=$expected actual=$actual" >&2
    exit 1
  fi
  echo "[done] $file $(date --iso-8601=seconds) $actual"
done

dspark=dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
dspark_revision=6cd2eacaad3f5d9b61e6913c188a934a3a5eac22
mkdir -p "$dest/dspark"
if [[ "$(stat -c%s "$dest/dspark/$dspark" 2>/dev/null || echo 0)" != 7976914240 ]]; then
  curl -fL --retry 8 --retry-delay 5 -C - \
    -o "$dest/dspark/$dspark" \
    "https://www.modelscope.cn/models/$dspark_repo/resolve/$dspark_revision/$dspark"
fi
test "$(stat -c%s "$dest/dspark/$dspark")" = 7976914240

cat >"$dest/SHA256SUMS" <<'EOF'
f356f198561f56df36f7ab92d7238a926646d76de992c34f15bcb939a5f529b9  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf
25af3785ccf925f54137e1ead4c481a8952315dc0b1af6519a99407f67bce8f3  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf
8168d0998fc494def2e1952d83c131425a108660aae9d29190f5a8670ab62b48  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf
9eb9b4de1968f150dfc207eb31c27be9cd9b5666a9a6246bde8a22b184b4bee0  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf
7e35f6b4e6fc46f6c4b152b593d84d683c5ece695ae6ad734f556198c041bec1  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf
3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3  DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a  dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
EOF

(cd "$dest" && sha256sum -c SHA256SUMS)

ln -sfn \
  DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf \
  "$dest/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00006-of-00006.gguf"
