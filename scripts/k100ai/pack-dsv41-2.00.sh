#!/usr/bin/env bash
set -euo pipefail

root=/data/strata-dsv41-k100ai
model=${MODEL_DIR:-/data/models/dsv41-2.00}
out=${PACK_DIR:-"$model/maya-ds4"}
profile=${PROFILE:-"$root/data/dsv41/profile-2.00.bin"}
drafter=${DSPARK_DIR:-"$model/dspark"}
drafter_gguf=${DSPARK_GGUF:-}

shards=(
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf
)
engram=DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf

for file in "${shards[@]}" "$engram"; do
  if [[ ! -s "$model/$file" ]]; then
    echo "missing model file: $model/$file" >&2
    exit 2
  fi
done
if [[ ! -s "$profile" ]]; then
  echo "missing expert profile: $profile" >&2
  exit 2
fi
if [[ -n "$drafter_gguf" ]]; then
  if [[ ! -s "$drafter_gguf" ]]; then
    echo "missing DSpark GGUF: $drafter_gguf" >&2
    exit 2
  fi
elif [[ ! -s "$drafter/dspark.json" || ! -s "$drafter/dspark.bin" ]]; then
  echo "missing DSpark drafter: set DSPARK_GGUF or provide $drafter/dspark.{json,bin}" >&2
  exit 2
fi

if [[ "${SKIP_HASH:-0}" != 1 ]]; then
  (
    cd "$model"
    sha256sum -c SHA256SUMS
  )
fi

drafter_args=()
if [[ -n "$drafter_gguf" ]]; then
  drafter_args=(
    --dspark-gguf "$drafter_gguf"
    --dspark-noise-token 128799
    --dspark-markov-rank 256
  )
else
  drafter_args=(--drafter "$drafter/dspark.json")
fi

python3 "$root/tools/dsv41/pack_gguf.py" \
  --model "$model" \
  --out "$out" \
  --shards "${shards[@]}" \
  --engram "$model/$engram" \
  "${drafter_args[@]}"

cp "$profile" "$out/expert-profile.bin"

printf 'pack: %s\n' "$out"
find "$out" -maxdepth 2 -type f -printf '%P %s\n' | sort
sha256sum "$out/ds4.txt" "$out/tensors.txt" "$out/engram.bin" "$out/expert-profile.bin"
