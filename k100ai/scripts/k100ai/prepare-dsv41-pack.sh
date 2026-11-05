#!/usr/bin/env bash
set -euo pipefail

# Turn the pinned six-shard DeepSeek downloads into the pack read by
# strata-dsv41. The model weights remain outside the release archive.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=${K100AI_ROOT:-$(cd -- "$script_dir/../.." && pwd -P)}
model_dir=${1:-"$root/assets/dsv41-2.00"}
pack_dir=${2:-"$model_dir/maya-ds4-dspark"}
image=${BASE_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68}

shards=(
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf
)
engram=DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
if [[ -f "$model_dir/dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf" ]]; then
  dspark=dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
elif [[ -f "$model_dir/dspark/dspark-unsloth-Q8_0.gguf" ]]; then
  # Earlier K100AI runs retained the downloaded file under this local alias.
  dspark=dspark/dspark-unsloth-Q8_0.gguf
else
  dspark=dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
fi

for file in "${shards[@]}" "$engram" "$dspark"; do
  test -f "$model_dir/$file" || {
    echo "missing DeepSeek asset: $model_dir/$file" >&2
    echo "run scripts/k100ai/fetch-dsv41-iq2xxs.sh $model_dir first" >&2
    exit 2
  }
done

if [[ -e "$pack_dir" ]]; then
  echo "pack already exists: $pack_dir" >&2
  exit 3
fi
if [[ ! -d "$root/.python-packages" ]]; then
  echo "run scripts/k100ai/setup-runtime.sh first" >&2
  exit 2
fi

docker run --rm --name "strata-dsv41-pack-$$" \
  -v "$root:$root" \
  -v "$model_dir:$model_dir" \
  -w "$root" \
  -e "PYTHONPATH=$root/.python-packages:$root/source:$root" \
  -e PYTHONWARNINGS=ignore \
  --entrypoint python3 \
  "$image" \
  source/tools/dsv41/pack_gguf.py \
    --model "$model_dir" \
    --shards "${shards[@]}" \
    --engram "$model_dir/$engram" \
    --dspark-gguf "$model_dir/$dspark" \
    --dspark-noise-token 128799 \
    --dspark-markov-rank 256 \
    --out "$pack_dir"

cat >"$pack_dir/SHA256SUMS" <<'EOF'
87e04d460f8f9ab82a14cb3793fe0c48f85348d81da5f2e93d200bc5dbc2d6f4  ds4.txt
f91733dc81ff262a0e6f65c0c5c3b86aa1390cfcd144bba75858357d62a50284  tensors.txt
08d55d6433f389b14aa115f4ea4eb31d6a2d63df1ae40821f03facf4c2d49142  engram.bin
4d61d22c90ea0cdf15c7e298f7215f23eeffd20f8eed8fb517fda5fb64693d99  tokenizer/chat_template.jinja
547bace837acea98652eef5669c9b01cee1c73ad1d2a44d6fdce9e6a81c22147  tokenizer/merges.txt
0a61021438e5ce6041133495f18fc360a6587aea748b17d953f552d2119ba51e  tokenizer/token_type.json
5a47981b685b75154e29bfb0299584c9b3a4a28679585e86d524c34c065d037f  tokenizer/tokenizer.json
2f0bf5b31f4aa191d5d0d78291927a4f28c9ceb498a5e345b97e926c4cb99c1a  tokenizer/vocab.json
EOF
(cd "$pack_dir" && sha256sum -c SHA256SUMS)

printf 'DSV41_PACK_OK\npack=%s\n' "$pack_dir"
