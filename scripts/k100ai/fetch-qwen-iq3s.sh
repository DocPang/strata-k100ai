#!/usr/bin/env bash
set -euo pipefail

# Prepare the Qwen3.8-Flash-Next IQ3_S model, expert pack and MTP runtime for
# v0.3.1. Model weights remain outside the release archive.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=${K100AI_ROOT:-$(cd -- "$script_dir/../.." && pwd -P)}
dest=${1:-"$root/assets/qwen3.8-flash-next-iq3-s"}
image=${BASE_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68}

repo=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
revision=6ac9406afe74db316b11a3865a3382c7bedbcf34
base="https://www.modelscope.cn/models/$repo/resolve/$revision"
model_dir="$dest/model/IQ3_S"
pack_dir="$dest/pack/iq3s"
mtp_source="$dest/mtp-source"
mtp_gguf="$dest/mtp-q2_0.gguf"
mtp_rt="$dest/mtp/rt"
draft_vocab="$root/data/k100ai-qwen-iq3s-draft-vocab-zh60k.bin"

shard1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
shard2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
declare -A sizes=(
  ["$shard1"]=54817524224
  ["$shard2"]=28800138432
)
declare -A hashes=(
  ["$shard1"]=4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3
  ["$shard2"]=316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113
)

if [[ ! -d "$root/.python-packages" ]]; then
  echo "run scripts/k100ai/setup-runtime.sh first" >&2
  exit 2
fi
bash "$script_dir/fetch-llama-gguf-py.sh" "$root"

mkdir -p "$model_dir" "$pack_dir" "$mtp_source" "$mtp_rt"

for file in "$shard1" "$shard2"; do
  target="$model_dir/$file"
  expected=${sizes[$file]}
  actual=$(stat -c %s "$target" 2>/dev/null || printf 0)
  if (( actual > expected )); then
    echo "$file is larger than its pinned size ($actual > $expected)" >&2
    echo "remove the incomplete file and retry" >&2
    exit 2
  fi
  if (( actual < expected )); then
    echo "downloading $file"
    curl -fL --retry 8 --retry-delay 5 -C - -o "$target" "$base/IQ3_S/$file"
  fi
done

cat >"$model_dir/SHA256SUMS" <<EOF
${hashes[$shard1]}  $shard1
${hashes[$shard2]}  $shard2
EOF
(cd "$model_dir" && sha256sum -c SHA256SUMS)

if [[ ! -f "$draft_vocab" ]]; then
  echo "missing release draft vocabulary: $draft_vocab" >&2
  exit 2
fi
printf '%s  %s\n' \
  e93f8cd23075d98ccdb4bf10d10c69613a25ec449afc062d0c9dbd33dfb96122 \
  "$draft_vocab" | sha256sum -c -

run_python=(
  docker run --rm
  --name "strata-qwen-iq3s-assets-$$"
  -v "$root:$root"
  -v "$dest:$dest"
  -w "$root"
  -e "PYTHONPATH=$root/.python-packages:$root"
  -e PYTHONWARNINGS=ignore
  -e STRATA_SOURCE=modelscope
  --entrypoint python3
  "$image"
)

"${run_python[@]}" tools/iq_pack.py \
  --gguf "$model_dir/$shard1" \
  --out "$pack_dir" \
  --experts-bin

cat >"$pack_dir/SHA256SUMS" <<'EOF'
ed9d0a872b4d9fb992a05f3f5114107b0689767f0d95e2fb7e2ffe091000e67c  experts.bin
d2af3d8331e5c7a1985d1d589747230f9d56719a6d33c7cddc9ee6b81cde6b13  dense.bin
b2a014878c3b61d6a6f915e194055a5e68ede34f9465dd1cb13249d902f008ac  index.txt
d9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d  native_experts.txt
51df3cd6ffd0d38c2da8e2d3a95a604b4a96c20b06fc639a9bf477282a42b86a  conversions.json
5a18d0c1f89ac64c08998010c81536fcd6b67557778f8f2dfc36539ad0150cad  experts.bin.src.json
12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce  tokenizer/chat_template.jinja
5088c8c298fc06af8382ddb3b76c888703ac2634e82263b97eedcc2ad202738b  tokenizer/token_type.json
4ba698529766c7a2f5052083ea10cdf2485577f5237a332ee291054eb11e3332  tokenizer/merges.txt
4ba64f0332abcfb0b600b7df1537d1e836e68009d5fb7cdb77339738dc6365c4  tokenizer/vocab.json
c0bd435efbc6c3b28964b47999afe16c4ad268c7c7c4e8cb3af410796a6f30bf  tokenizer/tokenizer.json
EOF
(cd "$pack_dir" && sha256sum -c SHA256SUMS)

"${run_python[@]}" tools/mtp_fetch.py fetch --out "$mtp_source"
"${run_python[@]}" tools/mtp_fetch.py verify --out "$mtp_source"
"${run_python[@]}" tools/mtp_pack.py --src "$mtp_source" --experts q2_0 --out "$mtp_gguf"
"${run_python[@]}" tools/mtp_rt.py --gguf "$mtp_gguf" --out "$mtp_rt"

curl -fL --retry 8 --retry-delay 5 -o "$mtp_rt/draft_vocab.bin" \
  "https://raw.githubusercontent.com/Niko1221/Strata/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/data/draft_vocab.bin"
cp "$draft_vocab" "$mtp_rt/draft_vocab_zh60k.bin"

cat >"$mtp_rt/SHA256SUMS" <<'EOF'
09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f  experts.bin
c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a  dense.bin
991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd  dense.txt
b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91  draft_vocab.bin
e93f8cd23075d98ccdb4bf10d10c69613a25ec449afc062d0c9dbd33dfb96122  draft_vocab_zh60k.bin
EOF
(cd "$mtp_rt" && sha256sum -c SHA256SUMS)

cat <<EOF
QWEN_IQ3S_ASSETS_OK
model=$model_dir
pack=$pack_dir
mtp=$mtp_rt
draft_vocab=$draft_vocab
EOF
