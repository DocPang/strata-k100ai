#!/usr/bin/env bash
set -euo pipefail

# Recreate the Qwen IQ3_XXS model-pack and MTP runtime used by this release.
# Model weights stay outside the release archive.

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=${K100AI_ROOT:-$(cd -- "$script_dir/../.." && pwd -P)}
dest=${1:-"$root/assets/qwen3.8-flash-next-iq3-xxs"}
image=${BASE_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68}

repo=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
revision=1868010e27505d4b8cc15970cdcc4dd76005d7de
base="https://www.modelscope.cn/models/$repo/resolve/$revision"
model_dir="$dest/model/IQ3_XXS"
pack_dir="$dest/pack/iq3xxs"
mtp_source="$dest/mtp-source"
mtp_gguf="$dest/mtp-q2_0.gguf"
mtp_rt="$dest/mtp/rt"

shard1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
shard2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
declare -A sizes=(
  ["$shard1"]=47039860096
  ["$shard2"]=28800138432
)
declare -A hashes=(
  ["$shard1"]=219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15
  ["$shard2"]=316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113
)

mkdir -p "$model_dir" "$pack_dir" "$mtp_source" "$mtp_rt"

for file in "$shard1" "$shard2"; do
  target="$model_dir/$file"
  expected=${sizes[$file]}
  actual=$(stat -c %s "$target" 2>/dev/null || printf 0)
  if (( actual > expected )); then
    echo "$file is larger than its pinned size ($actual > $expected); remove it and retry" >&2
    exit 2
  fi
  if (( actual < expected )); then
    echo "downloading $file"
    curl -fL --retry 8 --retry-delay 5 -C - -o "$target" "$base/$file"
  fi
done

cat >"$model_dir/SHA256SUMS" <<EOF
${hashes[$shard1]}  $shard1
${hashes[$shard2]}  $shard2
EOF
(cd "$model_dir" && sha256sum -c SHA256SUMS)

run_python=(
  docker run --rm
  --name "strata-qwen-assets-$$"
  -v "$root:$root"
  -v "$dest:$dest"
  -w "$root"
  -e "PYTHONPATH=$root/.python-packages:$root"
  -e PYTHONWARNINGS=ignore
  -e STRATA_SOURCE=modelscope
  --entrypoint python3
  "$image"
)

if [[ ! -d "$root/.python-packages" ]]; then
  echo "run scripts/k100ai/setup-runtime.sh first" >&2
  exit 2
fi
if [[ ! -f "$root/third_party/llama.cpp/gguf-py/gguf/__init__.py" ]]; then
  echo "missing bundled gguf-py: $root/third_party/llama.cpp/gguf-py" >&2
  exit 2
fi

"${run_python[@]}" tools/iq_pack.py \
  --gguf "$model_dir/$shard1" \
  --out "$pack_dir" \
  --experts-bin

cat >"$pack_dir/SHA256SUMS" <<'EOF'
e76956703f0cb960fa959cc0c33b0e38284ee98c8eb360417c74e7238c7bf005  experts.bin
d2af3d8331e5c7a1985d1d589747230f9d56719a6d33c7cddc9ee6b81cde6b13  dense.bin
1afe12b36cd2d13bf122b741fa0e81e1f751319cc3c451e54ba75524d825f148  index.txt
ae28cf7a2a1c9063c4b55d6c8cf86d4d079dcddcf597f9e8697c9157d462c0ee  native_experts.txt
68beec784ea64aeecace591b33debb8ba331a9ad257a3e1b68c5c701dd22b267  conversions.json
30bf3328b49b4b3444dbe29bdae9b1fe65e86d33fda10fffbdb70bb9a5316644  experts.bin.src.json
EOF
(cd "$pack_dir" && sha256sum -c SHA256SUMS)

"${run_python[@]}" tools/mtp_fetch.py fetch --out "$mtp_source"
"${run_python[@]}" tools/mtp_fetch.py verify --out "$mtp_source"
"${run_python[@]}" tools/mtp_pack.py --src "$mtp_source" --experts q2_0 --out "$mtp_gguf"
"${run_python[@]}" tools/mtp_rt.py --gguf "$mtp_gguf" --out "$mtp_rt"

curl -fL --retry 8 --retry-delay 5 -o "$mtp_rt/draft_vocab.bin" \
  "https://raw.githubusercontent.com/Niko1221/Strata/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/data/draft_vocab.bin"

cat >"$mtp_rt/SHA256SUMS" <<'EOF'
09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f  experts.bin
c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a  dense.bin
991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd  dense.txt
b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91  draft_vocab.bin
EOF
(cd "$mtp_rt" && sha256sum -c SHA256SUMS)

cat <<EOF
QWEN_ASSETS_OK
model=$model_dir
pack=$pack_dir
mtp=$mtp_rt
EOF
