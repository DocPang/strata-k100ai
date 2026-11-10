# K100AI 手动安装与运行

[返回首页](../../README.md) · [历史运行说明](README.md) · [源码复现](../../REPRODUCIBILITY.md)

适用于 **v0.2.3**：Linux x86_64、K100AI/gfx928、DTK 2604。Qwen 需要一张 64 GiB 卡，DeepSeek 需要四张。模型下载、转换和 MTP 构造需要数百 GB 空间。


本文列出模型下载、文件校验、pack 生成和 Docker 启动的完整步骤。需要减少重复输入时，可使用文末的便利脚本。

### 1. 下载并校验运行包

从 [Releases](https://github.com/DocPang/strata-k100ai/releases) 下载：

- `strata-v0.2.9-k100ai-dual.tar.gz`
- `strata-v0.2.9-k100ai-dual.tar.gz.sha256`
- 需要自行重建时，同时下载 `strata-v0.2.9-k100ai-dual-source.tar.gz`

```bash
VERSION=v0.2.3-k100ai
BASE_URL="https://github.com/DocPang/strata-k100ai/releases/download/$VERSION"

curl -fL --retry 3 -O \
  "$BASE_URL/strata-v0.2.9-k100ai-dual.tar.gz"
curl -fL --retry 3 -O \
  "$BASE_URL/strata-v0.2.9-k100ai-dual.tar.gz.sha256"

sha256sum -c strata-v0.2.9-k100ai-dual.tar.gz.sha256
mkdir -p /data/strata-k100ai-v0.2.3
tar -xzf strata-v0.2.9-k100ai-dual.tar.gz \
  -C /data/strata-k100ai-v0.2.3 --strip-components=1
cd /data/strata-k100ai-v0.2.3
sha256sum -c SHA256SUMS
```

### 2. 安装 Python 运行时

```bash
test ! -e .python-packages

docker run --rm --name strata-k100ai-python-setup \
  -v "$PWD:$PWD" \
  -w "$PWD" \
  -e "K100AI_ROOT=$PWD" \
  -e "K100AI_PACKAGES=$PWD/.python-packages" \
  --entrypoint /bin/bash \
  harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68 \
  -lc '
    set -euo pipefail
    python3 -m pip install --disable-pip-version-check --no-cache-dir \
      --target "$K100AI_PACKAGES" -r "$K100AI_ROOT/requirements.txt"
    PYTHONPATH="$K100AI_PACKAGES" python3 -c \
      "import jinja2, numpy, PIL, psutil, regex, requests, yaml"
    sha256sum "$K100AI_ROOT/requirements.txt" > "$K100AI_PACKAGES/.k100ai-ready"
  '
```

这条命令在固定 DTK 容器内把 `requirements.txt` 安装到发布目录的
`.python-packages`，并人工导入所有顶层依赖。重新部署时应再次执行并用下列命令检查：

```bash
docker run --rm --name strata-k100ai-python-check \
  -v "$PWD:$PWD" \
  -w "$PWD" \
  -e "PYTHONPATH=$PWD/.python-packages" \
  --entrypoint python3 \
  harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68 \
  -c "import jinja2, numpy, PIL, psutil, regex, requests, yaml"
```

发布包中的 `setup-runtime.sh` 是同一组步骤的便利封装。

安装完成后设置本页反复使用的变量：

```bash
ROOT=/data/strata-k100ai-v0.2.3
ASSETS=/data/k100ai-assets
IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
RUN_DIR="$ROOT/run"

mkdir -p \
  "$ASSETS/qwen3.8-flash-next-iq3-xxs/model/IQ3_XXS" \
  "$ASSETS/qwen3.8-flash-next-iq3-xxs/pack/iq3xxs" \
  "$ASSETS/qwen3.8-flash-next-iq3-xxs/mtp-source" \
  "$ASSETS/qwen3.8-flash-next-iq3-xxs/mtp/rt" \
  "$ASSETS/dsv41-2.00/dspark" \
  "$RUN_DIR"
```

### 3. 手动下载并校验 Qwen 主权重

ModelScope 仓库：`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`<br>
固定 snapshot：`1868010e27505d4b8cc15970cdcc4dd76005d7de`

| 文件 | ModelScope URL | 大小 | SHA-256 |
| --- | --- | ---: | --- |
| `IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | `https://www.modelscope.cn/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/1868010e27505d4b8cc15970cdcc4dd76005d7de/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | 47,039,860,096 B | `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15` |
| `IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` | `https://www.modelscope.cn/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/1868010e27505d4b8cc15970cdcc4dd76005d7de/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` | 28,800,138,432 B | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

```bash
QWEN_REPO=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
QWEN_REV=1868010e27505d4b8cc15970cdcc4dd76005d7de
QWEN_BASE="https://www.modelscope.cn/models/$QWEN_REPO/resolve/$QWEN_REV"
QWEN_MODEL="$ASSETS/qwen3.8-flash-next-iq3-xxs/model/IQ3_XXS"
QWEN_PACK="$ASSETS/qwen3.8-flash-next-iq3-xxs/pack/iq3xxs"
QWEN_SHARD1=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
QWEN_SHARD2=Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf

curl -fL --retry 8 --retry-delay 5 -C - \
  -o "$QWEN_MODEL/$QWEN_SHARD1" \
  "$QWEN_BASE/IQ3_XXS/$QWEN_SHARD1"

curl -fL --retry 8 --retry-delay 5 -C - \
  -o "$QWEN_MODEL/$QWEN_SHARD2" \
  "$QWEN_BASE/IQ3_XXS/$QWEN_SHARD2"

cat >"$QWEN_MODEL/SHA256SUMS" <<'EOF'
219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113  Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf
EOF

(cd "$QWEN_MODEL" && sha256sum -c SHA256SUMS)
```

### 4. 人工生成 Qwen pack

转换容器不需要 GPU，也不联网。输入是刚校验的第一个 shard，输出必须位于空目录：

```bash
test ! -e "$QWEN_PACK/experts.bin"

docker run --rm --name strata-qwen-pack \
  -v "$ROOT:$ROOT" \
  -v "$ASSETS:$ASSETS" \
  -w "$ROOT" \
  -e "PYTHONPATH=$ROOT/.python-packages:$ROOT" \
  -e PYTHONWARNINGS=ignore \
  --entrypoint python3 \
  "$IMAGE" \
  tools/iq_pack.py \
    --gguf "$QWEN_MODEL/$QWEN_SHARD1" \
    --out "$QWEN_PACK" \
    --experts-bin
```

人工核对生成结果：

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `experts.bin` | 42,912,972,800 B | `e76956703f0cb960fa959cc0c33b0e38284ee98c8eb360417c74e7238c7bf005` |
| `dense.bin` | 1,538,035,200 B | `d2af3d8331e5c7a1985d1d589747230f9d56719a6d33c7cddc9ee6b81cde6b13` |
| `index.txt` | 93,687 B | `1afe12b36cd2d13bf122b741fa0e81e1f751319cc3c451e54ba75524d825f148` |
| `native_experts.txt` | 3,296 B | `ae28cf7a2a1c9063c4b55d6c8cf86d4d079dcddcf597f9e8697c9157d462c0ee` |
| `conversions.json` | 310 B | `68beec784ea64aeecace591b33debb8ba331a9ad257a3e1b68c5c701dd22b267` |
| `experts.bin.src.json` | — | `30bf3328b49b4b3444dbe29bdae9b1fe65e86d33fda10fffbdb70bb9a5316644` |

```bash
cat >"$QWEN_PACK/SHA256SUMS" <<'EOF'
e76956703f0cb960fa959cc0c33b0e38284ee98c8eb360417c74e7238c7bf005  experts.bin
d2af3d8331e5c7a1985d1d589747230f9d56719a6d33c7cddc9ee6b81cde6b13  dense.bin
1afe12b36cd2d13bf122b741fa0e81e1f751319cc3c451e54ba75524d825f148  index.txt
ae28cf7a2a1c9063c4b55d6c8cf86d4d079dcddcf597f9e8697c9157d462c0ee  native_experts.txt
68beec784ea64aeecace591b33debb8ba331a9ad257a3e1b68c5c701dd22b267  conversions.json
30bf3328b49b4b3444dbe29bdae9b1fe65e86d33fda10fffbdb70bb9a5316644  experts.bin.src.json
EOF

(cd "$QWEN_PACK" && sha256sum -c SHA256SUMS)
```

### 5. 人工准备 Qwen MTP

MTP 不在 GSQ-RCO GGUF 中，而是来自 BF16 checkpoint。这里的脚本只负责读取 safetensors header 并发出带
`Range` 的 HTTP 请求；它不会下载完整的 360 GB checkpoint，并且会验证每个响应必须是 HTTP 206、长度正确、SHA-256 正确。

固定的 BF16 checkpoint revision 是：

```text
de4b8e4d43b917e7706784d8bb445c9af86a3540
```

ModelScope 地址前缀是：

```text
https://www.modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master/
```

以下 31 个 tensor 是 MTP runtime 的完整输入。`source shard` 是该 tensor 所在的实际 safetensors 文件名，完整下载地址为
“前缀 + source shard”。

| Tensor | Source shard | Tensor SHA-256 |
| --- | --- | --- |
| `mtp.fc_embedding.weight` | `model-00120-of-00131.safetensors` | `241d6e87ded317a61b1aeb3ca4e28c53f1caba09ad5be66092e91fe5e3e14751` |
| `mtp.fc_hidden.weight` | `model-00124-of-00131.safetensors` | `97c8e427ad0177e2c713ed24a397d815b523eda15ef453f1ae6463f6c70a8f96` |
| `mtp.hyper_connection_mixer.hc_norm.weight` | `model-00037-of-00131.safetensors` | `ce948a398e15aeaa403c424cda34e6eed296ab2570d591bb5ff3da642b255d64` |
| `mtp.hyper_connection_mixer.input_mix_weight_down.weight` | `model-00041-of-00131.safetensors` | `c633d2841fda463e82045e0b93fb8cb8d511ec0c9eda616020f01aca18d71947` |
| `mtp.hyper_connection_mixer.input_mix_weight_up.weight` | `model-00043-of-00131.safetensors` | `6565285f0ba2923703252763cfc11d8a9bda40ab42e7548e12464b04dbdbec13` |
| `mtp.layers.0.attn_hyper_connection.block_inject_weight.weight` | `model-00047-of-00131.safetensors` | `c8afff64e7f54412d02120097c4255033c55245e1c1eed84cc630d428ccb864c` |
| `mtp.layers.0.attn_hyper_connection.hc_norm.weight` | `model-00049-of-00131.safetensors` | `1fb619aeaaf637ccd70f46f1f8f8116e6a46448cb43026b246de8ee8286ed809` |
| `mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight` | `model-00053-of-00131.safetensors` | `e80ac8bf3e270e9d5a949ec7c913329dc8de017d69851334820fae7b8c86a7f5` |
| `mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight` | `model-00057-of-00131.safetensors` | `f80f1fd44e1183e7764ea496b11ff89ead9dd758f861d4e79a770bf54956b1a5` |
| `mtp.layers.0.mlp.experts.down_proj` | `model-00062-of-00131.safetensors` | `539c46234f73bac70450f731f40e715f77ac32faf77def0f3f800e58aaabbbc0` |
| `mtp.layers.0.mlp.experts.gate_up_proj` | `model-00060-of-00131.safetensors` | `839e98f9642ebc884b178abd2090b6dc8b0bcd81ed92d7f7a08314c6a68f96d5` |
| `mtp.layers.0.mlp.gate.weight` | `model-00066-of-00131.safetensors` | `f0d35fa41b360a08bc0c5b1e5766aa42df960291716ee189208b5d2599462c20` |
| `mtp.layers.0.mlp.shared_expert.down_proj.weight` | `model-00076-of-00131.safetensors` | `d2ced8c81367cc494599d1a8cdc279c3dd6579b628ded05acd7495e99cee01d8` |
| `mtp.layers.0.mlp.shared_expert.gate_proj.weight` | `model-00072-of-00131.safetensors` | `f4a81109165853b3b64f36d809f98094b599763364070f5fa079200e1c386d4c` |
| `mtp.layers.0.mlp.shared_expert.up_proj.weight` | `model-00072-of-00131.safetensors` | `172586861eea53efbff10afc43ded5b6e49c83879e9efc01e089c652bbb0a156` |
| `mtp.layers.0.mlp.shared_expert_gate.weight` | `model-00070-of-00131.safetensors` | `acdd306d9996d356e4667a16ff63af6fef086fc7f570f1bc00bbe882a2a93f6e` |
| `mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight` | `model-00078-of-00131.safetensors` | `c5782dd92b479a1f8ef43d683fe9c917fd9e8fba81f23c1aae8817bfe4b476a6` |
| `mtp.layers.0.mlp_hyper_connection.hc_norm.weight` | `model-00082-of-00131.safetensors` | `21a504876b5e7b788301557c68dfb757999430eba36c66ec8ab96a63604dbfd1` |
| `mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight` | `model-00086-of-00131.safetensors` | `b71240036dea954e034d5ae04e097bcd7ac51cb28a44e4c93d198af3447278a1` |
| `mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight` | `model-00088-of-00131.safetensors` | `40756866bf2a5e82c28e407f717b6fa07614d593f78e79c7ceb2cd4ffe6dda94` |
| `mtp.layers.0.self_attn.indexer.index_qk_proj.weight` | `model-00090-of-00131.safetensors` | `2befafae6e944c1fc651fd73d4d527eb6f138a6c1679065b003128ee3150e859` |
| `mtp.layers.0.self_attn.indexer.k_layernorm.weight` | `model-00094-of-00131.safetensors` | `2db621f6a08b4085756fc69d60a4a1bd1d7fbf5d406456d2477075842efaae54` |
| `mtp.layers.0.self_attn.indexer.q_layernorm.weight` | `model-00100-of-00131.safetensors` | `afaf59f68e251939970ffdeacad0667dffc657a522bea62cdc44a52816fc49d6` |
| `mtp.layers.0.self_attn.k_norm.weight` | `model-00102-of-00131.safetensors` | `8f924b0357cf936e14fc34518205d000e1423aa60626827df81b0804fae8cdfd` |
| `mtp.layers.0.self_attn.k_proj.weight` | `model-00108-of-00131.safetensors` | `1a295dc2d6e633e808aeb19e00c97856a1817bbfcc83f5b610e2bfcacfcb8c38` |
| `mtp.layers.0.self_attn.o_proj.weight` | `model-00106-of-00131.safetensors` | `9d2fcf0ac6ac19de87f638f710cfd1af6ad14141e75b7d3d5d29fbd06f374219` |
| `mtp.layers.0.self_attn.q_norm.weight` | `model-00112-of-00131.safetensors` | `57957aa1e7c5a3ac628bff0ec5522ad55a39b8c0b8bd2d324ae640e3771617ef` |
| `mtp.layers.0.self_attn.q_proj.weight` | `model-00108-of-00131.safetensors` | `5220aef3d288f40759f7356af9ef453e5ac0256a38749a9bb803a4e9db58150d` |
| `mtp.layers.0.self_attn.v_proj.weight` | `model-00108-of-00131.safetensors` | `39cb12f5424e965fa71fb81db0ee230c701b7b1e0b72c36641fde7e0e41bff41` |
| `mtp.pre_fc_norm_embedding.weight` | `model-00114-of-00131.safetensors` | `04c4a570850e06f2d8913da8220d54d4c7f87db6eb6d45480b938e8ba41d6a86` |
| `mtp.pre_fc_norm_hidden.weight` | `model-00118-of-00131.safetensors` | `b933b44559879449b1baf2af8385c5c237347307e3fd452394870cb53e6eab6d` |

人工检查 header、抓取 tensor、逐 tensor 校验：

```bash
MTP_SRC="$ASSETS/qwen3.8-flash-next-iq3-xxs/mtp-source"
MTP_GGUF="$ASSETS/qwen3.8-flash-next-iq3-xxs/mtp-q2_0.gguf"
MTP_RT="$ASSETS/qwen3.8-flash-next-iq3-xxs/mtp/rt"

docker run --rm --name strata-mtp-inventory \
  -v "$ROOT:$ROOT" -v "$ASSETS:$ASSETS" -w "$ROOT" \
  -e STRATA_SOURCE=modelscope \
  -e "PYTHONPATH=$ROOT/.python-packages:$ROOT" \
  --entrypoint python3 "$IMAGE" \
  tools/mtp_fetch.py inventory --out "$MTP_SRC"

docker run --rm --name strata-mtp-fetch \
  -v "$ROOT:$ROOT" -v "$ASSETS:$ASSETS" -w "$ROOT" \
  -e STRATA_SOURCE=modelscope \
  -e "PYTHONPATH=$ROOT/.python-packages:$ROOT" \
  --entrypoint python3 "$IMAGE" \
  tools/mtp_fetch.py fetch --out "$MTP_SRC"

docker run --rm --network none --name strata-mtp-verify \
  -v "$ROOT:$ROOT" -v "$ASSETS:$ASSETS" -w "$ROOT" \
  -e "PYTHONPATH=$ROOT/.python-packages:$ROOT" \
  --entrypoint python3 "$IMAGE" \
  tools/mtp_fetch.py verify --out "$MTP_SRC"

docker run --rm --network none --name strata-mtp-pack \
  -v "$ROOT:$ROOT" -v "$ASSETS:$ASSETS" -w "$ROOT" \
  -e "PYTHONPATH=$ROOT/.python-packages:$ROOT" \
  --entrypoint python3 "$IMAGE" \
  tools/mtp_pack.py --src "$MTP_SRC" --experts q2_0 --out "$MTP_GGUF"

docker run --rm --network none --name strata-mtp-runtime \
  -v "$ROOT:$ROOT" -v "$ASSETS:$ASSETS" -w "$ROOT" \
  -e "PYTHONPATH=$ROOT/.python-packages:$ROOT" \
  --entrypoint python3 "$IMAGE" \
  tools/mtp_rt.py --gguf "$MTP_GGUF" --out "$MTP_RT"

curl -fL --retry 8 --retry-delay 5 \
  -o "$MTP_RT/draft_vocab.bin" \
  https://raw.githubusercontent.com/Niko1221/Strata/c62d5e3a15b40ea51c96aae9e8d36928965f53ab/data/draft_vocab.bin
```

人工核对 MTP runtime：

```bash
cat >"$MTP_RT/SHA256SUMS" <<'EOF'
09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f  experts.bin
c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a  dense.bin
991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd  dense.txt
b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91  draft_vocab.bin
EOF
(cd "$MTP_RT" && sha256sum -c SHA256SUMS)
```

### 6. 手动下载并校验 DeepSeek 资产

Backbone 仓库：`ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF`<br>
DSpark 仓库：`unsloth/DeepSeek-V4.1-Flash-1017-GGUF`

| 文件 | ModelScope URL | 大小 | SHA-256 |
| --- | --- | ---: | --- |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf` | `https://www.modelscope.cn/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/40d2af57944f5a4a5d7cb17f921f30db8c45ecf3/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf` | 5,780,704 B | `f356f198561f56df36f7ab92d7238a926646d76de992c34f15bcb939a5f529b9` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf` | `https://www.modelscope.cn/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/40d2af57944f5a4a5d7cb17f921f30db8c45ecf3/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf` | 39,535,781,184 B | `25af3785ccf925f54137e1ead4c481a8952315dc0b1af6519a99407f67bce8f3` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf` | `https://www.modelscope.cn/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/08d153046911aec47936988a9897a8b9598adffd/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf` | 39,736,712,000 B | `8168d0998fc494def2e1952d83c131425a108660aae9d29190f5a8670ab62b48` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf` | `https://www.modelscope.cn/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/08d153046911aec47936988a9897a8b9598adffd/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf` | 39,036,908,384 B | `9eb9b4de1968f150dfc207eb31c27be9cd9b5666a9a6246bde8a22b184b4bee0` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf` | `https://www.modelscope.cn/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/20577e0adcacb9ce9c225debe076fbd4b974fffe/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf` | 19,923,784,832 B | `7e35f6b4e6fc46f6c4b152b593d84d683c5ece695ae6ad734f556198c041bec1` |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | `https://www.modelscope.cn/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/20577e0adcacb9ce9c225debe076fbd4b974fffe/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | 110,595,290,816 B | `3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3` |
| `dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` | `https://www.modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/6cd2eacaad3f5d9b61e6913c188a934a3a5eac22/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` | 7,976,914,240 B | `6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a` |

逐个下载并人工建立校验表：

```bash
DS="$ASSETS/dsv41-2.00"
DS_REPO=ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF
DS_REV1=40d2af57944f5a4a5d7cb17f921f30db8c45ecf3
DS_REV3=08d153046911aec47936988a9897a8b9598adffd
DS_REV5=20577e0adcacb9ce9c225debe076fbd4b974fffe

curl -fL --retry 8 -C - -o "$DS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf" \
  "https://www.modelscope.cn/models/$DS_REPO/resolve/$DS_REV1/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf"
curl -fL --retry 8 -C - -o "$DS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf" \
  "https://www.modelscope.cn/models/$DS_REPO/resolve/$DS_REV1/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf"
curl -fL --retry 8 -C - -o "$DS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf" \
  "https://www.modelscope.cn/models/$DS_REPO/resolve/$DS_REV3/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf"
curl -fL --retry 8 -C - -o "$DS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf" \
  "https://www.modelscope.cn/models/$DS_REPO/resolve/$DS_REV3/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf"
curl -fL --retry 8 -C - -o "$DS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf" \
  "https://www.modelscope.cn/models/$DS_REPO/resolve/$DS_REV5/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf"
curl -fL --retry 8 -C - -o "$DS/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf" \
  "https://www.modelscope.cn/models/$DS_REPO/resolve/$DS_REV5/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
curl -fL --retry 8 -C - -o "$DS/dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf" \
  "https://www.modelscope.cn/models/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/resolve/6cd2eacaad3f5d9b61e6913c188a934a3a5eac22/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"
```

```bash
cat >"$DS/SHA256SUMS" <<'EOF'
f356f198561f56df36f7ab92d7238a926646d76de992c34f15bcb939a5f529b9  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf
25af3785ccf925f54137e1ead4c481a8952315dc0b1af6519a99407f67bce8f3  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf
8168d0998fc494def2e1952d83c131425a108660aae9d29190f5a8670ab62b48  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf
9eb9b4de1968f150dfc207eb31c27be9cd9b5666a9a6246bde8a22b184b4bee0  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf
7e35f6b4e6fc46f6c4b152b593d84d683c5ece695ae6ad734f556198c041bec1  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf
3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3  DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf
6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a  dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
EOF
(cd "$DS" && sha256sum -c SHA256SUMS)
```

### 7. 人工生成 DeepSeek pack

```bash
DSPACK="$DS/maya-ds4-dspark"
test ! -e "$DSPACK"

DEEPSEEK_SHARDS=(
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf
  DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf
)

docker run --rm --name strata-dsv41-pack \
  -v "$ROOT:$ROOT" -v "$DS:$DS" -w "$ROOT" \
  -e "PYTHONPATH=$ROOT/.python-packages:$ROOT/source:$ROOT" \
  -e PYTHONWARNINGS=ignore \
  --entrypoint python3 "$IMAGE" \
  source/tools/dsv41/pack_gguf.py \
    --model "$DS" \
    --shards "${DEEPSEEK_SHARDS[@]}" \
    --engram "$DS/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf" \
    --dspark-gguf "$DS/dspark/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf" \
    --dspark-noise-token 128799 \
    --dspark-markov-rank 256 \
    --out "$DSPACK"
```

人工核对 pack：

```bash
cat >"$DSPACK/SHA256SUMS" <<'EOF'
87e04d460f8f9ab82a14cb3793fe0c48f85348d81da5f2e93d200bc5dbc2d6f4  ds4.txt
f91733dc81ff262a0e6f65c0c5c3b86aa1390cfcd144bba75858357d62a50284  tensors.txt
08d55d6433f389b14aa115f4ea4eb31d6a2d63df1ae40821f03facf4c2d49142  engram.bin
4d61d22c90ea0cdf15c7e298f7215f23eeffd20f8eed8fb517fda5fb64693d99  tokenizer/chat_template.jinja
547bace837acea98652eef5669c9b01cee1c73ad1d2a44d6fdce9e6a81c22147  tokenizer/merges.txt
0a61021438e5ce6041133495f18fc360a6587aea748b17d953f552d2119ba51e  tokenizer/token_type.json
5a47981b685b75154e29bfb0299584c9b3a4a28679585e86d524c34c065d037f  tokenizer/tokenizer.json
2f0bf5b31f4aa191d5d0d78291927a4f28c9ceb498a5e345b97e926c4cb99c1a  tokenizer/vocab.json
EOF
(cd "$DSPACK" && sha256sum -c SHA256SUMS)
```

### 8. 人工检查 HCU 占用

```bash
test -c /dev/kfd
for d in /dev/dri/renderD128 /dev/dri/renderD129 \
         /dev/dri/renderD130 /dev/dri/renderD131; do
  test -e "$d"
  fuser "$d" 2>/dev/null && {
    echo "$d is busy"
    exit 1
  }
done

docker ps --format '{{.Names}}\t{{.Ports}}'
```

任何目标 render 节点有进程，或已有容器占用它时，先人工确认并处理。不要为了启动本服务停止其他容器。

### 9. 人工渲染并启动 Qwen

先生成一份不存在占位符的配置：

```bash
export K100AI_ROOT="$ROOT"
export K100AI_RUN_DIR="$RUN_DIR"
export K100AI_QWEN_ENGINE=bin/strata
export K100AI_QWEN_PACK="$QWEN_PACK"
export K100AI_QWEN_NATIVE="$QWEN_MODEL/$QWEN_SHARD1"
export K100AI_QWEN_PLE="$QWEN_MODEL/$QWEN_SHARD2"
export K100AI_QWEN_TOKENIZER="$QWEN_PACK/tokenizer"
export K100AI_QWEN_MTP_RT="$ASSETS/qwen3.8-flash-next-iq3-xxs/mtp/rt"

python3 scripts/k100ai/render-config.py \
  configs/k100ai/qwen-server.template.json \
  "$RUN_DIR/qwen-manual.json"

python3 -m json.tool "$RUN_DIR/qwen-manual.json" >/dev/null
grep -F '${' "$RUN_DIR/qwen-manual.json" && exit 1 || true
```

直接创建容器：

```bash
LIBS=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib:

docker run -d --name strata-k100ai-qwen-manual \
  --device=/dev/kfd \
  --device=/dev/dri/renderD128 \
  --shm-size=8g \
  -p 127.0.0.1:18096:18096 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /data:/data:ro \
  -v "$ROOT:$ROOT" \
  -v "$RUN_DIR:$RUN_DIR" \
  -w "$ROOT" \
  -e "LD_LIBRARY_PATH=$LIBS" \
  -e HIP_VISIBLE_DEVICES=0 \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e STRATA_K100AI_EXACT_MMAC=1 \
  -e STRATA_K100AI_EXACT_GATE_FORMATS=IQ3_XXS,IQ3_S,IQ4_XS \
  -e STRATA_K100AI_EXACT_DOWN_FORMATS=Q2_0 \
  -e "PYTHONPATH=$ROOT/.python-packages" \
  --entrypoint python3 \
  "$IMAGE" -m serve.server \
    --engine strata \
    --config "$RUN_DIR/qwen-manual.json" \
    --host 0.0.0.0 \
    --port 18096 \
    --no-api-key
```

### 10. 人工渲染并启动 DeepSeek

```bash
export K100AI_DSV41_PACK="$DSPACK"
export K100AI_DSV41_TOKENIZER="$DSPACK/tokenizer"

python3 scripts/k100ai/render-config.py \
  configs/k100ai/dsv41-server.template.json \
  "$RUN_DIR/dsv41-manual.json"

python3 -m json.tool "$RUN_DIR/dsv41-manual.json" >/dev/null
grep -F '${' "$RUN_DIR/dsv41-manual.json" && exit 1 || true
```

```bash
docker run -d --name strata-k100ai-dsv41-manual \
  --device=/dev/kfd \
  --device=/dev/dri/renderD128 \
  --device=/dev/dri/renderD129 \
  --device=/dev/dri/renderD130 \
  --device=/dev/dri/renderD131 \
  --shm-size=8g \
  -p 127.0.0.1:18097:18097 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /data:/data:ro \
  -v "$ROOT:$ROOT" \
  -v "$RUN_DIR:$RUN_DIR" \
  -w "$ROOT" \
  -e "LD_LIBRARY_PATH=$LIBS" \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  -e DS4_SPLIT=10,10,10,10 \
  -e HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -e "PYTHONPATH=$ROOT/.python-packages" \
  --entrypoint python3 \
  "$IMAGE" -m serve.server \
    --engine strata \
    --config "$RUN_DIR/dsv41-manual.json" \
    --host 0.0.0.0 \
    --port 18097 \
    --no-api-key
```

### 11. 人工检查并停止

```bash
docker logs -f strata-k100ai-qwen-manual
curl --fail http://127.0.0.1:18096/health
```

```bash
docker logs -f strata-k100ai-dsv41-manual
curl --fail http://127.0.0.1:18097/health
```

停止时只操作自己创建的容器：

```bash
docker stop strata-k100ai-qwen-manual
docker rm strata-k100ai-qwen-manual

docker stop strata-k100ai-dsv41-manual
docker rm strata-k100ai-dsv41-manual
```

### 12. 可选人工 API 请求

Qwen：

```bash
curl --fail http://127.0.0.1:18096/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.8-flash-next",
    "messages":[{"role":"user","content":"请用一句话介绍你自己。"}],
    "max_tokens":32,
    "temperature":0
  }'
```

DeepSeek：

```bash
curl --fail http://127.0.0.1:18097/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4.1-flash",
    "messages":[{"role":"user","content":"请用一句话介绍你自己。"}],
    "max_tokens":32,
    "temperature":0
  }'
```

模型下载、pack 生成和 MTP 构造需要数百 GB 空间及较长时间。完整输入、revision 与文件摘要见
[`configs/k100ai/release-inputs.json`](../../../../configs/k100ai/release-inputs.json)。

### 发布包中的便利脚本

以下脚本用于已经理解上述步骤之后减少重复输入，不是唯一入口：

```bash
bash scripts/k100ai/fetch-qwen-iq3xxs.sh "$ASSETS/qwen3.8-flash-next-iq3-xxs"
bash scripts/k100ai/fetch-dsv41-iq2xxs.sh "$ASSETS/dsv41-2.00"
bash scripts/k100ai/prepare-dsv41-pack.sh "$ASSETS/dsv41-2.00" "$DSPACK"
```

launcher 是另一条便利路径：

```bash
K100AI_ASSETS_DIR="$ASSETS" PORT=18096 bash scripts/k100ai/serve-k100ai.sh qwen
K100AI_ASSETS_DIR="$ASSETS" PORT=18097 bash scripts/k100ai/serve-k100ai.sh dsv41
```

服务提供：

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/messages`
- `GET /metrics`
