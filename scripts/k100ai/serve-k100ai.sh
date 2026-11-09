#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=${K100AI_ROOT:-$(cd -- "$script_dir/../.." && pwd -P)}
model=${K100AI_MODEL:-${1:-}}
port=${PORT:-8095}
bind_host=${BIND_HOST:-0.0.0.0}
image=${BASE_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68}
config=${CONFIG:-}
name=${CONTAINER_NAME:-}
run_dir=${K100AI_RUN_DIR:-"$root/run"}
log=${LOG:-}
assets=${K100AI_ASSETS_DIR:-"$root/assets"}

case "$model" in
  qwen)
    devices=(--device=/dev/dri/renderD128)
    visible=0
    engine_env=(
      -e STRATA_K100AI_EXACT_MMAC=1
      -e STRATA_K100AI_EXACT_GATE_FORMATS=IQ3_XXS,IQ3_S,IQ4_XS
      -e STRATA_K100AI_EXACT_DOWN_FORMATS=Q2_0
    )
    name=${name:-strata-k100ai-serve-qwen}
    template="$root/configs/k100ai/qwen-server.template.json"
    qwen_assets="$assets/qwen3.8-flash-next-iq3-xxs"
    export K100AI_QWEN_PACK=${K100AI_QWEN_PACK:-"$qwen_assets/pack/iq3xxs"}
    export K100AI_QWEN_NATIVE=${K100AI_QWEN_NATIVE:-"$qwen_assets/model/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"}
    export K100AI_QWEN_PLE=${K100AI_QWEN_PLE:-"$qwen_assets/model/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"}
    export K100AI_QWEN_TOKENIZER=${K100AI_QWEN_TOKENIZER:-$K100AI_QWEN_PACK/tokenizer}
    export K100AI_QWEN_MTP_RT=${K100AI_QWEN_MTP_RT:-"$qwen_assets/mtp/rt"}
    export K100AI_QWEN_ENGINE=${K100AI_QWEN_ENGINE:-bin/strata-0.1.47}
    if [[ "$K100AI_QWEN_ENGINE" = /* ]]; then
      qwen_engine_path=$K100AI_QWEN_ENGINE
    else
      qwen_engine_path="$root/$K100AI_QWEN_ENGINE"
    fi
    if [[ ! -x "$qwen_engine_path" ]]; then
      echo "missing Qwen engine executable: $qwen_engine_path" >&2
      exit 2
    fi
    for asset in \
      "$K100AI_QWEN_PACK/experts.bin" \
      "$K100AI_QWEN_PACK/dense.bin" \
      "$K100AI_QWEN_PACK/index.txt" \
      "$K100AI_QWEN_NATIVE" \
      "$K100AI_QWEN_PLE" \
      "$K100AI_QWEN_TOKENIZER/tokenizer.json"; do
      if [[ ! -f "$asset" ]]; then
        echo "missing Qwen external asset: $asset" >&2
        echo "set K100AI_QWEN_PACK, K100AI_QWEN_NATIVE, K100AI_QWEN_PLE, K100AI_QWEN_TOKENIZER, or K100AI_ASSETS_DIR" >&2
        exit 2
      fi
    done
    for mtp_file in experts.bin dense.bin dense.txt draft_vocab.bin; do
      if [[ ! -f "$K100AI_QWEN_MTP_RT/$mtp_file" ]]; then
        echo "missing Qwen MTP runtime file: $K100AI_QWEN_MTP_RT/$mtp_file" >&2
        echo "the 787 MiB MTP runtime is external to the release archive" >&2
        exit 2
      fi
    done
    log=${log:-"$run_dir/qwen-server.log"}
    ;;
  dsv41)
    devices=(
      --device=/dev/dri/renderD128
      --device=/dev/dri/renderD129
      --device=/dev/dri/renderD130
      --device=/dev/dri/renderD131
    )
    visible=0,1,2,3
    engine_env=()
    name=${name:-strata-k100ai-serve-dsv41}
    template="$root/configs/k100ai/dsv41-server.template.json"
    export K100AI_DSV41_PACK=${K100AI_DSV41_PACK:-"$assets/dsv41-2.00/maya-ds4-dspark"}
    export K100AI_DSV41_TOKENIZER=${K100AI_DSV41_TOKENIZER:-$K100AI_DSV41_PACK/tokenizer}
    for asset in \
      "$K100AI_DSV41_PACK/ds4.txt" \
      "$K100AI_DSV41_PACK/tensors.txt" \
      "$K100AI_DSV41_PACK/engram.bin" \
      "$K100AI_DSV41_TOKENIZER/tokenizer.json"; do
      if [[ ! -f "$asset" ]]; then
        echo "missing DeepSeek external asset: $asset" >&2
        echo "set K100AI_DSV41_PACK, K100AI_DSV41_TOKENIZER, or K100AI_ASSETS_DIR" >&2
        exit 2
      fi
    done
    log=${log:-"$run_dir/dsv41-server.log"}
    ;;
  *)
    echo "usage: $0 qwen|dsv41" >&2
    exit 2
    ;;
esac

export K100AI_ROOT="$root"
export K100AI_RUN_DIR="$run_dir"
mkdir -p "$run_dir"

research=(
  strata-next-hcu0-xxs-20261015
  strata-next-hcu1-xxs-20261016
  geak-k100ai-hcu23-20261030
)

if [[ -e "$root/bin/strata" && -e "$root/bin/strata-dsv41" ]]; then
  :
else
  echo "the release binaries are missing from $root/bin" >&2
  exit 2
fi
python_packages="$root/.python-packages"
if [[ ! -f "$python_packages/.k100ai-ready" ]]; then
  echo "runtime dependencies are not installed: run $root/scripts/k100ai/setup-runtime.sh" >&2
  exit 2
fi

for container in "${research[@]}"; do
  state=$(docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || true)
  if [[ "$state" == running ]]; then
    echo "refusing to stop research container $container; stop it yourself before starting this service" >&2
    exit 3
  fi
done

if docker inspect "$name" >/dev/null 2>&1; then
  echo "container $name already exists; refusing to replace it" >&2
  exit 4
fi

# preflight is side-effect free and refuses any busy HCU0-3 lease.
bash "$script_dir/preflight.sh"

if [[ -z "$config" ]]; then
  config="$run_dir/${name}-$(date +%Y%m%d-%H%M%S)-$$.json"
  test ! -e "$config" || {
    echo "rendered config already exists: $config" >&2
    exit 4
  }
  python3 "$script_dir/render-config.py" "$template" "$config"
elif [[ ! -f "$config" ]]; then
  echo "missing server config: $config" >&2
  exit 2
fi

libs=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib:
created_id=
container_pid=
cleanup() {
  status=${1:-$?}
  trap - EXIT INT TERM
  if [[ -n "$created_id" ]] && docker inspect "$created_id" >/dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.State.Running}}' "$created_id" 2>/dev/null || true)" == true ]]; then
      docker stop --time 30 "$created_id" >/dev/null 2>&1 || true
    fi
    docker rm "$created_id" >/dev/null 2>&1 || true
  fi
  if [[ -n "$container_pid" ]]; then
    wait "$container_pid" >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap 'cleanup $?' EXIT
trap 'cleanup 130' INT
trap 'cleanup 143' TERM

set +e
created_id=$(docker create \
  --name "$name" \
  --device=/dev/kfd \
  "${devices[@]}" \
  --shm-size=8g \
  -p "127.0.0.1:${port}:${port}" \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /data:/data:ro \
  -v "$root:$root" \
  -v "$run_dir:$run_dir" \
  -w "$root" \
  -e LD_LIBRARY_PATH="$libs" \
  -e HIP_VISIBLE_DEVICES="$visible" \
  -e ROCR_VISIBLE_DEVICES="$visible" \
  "${engine_env[@]}" \
  -e K100AI_ROOT="$root" \
  -e K100AI_RUN_DIR="$run_dir" \
  -e PYTHONPATH="$python_packages" \
  --entrypoint python3 \
  "$image" -m serve.server --engine strata --config "$config" \
  --host "$bind_host" --port "$port" --no-api-key)
create_rc=$?
set -e
if [[ $create_rc -ne 0 || -z "$created_id" ]]; then
  echo "docker create failed; no existing container was removed" >&2
  exit "$create_rc"
fi

# Wait for an interrupted child so traps run immediately when this wrapper
# receives SIGINT/SIGTERM. A foreground child would defer those traps until
# the attached `docker start` happened to exit on its own.
set +e
docker start -a "$created_id" > >(tee "$log") 2>&1 &
container_pid=$!
wait "$container_pid"
rc=$?
set -e
exit "$rc"
