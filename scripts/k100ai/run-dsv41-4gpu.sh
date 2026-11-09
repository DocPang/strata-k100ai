#!/usr/bin/env bash
set -euo pipefail

root=/data/strata-dsv41-k100ai
image=${BASE_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10}
exe=${DSV41_EXE:-"$root/build-gfx928-dsv41/strata-dsv41"}
entrypoint=${ENTRYPOINT_OVERRIDE:-"$exe"}
name=${CONTAINER_NAME:-strata-k100ai-dsv41-4gpu}
log=${LOG:-"$root/dsv41-4gpu-$(date +%Y%m%d-%H%M%S).log"}
DS4_SPLIT=${DS4_SPLIT:-10,10,10,10}
export DS4_SPLIT

engine_defaults=(
  --experts "${DSV41_EXPERTS:-6}"
  --vision "${DSV41_VISION:-off}"
  --max-context "${DSV41_MAX_CONTEXT:-1048576}"
  --prefill-chunk "${DSV41_PREFILL_CHUNK:-1024}"
  --vram-experts "${DSV41_VRAM_EXPERTS:-1.0}"
  --ssd-cache "${DSV41_SSD_CACHE:-0}"
  --dspark "${DSV41_DSPARK:-on}"
  --dspark-conf "${DSV41_DSPARK_CONF:-0.65}"
  --tp "${DSV41_TP:-4}"
)

research=(
  strata-next-hcu0-xxs-20261015
  strata-next-hcu1-xxs-20261016
  geak-k100ai-hcu23-20261030
)
running=()
extra_env=()
for key in DS4_SPLIT DS4_TP_ALIGN DS4_PROF DS4_PROF_SYNC DS4_Q5_WARPS DS4_GEMV_F32_WARPS8 \
           DS4_ROUTE_WARPS16 DS4_ROUTE_PUBLISH_WARPS16 DS4_ROUTE_AHEAD_WARPS16 DS4_TLOG DS4_TLOG2 \
           DS4_NO_EOS DS4_TOKLOG DS4_ROUND_LOG DS4_DSPARK_DUMP DS4_DSPARK_MAX DS4_DSPARK_SAMPLE; do
  if value=$(printenv "$key"); then
    extra_env+=(-e "$key=$value")
  fi
done

cleanup() {
  for container in "${running[@]}"; do
    if ! docker start "$container" >/dev/null 2>&1; then
      echo "failed to restore research container: $container" >&2
    fi
  done
}
trap cleanup EXIT

if [[ ! -x "$exe" ]]; then
  echo "missing executable: $exe" >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo "usage: $0 --pack DIR [strata-dsv41 arguments...]" >&2
  exit 2
fi

for container in "${research[@]}"; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" == true ]]; then
    running+=("$container")
  fi
done

for container in "${running[@]}"; do
  docker stop --time 30 "$container" >/dev/null
done

bash "$root/scripts/k100ai/preflight.sh"

libs=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib:
set +e
docker run --rm --name "$name" \
  --device=/dev/kfd \
  --device=/dev/dri/renderD128 \
  --device=/dev/dri/renderD129 \
  --device=/dev/dri/renderD130 \
  --device=/dev/dri/renderD131 \
  --shm-size=8g \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /data:/data \
  -w "$root" \
  -e LD_LIBRARY_PATH="$libs" \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e ROCR_VISIBLE_DEVICES=0,1,2,3 \
  "${extra_env[@]}" \
  --entrypoint "$entrypoint" \
  "$image" "${engine_defaults[@]}" "$@" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e

exit "$rc"
