#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source scripts/config.sh
load_deployment_env
[[ $# == 0 || ( $# == 1 && "$1" == --check ) ]] || { echo 'Usage: ./start.sh [--check]' >&2; exit 2; }
for variable in MODEL_DIR PACK_DIR MTP_DIR HYHAL_DIR; do
  value="${!variable:-}"
  [[ "$value" == /* && -d "$value" ]] || { echo "$variable must be an existing absolute directory" >&2; exit 1; }
done
[[ -c /dev/kfd && -c "${RENDER_DEVICE:-}" ]] || { echo 'Set an existing K100AI render device; /dev/kfd is also required.' >&2; exit 1; }
[[ "${PORT:-}" =~ ^[0-9]+$ ]] && ((10#$PORT > 0 && 10#$PORT <= 65535)) || { echo 'Invalid PORT' >&2; exit 1; }
[[ ${#STRATA_API_KEY} -ge 16 && "$STRATA_API_KEY" != CHANGE_ME_GENERATE_A_PRIVATE_KEY ]] || { echo 'Set a private STRATA_API_KEY of at least 16 characters.' >&2; exit 1; }
for shard in "${SHARD1:-}" "${SHARD2:-}"; do
  [[ -n "$shard" && "$shard" != */* && -f "$MODEL_DIR/$shard" ]] || { echo 'Set SHARD1/SHARD2 to existing filenames inside MODEL_DIR.' >&2; exit 1; }
done
[[ -d "$PACK_DIR/tokenizer" ]] || { echo 'PACK_DIR/tokenizer is missing.' >&2; exit 1; }
[[ -d "$EXTRAS_DIR" ]] || { echo 'EXTRAS_DIR must exist.' >&2; exit 1; }
extras="$(cd "$EXTRAS_DIR" && pwd)"
if [[ "$PROFILE" == next-iq3xxs-single ]]; then image=k100ai/strata-xxs-20261023-k100ai022:v0.2.1; else image=k100ai/strata-c4:v0.1.0-rc.6; fi
docker image inspect "$image" >/dev/null
args=(run --detach --name "k100ai-$INSTANCE" --restart=no
  --publish "$BIND_HOST:$PORT:8080" --env-file "$PWD/deployment.env"
  --device /dev/kfd:/dev/kfd --device "$RENDER_DEVICE:$RENDER_DEVICE"
  --mount "type=bind,src=$MODEL_DIR,dst=/models,readonly"
  --mount "type=bind,src=$PACK_DIR,dst=/pack,readonly"
  --mount "type=bind,src=$MTP_DIR,dst=/mtp,readonly"
  --mount "type=bind,src=$HYHAL_DIR,dst=/opt/hyhal,readonly"
  --mount "type=bind,src=$PWD/profiles/$PROFILE/settings.json,dst=/config/settings.json,readonly"
  --mount "type=bind,src=$extras,dst=/extras,readonly"
  --shm-size=2g --ulimit memlock=-1:-1 --security-opt seccomp=unconfined
  --health-cmd "python3 -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=10)\""
  --health-interval=30s --health-timeout=15s --health-start-period=20m --health-retries=3
  "$image")
if [[ "${1:-}" == --check ]]; then
  printf '%q ' docker "${args[@]}"; printf '\n'
  echo 'Configuration paths and devices checked; no container started.'
  exit 0
fi
exec docker "${args[@]}"
