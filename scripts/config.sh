#!/usr/bin/env bash
# Read literal KEY=value settings without sourcing or evaluating user input.
load_deployment_env() {
  [[ -f deployment.env ]] || { echo 'Run install.sh and edit deployment.env first.' >&2; return 1; }
  BIND_HOST=127.0.0.1; PORT=8080; HYHAL_DIR=/opt/hyhal; EXTRAS_DIR=./extras
  CONTROL_VECTOR=''; INSTANCE=next-iq3s-single; STRATA_API_KEY=''; SHARD1=''; SHARD2=''
  local line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == *=* ]] || { echo 'Expected literal KEY=value in deployment.env' >&2; return 1; }
    key="${line%%=*}"; value="${line#*=}"
    case "$key" in
      MODEL_DIR|SHARD1|SHARD2|PACK_DIR|MTP_DIR|HYHAL_DIR|RENDER_DEVICE|BIND_HOST|PORT|STRATA_API_KEY|EXTRAS_DIR|CONTROL_VECTOR|INSTANCE)
        printf -v "$key" '%s' "$value";;
      *) echo "Unknown deployment setting: $key" >&2; return 1;;
    esac
  done < deployment.env
  [[ "$INSTANCE" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid INSTANCE' >&2; return 1; }
}
