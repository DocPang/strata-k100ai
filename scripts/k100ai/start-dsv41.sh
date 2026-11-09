#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd -- "$script_dir/../.." && pwd -P)
config_arg=${1:?usage: $0 CONFIG.json}
if [[ "$config_arg" = /* ]]; then
  config=$config_arg
else
  config="$root/$config_arg"
fi

image=${BASE_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68}
port=${PORT:-18097}
name=${CONTAINER_NAME:-strata-k100ai-v030}
binary="$root/bin/strata-dsv41"
packages="$root/.python-packages"
expected=4fd967c1fa6769e60298357d55683c18e5522f6689441b12fe5a97489b3a35d6
libs=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib:

test -f "$config" || { echo "missing config: $config" >&2; exit 2; }
test -x "$binary" || { echo "missing executable: $binary" >&2; exit 2; }
actual=$(sha256sum "$binary" | awk '{print $1}')
[[ "$actual" == "$expected" ]] || {
  echo "binary checksum mismatch: expected $expected, got $actual" >&2
  exit 2
}
python3 -m json.tool "$config" >/dev/null
test -f "$packages/.k100ai-ready" || {
  echo "runtime dependencies are missing; run scripts/k100ai/setup-runtime.sh" >&2
  exit 2
}
if docker inspect "$name" >/dev/null 2>&1; then
  echo "container already exists: $name" >&2
  exit 2
fi

bash "$script_dir/preflight.sh"
mkdir -p "$root/run"

args=(
  --rm --name "$name"
  --device=/dev/kfd
  --device=/dev/dri/renderD128
  --device=/dev/dri/renderD129
  --device=/dev/dri/renderD130
  --device=/dev/dri/renderD131
  --shm-size=8g
  -v /opt/hyhal:/opt/hyhal:ro
  -v /data:/data
  -v "$root:$root"
  -w "$root"
  -p "127.0.0.1:${port}:${port}"
  -e LD_LIBRARY_PATH="$libs"
  -e HIP_VISIBLE_DEVICES=0,1,2,3
  -e ROCR_VISIBLE_DEVICES=0,1,2,3
  -e PYTHONPATH="$packages"
  --entrypoint python3
  "$image"
  -m serve.server
  --engine strata
  --config "$config"
  --host 127.0.0.1
  --port "$port"
  --no-api-key
)

exec docker run "${args[@]}"
