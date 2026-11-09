#!/usr/bin/env bash
set -euo pipefail

root=/data/strata-dsv41-k100ai
name=${CONTAINER_NAME:-strata-k100ai-qwen-build}
log=${LOG:-"$root/qwen-build.log"}
pid_file=${PID_FILE:-"$root/qwen-build.pid"}
image=${BASE_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10}
build_dir=${BUILD_DIR:-"$root/build-gfx928-qwen"}
dsv41=${DSV41:-0}

nohup docker run --rm --name "$name" \
  -v /data:/data \
  -w "$root" \
  -e JOBS="${JOBS:-64}" \
  -e TARGET="${TARGET:-strata}" \
  -e BUILD_DIR="$build_dir" \
  -e DSV41="$dsv41" \
  "$image" \
  bash "$root/scripts/k100ai/build-in-dtk.sh" "$root" \
  >"$log" 2>&1 &

echo $! >"$pid_file"
echo "build pid=$(cat "$pid_file") log=$log"
