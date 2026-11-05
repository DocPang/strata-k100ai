#!/usr/bin/env bash
set -euo pipefail

devices=(
  /dev/dri/renderD128
  /dev/dri/renderD129
  /dev/dri/renderD130
  /dev/dri/renderD131
)

# fuser can miss a lease when the serving process lives in a container. Inspect
# running containers explicitly so a differently named service cannot bypass
# the launcher's safety check. HCU4-7 use renderD132-135 and remain untouched.
while IFS= read -r container_id; do
  [[ -n "$container_id" ]] || continue
  container_name=$(docker inspect -f '{{.Name}}' "$container_id")
  while IFS= read -r host_device; do
    for device in "${devices[@]}"; do
      if [[ "$host_device" == "$device" ]]; then
        echo "busy leased device: $device mapped by running container $container_name" >&2
        exit 1
      fi
    done
  done < <(docker inspect -f '{{range .HostConfig.Devices}}{{println .PathOnHost}}{{end}}' "$container_id")
done < <(docker ps -q)

for device in "${devices[@]}"; do
  if [[ ! -c "$device" ]]; then
    echo "missing device: $device" >&2
    exit 1
  fi
  if fuser "$device" >/dev/null 2>&1; then
    echo "busy leased device: $device" >&2
    fuser -v "$device" >&2 || true
    exit 1
  fi
done

echo "HCU0-3 are free"
