#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd -- "$script_dir/../.." && pwd -P)
image=${BASE_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68}
packages="$root/.python-packages"
ready="$packages/.k100ai-ready"

if [[ -f "$ready" ]]; then
  docker run --rm \
    -v "$root:$root" \
    -w "$root" \
    -e K100AI_PACKAGES="$packages" \
    --entrypoint /bin/bash \
    "$image" -lc '
      PYTHONPATH="$K100AI_PACKAGES" python3 -c \
        "import jinja2, numpy, PIL, psutil, regex, requests, yaml"
    '
  echo "runtime dependencies already installed: $packages"
  exit 0
fi
test ! -e "$packages" || {
  echo "incomplete runtime package directory exists: $packages" >&2
  echo "remove it explicitly and rerun this setup script" >&2
  exit 2
}

docker run --rm \
  -v "$root:$root" \
  -w "$root" \
  -e K100AI_ROOT="$root" \
  -e K100AI_PACKAGES="$packages" \
  --entrypoint /bin/bash \
  "$image" -lc '
    set -euo pipefail
    python3 -m pip install --disable-pip-version-check --no-cache-dir \
      --target "$K100AI_PACKAGES" -r "$K100AI_ROOT/requirements.txt"
    PYTHONPATH="$K100AI_PACKAGES" python3 -c \
      "import jinja2, numpy, PIL, psutil, regex, requests, yaml"
    sha256sum "$K100AI_ROOT/requirements.txt" > "$K100AI_PACKAGES/.k100ai-ready"
  ' 2>&1 | tee "$root/runtime-setup.log"

test -f "$ready"
echo "runtime dependencies installed: $packages"
