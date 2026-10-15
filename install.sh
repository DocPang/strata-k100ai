#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION=v0.1.0-rc.2
BASE_DEFAULT='harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68'
BASE_ID='sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad'
base="$BASE_DEFAULT"; local_only=0; bundle=''; profile=next-iq3s-single
usage() {
  cat <<'HELP'
Usage: ./install.sh [--base-image IMAGE] [--bundle FILE] [--local-only]
                    [--profile next-iq3s-single]
Installs a local image; never starts a service or changes the host driver.
--local-only requires BOTH a local compatible base image and a local runtime bundle.
Only the pinned DTK image content is supported, including local aliases of it.
HELP
}
while (($#)); do
  case "$1" in
    --base-image|--bundle|--profile)
      (($# >= 2)) || { usage >&2; exit 2; }
      case "$1" in --base-image) base="$2";; --bundle) bundle="$2";; --profile) profile="$2";; esac
      shift 2;;
    --local-only) local_only=1; shift;;
    -h|--help) usage; exit 0;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2;;
  esac
done
[[ "$profile" == next-iq3s-single ]] || { echo 'This release supports only next-iq3s-single; XXS/TP2 are not released.' >&2; exit 2; }
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'Linux x86_64 host required.' >&2; exit 1; }
command -v docker >/dev/null || { echo 'Install Docker Engine first.' >&2; exit 1; }
docker info >/dev/null
if ! docker image inspect "$base" >/dev/null 2>&1; then
  ((local_only == 0)) || { echo 'Local base image missing; --local-only prohibits pulling.' >&2; exit 1; }
  docker pull "$base"
fi
actual_id="$(docker image inspect --format '{{.Id}}' "$base")"
[[ "$actual_id" == "$BASE_ID" ]] || { echo "Unsupported base image content: $actual_id. See docs/INSTALL.md." >&2; exit 1; }
artifact=strata-c4-linux-amd64.tar.gz
mkdir -p "$ROOT/.cache/artifacts"
if [[ -z "$bundle" ]]; then
  bundle="$ROOT/.cache/artifacts/$artifact"
  if [[ ! -f "$bundle" ]]; then
    ((local_only == 0)) || { echo 'Runtime bundle missing; pass --bundle FILE for offline installation.' >&2; exit 1; }
    command -v curl >/dev/null || { echo 'Install curl or pass --bundle FILE.' >&2; exit 1; }
    curl --fail --location --retry 3 --output "$bundle.part" \
      "https://github.com/DocPang/strata-k100ai/releases/download/$VERSION/$artifact"
    mv "$bundle.part" "$bundle"
  fi
fi
[[ -f "$bundle" ]] || { echo 'Runtime bundle does not exist.' >&2; exit 1; }
expected="$(awk '{print $1}' "$ROOT/engines/strata/c4/artifact.sha256")"
if command -v sha256sum >/dev/null; then actual="$(sha256sum "$bundle" | awk '{print $1}')";
else actual="$(shasum -a 256 "$bundle" | awk '{print $1}')"; fi
[[ "$actual" == "$expected" ]] || { echo 'Runtime bundle checksum mismatch; refusing installation.' >&2; exit 1; }
stage="$(mktemp -d "$ROOT/.cache/install.XXXXXXXX")"
trap 'rm -rf "$stage"' EXIT
tar -tzf "$bundle" | awk '/^\// || /(^|\/)\.\.(\/|$)/ {bad=1} END {exit bad}' || { echo 'Unsafe archive paths.' >&2; exit 1; }
tar -xzf "$bundle" -C "$stage"
cp "$ROOT/containers/strata/Dockerfile" "$stage/Dockerfile"
cp "$ROOT/containers/strata/entrypoint.py" "$stage/entrypoint.py"
cp "$ROOT/profiles/$profile/profile.json" "$stage/profile.json"
# A Docker-backed default builder can read the Engine's local image store.
# Never use a docker-container/remote builder for the local-image installation path.
if docker buildx version >/dev/null 2>&1; then
  driver="$(docker buildx inspect default --format '{{.Driver}}')"
  [[ "$driver" == docker ]] || { echo 'Use a Docker-backed default builder for local image reuse.' >&2; exit 1; }
fi
alias_image=k100ai-base-dtk:2604-caeff0e9
# Local alias avoids resolving the registry digest again during FROM processing.
docker tag "$base" "$alias_image"
BUILDX_BUILDER=default docker build --pull=false --network=none \
  --build-arg "BASE_IMAGE=$alias_image" --tag "k100ai/strata-c4:$VERSION" "$stage"
if [[ ! -f "$ROOT/deployment.env" ]]; then
  cp "$ROOT/deployment.env.example" "$ROOT/deployment.env"
  chmod 600 "$ROOT/deployment.env"
  if command -v openssl >/dev/null; then
    key="$(openssl rand -hex 24)"
    sed -i "s/CHANGE_ME_GENERATE_A_PRIVATE_KEY/$key/" "$ROOT/deployment.env"
  fi
fi
printf '\nInstalled k100ai/strata-c4:%s\nEdit deployment.env paths/device and run ./start.sh.\n' "$VERSION"
