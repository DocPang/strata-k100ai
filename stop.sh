#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source scripts/config.sh
load_deployment_env
# Remove only this deployment's container; never prune images, caches or other services.
docker stop "k100ai-$INSTANCE"
docker rm "k100ai-$INSTANCE"
