#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
bash -n install.sh start.sh stop.sh scripts/config.sh
python3 -m unittest discover -s tests -v
if docker compose version >/dev/null 2>&1; then
  docker compose --env-file deployment.env.example config --quiet
else
  echo 'Compose plugin unavailable: optional compose.yaml validation skipped.'
fi
