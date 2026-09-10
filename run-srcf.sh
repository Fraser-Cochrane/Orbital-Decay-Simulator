#!/bin/bash
set -euo pipefail

APP_ROOT="/home/fwzc2/apps/orbital-decay"
RUNTIME_ROOT="/home/fwzc2/var/orbital-decay"
SOCKET_PATH="$APP_ROOT/web.sock"

cd "$APP_ROOT"
source "$APP_ROOT/.venv/bin/activate"
mkdir -p "$RUNTIME_ROOT/outputs" "$RUNTIME_ROOT/matplotlib" "$RUNTIME_ROOT/logs"
rm -f "$SOCKET_PATH"

export MPLBACKEND=Agg
export MPLCONFIGDIR="$RUNTIME_ROOT/matplotlib"
export PYTHONUNBUFFERED=1
export VELOX_DECAY_OUTPUT_DIR="$RUNTIME_ROOT/outputs"
export VELOX_WEB_ALLOW_FULL_RUNS=0
export VELOX_WEB_MAX_ACTIVE_JOBS=2
export VELOX_WEB_MAX_JOB_RECORDS=50

exec uvicorn velox_decay.web:app \
  --uds "$SOCKET_PATH" \
  --workers 1 \
  --proxy-headers \
  --forwarded-allow-ips="*" \
  --access-log \
  >>"$RUNTIME_ROOT/logs/service.log" 2>&1
