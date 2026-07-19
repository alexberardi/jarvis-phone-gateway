#!/usr/bin/env bash
# Dev runner: venv bootstrap + uvicorn with the WebSocket protocol lib.
# (Bare uvicorn has no WS support -> 404 on media-stream upgrade -> 0-second
# calls. That was P0 failure ladder item 1; uvicorn[standard] ships wsproto.)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

.venv/bin/python -c "import wsproto" 2>/dev/null || {
  echo "wsproto missing — reinstalling requirements (uvicorn[standard] needed)"
  .venv/bin/pip install -r requirements.txt
}

PORT="${SERVER_PORT:-7713}"
HOST="${SERVER_HOST:-0.0.0.0}"
exec .venv/bin/uvicorn main:app --host "$HOST" --port "$PORT"
