#!/bin/sh
# Start Echo AI: folder watcher plus web server in one command.
# Stop both with Ctrl-C.
cd "$(dirname "$0")"

PORT="${OTTER_PORT:-8000}"

# Point Python's SSL at certifi's CA bundle. Without this, uv-installed
# Pythons on macOS can fail Google token refreshes with
# CERTIFICATE_VERIFY_FAILED (hit live 2026-07-17).
CERT_FILE="$(.venv/bin/python -m certifi 2>/dev/null)"
if [ -n "$CERT_FILE" ]; then
  export SSL_CERT_FILE="$CERT_FILE"
  export REQUESTS_CA_BUNDLE="$CERT_FILE"
fi

.venv/bin/python -m app.watcher &
WATCHER_PID=$!
trap 'kill "$WATCHER_PID" 2>/dev/null' EXIT INT TERM

echo "Echo AI running: http://localhost:$PORT (Ctrl-C stops everything)"
# Bind explicitly to loopback. This is uvicorn's default too, but
# PLAN.md's security rules ask for localhost-only, and a default is
# not a statement.
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "$PORT"
