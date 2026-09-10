#!/bin/bash

# Launch the local VELOX-C1 website from Finder or Terminal on macOS.
set -euo pipefail

# Resolve the repository independently of the caller's current directory.
PROJECT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIRECTORY"

# Prefer an activated environment, then the repository environment. A fresh
# local environment is created when neither provides the required packages.
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON_EXECUTABLE="$VIRTUAL_ENV/bin/python"
elif [[ -x "$PROJECT_DIRECTORY/.venv/bin/python" ]]; then
  PYTHON_EXECUTABLE="$PROJECT_DIRECTORY/.venv/bin/python"
else
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3.10 or newer is required. Install Python, then run this launcher again."
    read -r -p "Press Return to close."
    exit 1
  fi
  echo "Creating the local Python environment…"
  python3 -m venv "$PROJECT_DIRECTORY/.venv"
  PYTHON_EXECUTABLE="$PROJECT_DIRECTORY/.venv/bin/python"
fi

# Install the project only when the selected environment is not ready.
if ! "$PYTHON_EXECUTABLE" -c "import fastapi, matplotlib, numpy, PIL, pymsis, uvicorn, velox_decay" >/dev/null 2>&1; then
  echo "Installing VELOX-C1 and its Python dependencies…"
  "$PYTHON_EXECUTABLE" -m pip install -e "$PROJECT_DIRECTORY"
fi

# Allocate a fresh loopback port so this launch cannot attach to an older
# Python process that still has an earlier version of the package in memory.
SERVER_PORT="$("$PYTHON_EXECUTABLE" -c 'import socket; sock = socket.socket(); sock.bind(("127.0.0.1", 0)); print(sock.getsockname()[1]); sock.close()')"
export VELOX_WEB_HOST="127.0.0.1"
export VELOX_WEB_PORT="$SERVER_PORT"
SERVER_URL="http://127.0.0.1:$SERVER_PORT"
HEALTH_URL="$SERVER_URL/api/health"

echo "Starting VELOX-C1 at $SERVER_URL"
echo "Keep this Terminal window open while using the simulator."
echo "Press Control-C here when you are finished."

# Run the service as a child so this launcher can wait for readiness before
# opening Safari and can stop the child cleanly when the window is closed.
"$PYTHON_EXECUTABLE" "$PROJECT_DIRECTORY/run_web.py" &
SERVER_PROCESS=$!

stop_server() {
  if kill -0 "$SERVER_PROCESS" >/dev/null 2>&1; then
    kill "$SERVER_PROCESS" >/dev/null 2>&1 || true
    wait "$SERVER_PROCESS" >/dev/null 2>&1 || true
  fi
}
trap stop_server EXIT INT TERM

# Wait up to ten seconds for the health endpoint before opening the page.
for _attempt in {1..40}; do
  if curl --silent --fail --max-time 1 "$HEALTH_URL" >/dev/null 2>&1; then
    open "$SERVER_URL"
    wait "$SERVER_PROCESS"
    exit $?
  fi
  if ! kill -0 "$SERVER_PROCESS" >/dev/null 2>&1; then
    wait "$SERVER_PROCESS"
    exit $?
  fi
  sleep 0.25
done

echo "The server did not become ready within ten seconds. Review the messages above."
exit 1
