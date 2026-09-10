from pathlib import Path
import json
import os
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser

from IPython.display import HTML, display


# Change this path only if the complete project folder is moved.
PROJECT_FOLDER = Path(
    "/Users/frasercochrane/Documents/Codex/2026-08-23/ed/velox-c1-orbital-decay"
).expanduser().resolve()

if not (PROJECT_FOLDER / "run_web.py").is_file():
    raise FileNotFoundError(f"run_web.py was not found in {PROJECT_FOLDER}")
if not (PROJECT_FOLDER / "docs" / "index.html").is_file():
    raise FileNotFoundError(f"Website files were not found in {PROJECT_FOLDER / 'docs'}")

# Stop a process from an earlier execution of this notebook. Other servers are untouched.
if "_velox_server" in globals() and _velox_server.poll() is None:
    _velox_server.terminate()
    _velox_server.wait(timeout=10)
if "_velox_log" in globals() and not _velox_log.closed:
    _velox_log.close()

# Use a fresh available port rather than silently attaching to stale code on port 8000.
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
    port_socket.bind(("127.0.0.1", 0))
    SERVER_PORT = port_socket.getsockname()[1]
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

def server_is_ready():
    try:
        with urlopen(f"{SERVER_URL}/api/health", timeout=1.0) as response:
            return response.status == 200 and json.load(response).get("status") == "ok"
    except (OSError, TimeoutError, URLError, json.JSONDecodeError):
        return False

output_folder = PROJECT_FOLDER / "outputs"
output_folder.mkdir(exist_ok=True)
environment = os.environ.copy()
environment["PYTHONPATH"] = str(PROJECT_FOLDER / "src")
environment["MPLCONFIGDIR"] = str(output_folder / ".matplotlib")
environment["VELOX_WEB_HOST"] = "127.0.0.1"
environment["VELOX_WEB_PORT"] = str(SERVER_PORT)
Path(environment["MPLCONFIGDIR"]).mkdir(exist_ok=True)

_velox_log = (output_folder / "jupyter_web_server.log").open("a", encoding="utf-8")
_velox_server = subprocess.Popen(
    [sys.executable, str(PROJECT_FOLDER / "run_web.py")],
    cwd=PROJECT_FOLDER,
    env=environment,
    stdout=_velox_log,
    stderr=subprocess.STDOUT,
)

deadline = time.monotonic() + 60.0
while time.monotonic() < deadline and not server_is_ready():
    if _velox_server.poll() is not None:
        _velox_log.flush()
        log_path = output_folder / "jupyter_web_server.log"
        log_text = log_path.read_text(encoding="utf-8")
        raise RuntimeError(f"Server startup failed.\n\n{log_text[-4000:]}")
    time.sleep(0.5)

if not server_is_ready():
    raise RuntimeError("The website did not become ready within 60 seconds.")

webbrowser.open(SERVER_URL)
display(HTML(
    f'<b>VELOX-C1 is ready:</b> <a href="{SERVER_URL}" target="_blank">'
    'open the orbital decay simulator</a>'
))
