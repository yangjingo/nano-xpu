"""Real subprocess helpers shared by control and integration tests."""
import json
import os
import socket
import subprocess
import sys
from tests.helpers import ROOT as ROOT

def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def helper(root, payload, check=True, cwd=None):
    environment = {**os.environ, "NANO_XPU_STATE_DIR": str(root)}
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    run = subprocess.run([sys.executable, "-m", "nanoxpu.deployment.helper"], input=json.dumps(payload),
                         env=environment, capture_output=True, text=True, timeout=45, cwd=cwd, **options)
    response = json.loads(run.stdout)
    if check:
        assert run.returncode == 0 and response["ok"], response
        return response["result"]
    return response
