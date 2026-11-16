"""One detached supervisor per deployment role. No restart or process-name killing."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from nanoxpu.deployment.state import (
    OWNER_ENV, atomic_json, process_identity, read_json, safe_name, state_root, stop_owned_tree,
)


def detached_options():
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def role_directory(name, role_id, root=None):
    return Path(root or state_root()) / safe_name(name) / safe_name(role_id)


def run(payload):
    directory = role_directory(payload["name"], payload["spec"]["id"])
    record_path = directory / "state.json"
    record = read_json(record_path)
    if record is None or record["owner"] != payload["owner"]:
        raise RuntimeError("supervisor bootstrap lost ownership")
    secret = payload.get("auth_token", "")
    spec = payload["spec"]
    environment = {**os.environ, **spec["env"], OWNER_ENV: payload["owner"]}
    if secret:
        environment["NANO_XPU_INTERNAL_TOKEN"] = secret
        if spec["role"] == "router":
            environment["OPENAI_API_KEY"] = secret
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    child = None
    reader = None
    failure = None
    try:
        child = subprocess.Popen(spec["argv"], cwd=spec["workdir"], env=environment,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 **detached_options())

        def copy_log():
            with (directory / "worker.log").open("ab") as log:
                for line in iter(child.stdout.readline, b""):
                    if secret:
                        line = line.replace(secret.encode(), b"[REDACTED]")
                    log.write(line)
                    log.flush()

        reader = threading.Thread(target=copy_log, daemon=True)
        reader.start()
        record["worker"] = process_identity(child.pid, payload["owner"])
        record["phase"] = "RUNNING"
        atomic_json(record_path, record)
        while child.poll() is None:
            request = read_json(directory / "stop.json")
            if stopping.is_set() or (request and request.get("owner") == payload["owner"]):
                record["phase"] = "STOPPING"
                atomic_json(record_path, record)
                stop_owned_tree(record["worker"], payload["drain_seconds"])
                child.wait(timeout=3)
                record["phase"] = "STOPPED"
                break
            time.sleep(0.05)
        else:
            record["phase"] = "FAILED"
            failure = f"worker exited unexpectedly with code {child.returncode}"
        record["exit_code"] = child.poll()
    except Exception as exc:
        failure = str(exc).replace(secret, "[REDACTED]") if secret else str(exc)
        record["phase"] = "FAILED"
        if child is not None:
            try:
                identity = record.get("worker") or process_identity(child.pid, payload["owner"])
                stop_owned_tree(identity, payload["drain_seconds"])
                child.wait(timeout=3)
            except Exception:
                record["phase"] = "CLEANUP_FAILED"
    finally:
        if reader is not None:
            reader.join(timeout=2)
        if child is not None and child.poll() is not None and child.stdout is not None:
            child.stdout.close()
        if failure:
            record["error"] = failure
        atomic_json(record_path, record)


def main():
    data = sys.stdin.buffer.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise SystemExit("supervisor bootstrap exceeds 1 MiB")
    run(json.loads(data))


if __name__ == "__main__":
    main()
