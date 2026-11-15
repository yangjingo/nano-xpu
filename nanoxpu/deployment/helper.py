"""Fixed remote helper: one bounded JSON request on stdin, one JSON result on stdout."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Literal
import uuid

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator

from nanoxpu.deployment.schema import StrictModel, Port, ipv4
from nanoxpu.deployment.state import (
    OWNER_ENV, atomic_json, file_lock, owned_process, private_dir, process_identity,
    read_json, safe_name, state_root, stop_owned_tree,
)
from nanoxpu.deployment.supervisor import detached_options, role_directory


class ProcessSpec(StrictModel):
    id: str
    role: Literal["router", "colocated", "prefill", "decode"]
    ip: str
    advertise_ip: str
    argv: list[str] = Field(min_length=1, max_length=1024)
    env: dict[str, str]
    workdir: str
    http_port: Port
    distributed_port: Port | None
    kv_port: Port | None
    gpus: list[int]
    instance_id: str

    _names = field_validator("id", "instance_id")(safe_name)
    _ip = field_validator("ip", "advertise_ip")(ipv4)

    @field_validator("argv")
    @classmethod
    def arguments(cls, value):
        if any(not arg or "\x00" in arg or len(arg) > 16384 for arg in value):
            raise ValueError("invalid process argument")
        return value

    @field_validator("env")
    @classmethod
    def environment(cls, value):
        if set(value) - {"CUDA_VISIBLE_DEVICES"}:
            raise ValueError("only the GPU mapping belongs in persisted process specs")
        if any("\x00" in item for item in value.values()):
            raise ValueError("invalid environment value")
        return value

    @field_validator("gpus")
    @classmethod
    def devices(cls, value):
        if any(device < 0 for device in value) or len(set(value)) != len(value):
            raise ValueError("invalid device reservation")
        return value


class HelperRequest(StrictModel):
    action: Literal["preflight", "start", "status", "logs", "stop"]
    name: str
    role_id: str
    spec: ProcessSpec | None = None
    config_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    generation: str | None = None
    owner: str | None = None
    auth_token: SecretStr | None = None
    drain_seconds: float = Field(default=30, gt=0, le=3600, allow_inf_nan=False)
    log_bytes: int = Field(default=16384, ge=1, le=65536)
    expected_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    model_path: str | None = None
    router_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")

    _names = field_validator("name", "role_id")(safe_name)

    @model_validator(mode="after")
    def required_fields(self):
        if self.action in ("start", "preflight") and (self.spec is None or self.spec.id != self.role_id):
            raise ValueError("start/preflight require a matching process spec")
        if self.action == "start":
            if not self.config_hash or not self.generation:
                raise ValueError("start requires config hash and generation")
            safe_name(self.generation)
        if self.action == "stop" and not self.owner:
            raise ValueError("stop requires the recorded ownership marker")
        return self


def require_ports_free(spec):
    for port in (spec.http_port, spec.distributed_port, spec.kv_port):
        if port is not None:
            with socket.socket() as sock:
                # Binding wildcard catches conflicts on any interface on this host.
                sock.bind(("0.0.0.0", port))


def require_resources_free(request, root):
    for path in Path(root).glob("*/*/state.json"):
        record = read_json(path)
        if record["phase"] in ("STOPPED", "FAILED") and owned_process(record.get("worker")) is None:
            continue
        if path == role_directory(request.name, request.role_id, root) / "state.json":
            continue
        other = record["spec"]
        if set(other["gpus"]) & set(request.spec.gpus):
            raise RuntimeError("GPU reservation overlaps an existing deployment; stop its owner first")
        ports = {request.spec.http_port, request.spec.distributed_port, request.spec.kv_port} - {None}
        if ports & ({other["http_port"], other["distributed_port"], other["kv_port"]} - {None}):
            raise RuntimeError("port reservation overlaps an existing deployment")


def preflight(request, root):
    spec = request.spec
    executable = Path(spec.argv[0])
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("configured executable is missing or not executable")
    if not Path(spec.workdir).is_dir():
        raise ValueError("configured workdir does not exist")
    require_ports_free(spec)
    require_resources_free(request, root)
    evidence = {"ports": "available", "executable": str(executable), "pid": os.getpid()}
    if request.expected_revision:
        from nanoxpu.runtime_identity import runtime_identity
        # Query the actual entry point as well as this helper, not an unrelated workdir HEAD.
        if "_worker" not in spec.argv:
            raise ValueError("runtime process must use the internal _worker entry point")
        result = subprocess.run([*spec.argv[:spec.argv.index("_worker")], "runtime-info"], cwd=spec.workdir,
                                capture_output=True, text=True, timeout=30, check=True)
        identities = {"worker": json.loads(result.stdout), "helper": runtime_identity()}
        for identity in identities.values():
            if (identity.get("revision") != request.expected_revision or identity.get("clean") is not True
                or Path(identity.get("source_root", "")).resolve() != Path(spec.workdir).resolve()
                or Path(identity.get("package_root", "")).resolve() != (Path(spec.workdir) / "nanoxpu").resolve()):
                raise ValueError("actual worker/helper package must come from the clean profile checkout")
        evidence["runtime"] = identities
        evidence["revision"] = request.expected_revision
    if request.model_path:
        from nanoxpu.server.metadata import model_fingerprints
        evidence["model"] = model_fingerprints(request.model_path)
    if request.router_revision:
        # The build receipt binds the source commit to this exact binary, not just --version.
        from nanoxpu.utils.hashing import sha256_file
        receipt = read_json(Path(str(executable) + ".source.json"))
        actual = sha256_file(executable)
        if not receipt or receipt.get("revision") != request.router_revision or receipt.get("sha256") != actual:
            raise ValueError("router binary has no matching source revision/hash receipt")
        subprocess.run([str(executable), "--help"], capture_output=True, timeout=15, check=True)
        evidence["router"] = receipt
    if spec.gpus:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=15, check=True)
        available = {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
        if not set(spec.gpus) <= available:
            raise ValueError("assigned GPU is not available on this host")
        evidence["gpus"] = spec.gpus
    return evidence


def status(request, root):
    record = read_json(role_directory(request.name, request.role_id, root) / "state.json")
    if record is None:
        return {"phase": "ABSENT"}
    record["supervisor_alive"] = owned_process(record.get("supervisor")) is not None
    record["worker_alive"] = owned_process(record.get("worker")) is not None
    if record["phase"] == "RUNNING" and not (record["supervisor_alive"] and record["worker_alive"]):
        record["phase"] = "DEGRADED"
    return record


def start(request, root):
    directory = role_directory(request.name, request.role_id, root)
    record_path = directory / "state.json"
    with file_lock(Path(root) / "host.lock"):
        stopped = read_json(directory / "stop.json")
        if request.owner and stopped and stopped.get("owner") == request.owner:
            raise RuntimeError("this start attempt was already cancelled")
        existing = read_json(record_path)
        if existing and existing["phase"] not in ("STOPPED", "FAILED"):
            if existing["config_hash"] != request.config_hash or existing["generation"] != request.generation:
                raise RuntimeError("role belongs to another config/generation; stop it before starting")
            if request.owner and request.owner != existing["owner"]:
                raise RuntimeError("role belongs to another controller attempt")
            return status(request, root)
        if existing and owned_process(existing.get("worker")) is not None:
            raise RuntimeError("prior worker is still alive; stop its recorded owner first")
        require_resources_free(request, root)
        require_ports_free(request.spec)
        private_dir(directory)
        # A controller supplies its nonce before RPC so a lost reply remains recoverable.
        owner = request.owner or uuid.uuid4().hex
        record = {"phase": "STARTING", "owner": owner, "name": request.name,
                  "config_hash": request.config_hash, "generation": request.generation,
                  "spec": request.spec.model_dump(), "started_at": time.time()}
        atomic_json(record_path, record)
        environment = {**os.environ, OWNER_ENV: owner, "NANO_XPU_STATE_DIR": str(Path(root).resolve())}
        with (directory / "supervisor.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, "-m", "nanoxpu.deployment.supervisor"],
                                       env=environment, stdin=subprocess.PIPE, stdout=log, stderr=log,
                                       **detached_options())
        try:
            record["supervisor"] = process_identity(process.pid, owner)
            atomic_json(record_path, record)
            payload = {"name": request.name, "owner": owner, "spec": request.spec.model_dump(),
                       "auth_token": request.auth_token.get_secret_value() if request.auth_token else "",
                       "drain_seconds": request.drain_seconds}
            process.stdin.write(json.dumps(payload).encode())
            process.stdin.close()
        except BaseException:
            process.stdin.close()
            process.terminate()
            process.wait(timeout=5)
            record["phase"] = "FAILED"
            atomic_json(record_path, record)
            raise
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = status(request, root)
        if result["phase"] != "STARTING":
            if result["phase"] != "RUNNING":
                raise RuntimeError("worker failed to start; inspect its retained supervisor/worker logs")
            return result
        time.sleep(0.05)
    raise TimeoutError("supervisor bootstrap timed out; role remains recorded for cleanup")


def stop(request, root):
    directory = role_directory(request.name, request.role_id, root)
    path = directory / "state.json"
    with file_lock(Path(root) / "host.lock"), file_lock(directory / "stop.lock"):
        record = read_json(path)
        if record is None:
            # Fence a delayed start RPC with the same nonce, even if no process exists yet.
            atomic_json(directory / "stop.json", {"owner": request.owner})
            return {"phase": "ABSENT"}
        if record["owner"] != request.owner:
            raise RuntimeError("stop ownership differs from the recorded run")
        supervisor = owned_process(record.get("supervisor"))
        if record["phase"] != "STOPPED" and supervisor is not None:
            atomic_json(directory / "stop.json", {"owner": request.owner})
            deadline = time.monotonic() + request.drain_seconds * 2 + 8
            while time.monotonic() < deadline:
                record = read_json(path)
                if record["phase"] in ("STOPPED", "FAILED", "CLEANUP_FAILED"):
                    break
                if owned_process(record.get("supervisor")) is None:
                    break
                time.sleep(0.05)
        # Recover from a dead/stuck supervisor only after verifying every recorded identity.
        if owned_process(record.get("worker")) is not None:
            stop_owned_tree(record["worker"], request.drain_seconds)
        if owned_process(record.get("supervisor")) is not None and record["phase"] != "STOPPED":
            stop_owned_tree(record["supervisor"], 1)
        if record["phase"] == "STARTING" and not record.get("worker"):
            raise RuntimeError("bootstrap was interrupted without a worker identity; cleanup cannot be proven")
        record["phase"] = "STOPPED"
        atomic_json(path, record)
        return record


def dispatch(request: HelperRequest, root=None):
    root = Path(root or state_root())
    if request.action == "preflight":
        return preflight(request, root)
    if request.action == "start":
        return start(request, root)
    if request.action == "status":
        return status(request, root)
    if request.action == "stop":
        return stop(request, root)
    path = role_directory(request.name, request.role_id, root) / "worker.log"
    if not path.is_file():
        return {"text": ""}
    with path.open("rb") as log:
        log.seek(max(0, path.stat().st_size - request.log_bytes))
        return {"text": log.read(request.log_bytes).decode("utf-8", errors="replace")}


def main():
    try:
        data = sys.stdin.buffer.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("helper request exceeds 1 MiB")
        request = HelperRequest.model_validate_json(data)
        result = dispatch(request)
        print(json.dumps({"ok": True, "result": result}))
    except ValidationError as exc:
        print(json.dumps({"ok": False, "error": [{"field": item["loc"], "type": item["type"]} for item in exc.errors(include_input=False)]}))
        raise SystemExit(1)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
