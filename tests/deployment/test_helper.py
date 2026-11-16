from tests.helpers.process import free_port, helper
import json
import hashlib
from pathlib import Path
import socket
import subprocess
import sys
import time

import psutil
import pytest

from nanoxpu.deployment.helper import HelperRequest, dispatch
from nanoxpu.deployment.state import atomic_json, owned_process, read_json
from nanoxpu.deployment.ssh import SSHExecutor
from nanoxpu.deployment.schema import load_config


ROOT = Path(__file__).parents[2]




def test_router_receipt_rejects_modified_binary_without_python311_hash_api(tmp_path, request_data, monkeypatch):
    executable = tmp_path / "router-fixture"
    executable.write_bytes(b"synthetic router receipt input")
    executable.chmod(0o700)
    revision = "b" * 40
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    Path(str(executable) + ".source.json").write_text(json.dumps({"revision": revision, "sha256": digest}))
    payload = {**request_data, "action": "preflight", "router_revision": revision,
               "spec": {**request_data["spec"], "role": "router", "argv": [str(executable)]}}
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    calls = []
    # This test checks the receipt, not a router build. Actual --help is a Linux gate.
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append(args))
    request = HelperRequest.model_validate(payload)
    assert dispatch(request, tmp_path / "state")["router"]["sha256"] == digest
    executable.write_bytes(b"tampered binary")
    with pytest.raises(ValueError, match="no matching source revision/hash receipt"):
        dispatch(request, tmp_path / "state")
    assert calls == [[str(executable), "--help"]]




@pytest.fixture
def request_data(tmp_path):
    return {"action": "start", "name": "fixture", "role_id": "worker",
            "config_hash": "a" * 64, "generation": "gen1", "drain_seconds": 0.3,
            "auth_token": "internal-secret-must-never-appear-in-ledger-or-logs",
            "spec": {"id": "worker", "role": "colocated", "ip": "127.0.0.1", "advertise_ip": "127.0.0.1",
                     "argv": [sys.executable, str(ROOT / "tests/helpers/process_fixture.py")], "env": {},
                     "workdir": str(tmp_path), "http_port": free_port(), "distributed_port": None,
                     "kv_port": None, "gpus": [], "instance_id": "fixture-worker"}}


def stop(root, owner):
    return helper(root, {"action": "stop", "name": "fixture", "role_id": "worker", "owner": owner, "drain_seconds": 0.3})


def wait_for_log(root, text):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = helper(root, {"action": "logs", "name": "fixture", "role_id": "worker"})["text"]
        if text in result:
            return result
        time.sleep(0.05)
    pytest.fail(f"worker log did not contain {text}")


def test_detached_lifecycle_repeated_start_stop_and_secret_redaction(tmp_path, request_data):
    root = tmp_path / "state"
    record = helper(root, request_data)
    try:
        assert record["worker_alive"] and record["supervisor_alive"]
        # The start helper has already exited; its independently supervised child remains alive.
        again = helper(root, request_data)
        assert again["worker"] == record["worker"]
        assert "[REDACTED]" in wait_for_log(root, "READY")
        for path in root.rglob("*"):
            if path.is_file():
                assert request_data["auth_token"] not in path.read_text(encoding="utf-8", errors="replace")
        changed = {**request_data, "config_hash": "b" * 64}
        assert not helper(root, changed, check=False)["ok"]
        assert stop(root, record["owner"])["phase"] == "STOPPED"
        assert stop(root, record["owner"])["phase"] == "STOPPED"
        assert owned_process(record["worker"]) is None
    finally:
        stop(root, record["owner"])


def test_stop_rejects_reused_pid_identity(tmp_path, request_data):
    root = tmp_path / "state"
    record = helper(root, request_data)
    path = root / "fixture/worker/state.json"
    try:
        forged = read_json(path)
        forged["supervisor"]["created_at"] -= 100
        atomic_json(path, forged)
        response = helper(root, {"action": "stop", "name": "fixture", "role_id": "worker",
                                 "owner": record["owner"], "drain_seconds": 0.3}, check=False)
        assert not response["ok"]
        assert "identity changed" in response["error"]
        assert owned_process(record["worker"]) is not None
    finally:
        atomic_json(path, record)
        stop(root, record["owner"])


def test_stop_wrong_owner_does_not_kill_worker(tmp_path, request_data):
    root = tmp_path / "state"
    record = helper(root, request_data)
    try:
        response = helper(root, {"action": "stop", "name": "fixture", "role_id": "worker", "owner": "wrong"}, check=False)
        assert not response["ok"]
        assert owned_process(record["worker"]) is not None
    finally:
        stop(root, record["owner"])


def test_host_reservations_prevent_overlapping_deployments(tmp_path, request_data):
    root = tmp_path / "state"
    record = helper(root, request_data)
    try:
        second = {**request_data, "name": "second"}
        response = helper(root, second, check=False)
        assert not response["ok"]
        assert "reservation overlaps" in response["error"]
    finally:
        stop(root, record["owner"])


def test_supervisor_stops_owned_descendants(tmp_path, request_data):
    request_data["spec"]["argv"] += ["--mode", "spawn", "--ignore-term"]
    root = tmp_path / "state"
    record = helper(root, request_data)
    try:
        log = wait_for_log(root, "CHILD_PID=")
        child_pid = int(log.split("CHILD_PID=", 1)[1].split()[0])
        assert psutil.pid_exists(child_pid)
        stop(root, record["owner"])
        assert owned_process(record["worker"]) is None
        if psutil.pid_exists(child_pid):
            assert psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    finally:
        stop(root, record["owner"])


def test_failed_worker_retains_logs_and_can_be_cleaned(tmp_path, request_data):
    request_data["spec"]["argv"] += ["--mode", "fail"]
    root = tmp_path / "state"
    result = helper(root, request_data, check=False)
    path = root / "fixture/worker/state.json"
    record = read_json(path)
    assert record is not None
    try:
        wait_for_log(root, "injected worker startup failure")
        assert stop(root, record["owner"])["phase"] == "STOPPED"
        assert owned_process(record.get("worker")) is None
    finally:
        stop(root, record["owner"])


def test_preflight_rejects_busy_port_without_launching(tmp_path, request_data):
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", 0))
        sock.listen()
        request_data["spec"]["http_port"] = sock.getsockname()[1]
        request_data["action"] = "preflight"
        with pytest.raises(OSError):
            dispatch(HelperRequest.model_validate(request_data), tmp_path / "state")
    assert not list(tmp_path.rglob("state.json"))


def test_cancelled_start_nonce_cannot_launch_later(tmp_path, request_data):
    root = tmp_path / "state"
    request_data["owner"] = "cancelled-attempt"
    assert stop(root, request_data["owner"])["phase"] == "ABSENT"
    response = helper(root, request_data, check=False)
    assert not response["ok"] and "already cancelled" in response["error"]
    assert not list(root.rglob("state.json"))
    request_data["owner"] = "fresh-attempt"
    record = helper(root, request_data)
    assert record["worker_alive"]
    stop(root, record["owner"])


def test_preflight_reads_real_model_bytes_without_starting(tmp_path, request_data):
    from tests.helpers.support import write_model_fixture
    from nanoxpu.server.metadata import model_fingerprints
    directory = write_model_fixture(tmp_path / "model")
    request_data.update(action="preflight", model_path=str(directory))
    result = helper(tmp_path / "state", request_data)
    assert result["model"] == model_fingerprints(directory)
    assert not list(tmp_path.rglob("state.json"))


def test_ssh_command_never_interpolates_configuration(example_files):
    _, profile = load_config(example_files / "deployment/cluster-colocated.yaml")
    command = SSHExecutor(profile).command("10.0.0.11")
    assert "BatchMode=yes" in command
    assert "StrictHostKeyChecking=yes" in command
    assert command[-3:] == [profile.runtime.helper_python, "-m", "nanoxpu.deployment.helper"]
    assert profile.model_path not in command
    with pytest.raises(ValueError):
        SSHExecutor(profile).command("host; touch /tmp/nope")


def test_ssh_custom_known_hosts_preserves_strict_verification(example_files):
    _, profile = load_config(example_files / "deployment/cluster-colocated.yaml")
    ssh = profile.ssh.model_copy(update={"known_hosts_file": "/tmp/host keys/known_hosts"})
    command = SSHExecutor(profile.model_copy(update={"ssh": ssh})).command("127.0.0.1")
    assert 'UserKnownHostsFile="/tmp/host keys/known_hosts"' in command
    assert "StrictHostKeyChecking=yes" in command
    assert command[-3:] == [profile.runtime.helper_python, "-m", "nanoxpu.deployment.helper"]


@pytest.mark.parametrize("field", ["identity_file", "known_hosts_file"])
@pytest.mark.parametrize("value", ["", "host\nkeys", "host\rkeys", "host\x00keys"])
def test_ssh_rejects_invalid_local_file_paths(field, value, example_files):
    _, profile = load_config(example_files / "deployment/cluster-colocated.yaml")
    data = profile.model_dump()
    data["ssh"][field] = value
    with pytest.raises(ValueError, match="SSH file path"):
        type(profile).model_validate(data)
