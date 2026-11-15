"""Real localhost SSH, detached processes, HTTP and pinned router in one control path.

The worker uses the explicit CPU test engine; this is deployment evidence, not CUDA.
Keys and helper state live outside the uploaded pytest artifacts.
"""
from dataclasses import replace
import getpass
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import psutil
import pytest

from nanoxpu.deployment.controller import Controller, DeploymentError
from nanoxpu.deployment.planner import LaunchSpec, plan_file
from nanoxpu.deployment.ssh import SSHExecutor
from nanoxpu.deployment.state import owned_process, read_json
from tests.helpers.checkout_fixture import committed_checkout
from tests.helpers.support import write_model_fixture
from tests.helpers.process import ROOT, free_port
from tests.helpers.http import payload


@pytest.fixture
def ssh_host(tmp_path, example_files):
    if sys.platform != "linux":
        pytest.fail("real SSH integration requires Linux")
    sshd = shutil.which("sshd") or "/usr/sbin/sshd"
    assert Path(sshd).is_file(), "install openssh-server"
    # Root is needed only for sshd's normal user authentication/privilege separation.
    privilege = [] if os.geteuid() == 0 else ["sudo", "-n"]
    subprocess.run([*privilege, "mkdir", "-p", "/run/sshd"], check=True, timeout=10)
    with tempfile.TemporaryDirectory(prefix="nano-xpu-ssh-") as directory:
        root = Path(directory)
        root.chmod(0o700)
        for name in ("host", "client"):
            subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(root / name)], check=True, timeout=10)
        authorized = root / "authorized_keys"
        shutil.copyfile(root / "client.pub", authorized)
        authorized.chmod(0o600)
        port = free_port()
        public = (root / "host.pub").read_text().split()
        known_hosts = root / "known_hosts"
        known_hosts.write_text(f"[127.0.0.1]:{port} {public[0]} {public[1]}\n")
        config = root / "sshd_config"
        config.write_text(f"""ListenAddress 127.0.0.1
Port {port}
HostKey {root / 'host'}
PidFile {root / 'sshd.pid'}
AuthorizedKeysFile {authorized}
AllowUsers {getpass.getuser()}
AuthenticationMethods publickey
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM yes
StrictModes yes
PermitRootLogin prohibit-password
LogLevel VERBOSE
""")
        subprocess.run([*privilege, sshd, "-t", "-f", str(config)], check=True, timeout=10)
        log = tmp_path / "sshd.log"
        daemon = None
        with log.open("w") as stream:
            launcher = subprocess.Popen([*privilege, sshd, "-D", "-e", "-f", str(config)],
                                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                assert launcher.poll() is None, log.read_text()
                pid_file = root / "sshd.pid"
                if pid_file.exists():
                    candidate = psutil.Process(int(pid_file.read_text()))
                    if any(str(config) in arg for arg in candidate.cmdline()):
                        daemon = candidate
                        try:
                            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                                break
                        except OSError:
                            pass
                time.sleep(0.1)
            else:
                pytest.fail(log.read_text())
            checkout, revision = committed_checkout(root / "checkout")
            wrapper = root / "helper-python"
            # Both actual helper and worker must import the same clean checkout.
            wrapper.write_text("#!/bin/sh\n"
                               f"export NANO_XPU_STATE_DIR={shlex.quote(str(root / 'remote-state'))}\n"
                               f"export LD_LIBRARY_PATH={shlex.quote(os.environ.get('LD_LIBRARY_PATH', ''))}\n"
                               f"cd {shlex.quote(str(checkout))} || exit 1\n"
                               f"exec {shlex.quote(sys.executable)} \"$@\"\n")
            wrapper.chmod(0o700)
            source = plan_file(example_files / "deployment/cluster-colocated.yaml")
            ssh = source.profile.ssh.model_copy(update={"user": getpass.getuser(), "port": port,
                                                       "identity_file": str(root / "client"), "known_hosts_file": str(known_hosts)})
            runtime = source.profile.runtime.model_copy(update={"helper_python": str(wrapper), "workdir": str(checkout),
                                                               "expected_nano_xpu_revision": revision})
            profile = source.profile.model_copy(update={"ssh": ssh, "runtime": runtime})
            yield root, source, profile
        finally:
            # Check the original psutil identity (PID plus creation time) before sudo kill.
            if daemon is not None and daemon.is_running() and any(str(config) in arg for arg in daemon.cmdline()):
                subprocess.run([*privilege, "kill", "-TERM", str(daemon.pid)], check=True, timeout=10)
            launcher.wait(timeout=15)


def test_ssh_rejects_untrusted_host_and_accepts_fixed_helper(ssh_host):
    root, _, profile = ssh_host
    executor = SSHExecutor(profile)
    request = {"action": "status", "name": "absent", "role_id": "worker"}
    assert executor.request("127.0.0.1", request)["phase"] == "ABSENT"
    missing = root / "empty_known_hosts"
    missing.write_text("")
    ssh = profile.ssh.model_copy(update={"known_hosts_file": str(missing)})
    with pytest.raises(RuntimeError, match="no valid response"):
        SSHExecutor(profile.model_copy(update={"ssh": ssh})).request("127.0.0.1", request)


@pytest.mark.parametrize("failure", [None, "preflight", "worker", "router"])
def test_controller_ssh_router_and_cli_recovery(tmp_path, ssh_host, router_binary, failure):
    root, source, profile = ssh_host
    checkout = Path(profile.runtime.workdir)
    model = write_model_fixture(root / "model")
    ports = set()
    while len(ports) < 3:
        ports.add(free_port())
    worker_port, distributed_port, router_port = ports
    worker = LaunchSpec("worker", "colocated", "127.0.0.1", "127.0.0.1",
                        [sys.executable, "-m", "tests.helpers.cpu_server", "_worker", str(model), "--served-model-name", "qwen-test",
                         "--port", str(worker_port), "--distributed-port", str(distributed_port), "--instance-id", "cpu-worker"],
                        {}, str(checkout), worker_port, distributed_port, None, [], "cpu-worker")
    router = LaunchSpec("router", "router", "127.0.0.1", "127.0.0.1",
                        [str(router_binary), "--host", "127.0.0.1", "--port", str(router_port), "--policy", "round_robin",
                         "--worker-urls", worker.url, "--health-check-endpoint", "/ready", "--disable-retries",
                         "--worker-startup-timeout-secs", "30", "--request-timeout-secs", "10"],
                        {}, str(checkout), router_port, None, None, [], "cpu-router")
    profile = profile.model_copy(update={"model_path": str(model), "served_model_name": "qwen-test",
                                         "router": profile.router.model_copy(update={"executable": str(router_binary), "port": router_port}),
                                         "timeouts": profile.timeouts.model_copy(update={"drain_seconds": 1, "startup_seconds": 45})})
    if failure == "worker":
        worker = replace(worker, argv=[*worker.argv, "--invalid-startup-option"])
    elif failure == "router":
        router = replace(router, argv=[*router.argv, "--invalid-startup-option"])
    plan = replace(source, name="ssh-integration", profile=profile, workers=[worker], router=router, unresolved=[])
    state = root / "controller-state"
    controller = Controller(plan, root=state)
    remote = []

    def cli(action):
        result = subprocess.run([sys.executable, "-m", "nanoxpu", "serve", action, "--name", plan.name,
                                 "--state-dir", str(state)], cwd=ROOT, text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    try:
        if failure == "preflight":
            with socket.socket() as occupied:
                occupied.bind(("127.0.0.1", worker_port))
                occupied.listen()
                with pytest.raises(RuntimeError):
                    controller.up()
            assert not controller.path.exists()
            assert all(controller.rpc(spec, "status")["phase"] == "ABSENT" for spec in [worker, router])
            return
        if failure:
            with pytest.raises((DeploymentError, RuntimeError)):
                controller.up()
            record = read_json(controller.path)
            assert record["phase"] == "STOPPED"
            assert all(entry["phase"] == "STOPPED" for entry in record["roles"])
            for entry in record["roles"]:
                stopped = controller.rpc(LaunchSpec(**entry["spec"]), "status")
                assert stopped["phase"] == "STOPPED"
                assert owned_process(stopped.get("worker")) is None
            return
        assert controller.up()["ready"]
        remote = [controller.rpc(spec, "status") for spec in [worker, router]]
        with httpx.Client(timeout=10, trust_env=False) as client:
            result = client.post(router.url + "/v1/chat/completions", json=payload())
            assert result.status_code == 200, result.text
            assert "CPU TEST ENGINE" in result.text
        assert controller.up()["ready"]
        assert [controller.rpc(spec, "status")["worker"] for spec in [worker, router]] == [item["worker"] for item in remote]
        controller.close()
        # A fresh CLI process reconstructs the deployment solely from the ledger.
        assert cli("status")["ready"]
        logs = cli("logs")
        assert "CPU TEST SERVER" in logs["worker"]
        for role, text in logs.items():
            (tmp_path / f"deployed-{role}.log").write_text(text)
        assert cli("down")["phase"] == "STOPPED"
        assert cli("down")["phase"] == "STOPPED"
        assert all(owned_process(item["worker"]) is None for item in remote)
        assert read_json(controller.path)["phase"] == "STOPPED"
    finally:
        with_cleanup = Controller.load(plan.name, root=state) if controller.path.exists() else None
        if with_cleanup is not None:
            try:
                with_cleanup.down()
            finally:
                with_cleanup.close()
        controller.close()


@pytest.mark.parametrize("compatible", [False, True])
def test_pd_controller_checks_capabilities_and_stops_in_order(tmp_path, ssh_host, router_binary, compatible):
    root, source, profile = ssh_host
    model = write_model_fixture(root / "model")
    ports = set()

    def port():
        value = free_port()
        while value in ports:
            value = free_port()
        ports.add(value)
        return value

    workers = []
    for role in ("prefill", "decode"):
        http_port, distributed_port, kv_port = port(), port(), port()
        workers.append(LaunchSpec(role, role, "127.0.0.1", "127.0.0.1",
                                 [sys.executable, "-m", "tests.helpers.router_worker", "_worker", str(model),
                                  "--served-model-name", "cpu-contract", "--port", str(http_port),
                                  "--distributed-port", str(distributed_port), "--kv-port", str(kv_port),
                                  "--kv-connector", "nixl-pull", "--role", role, "--instance-id", role,
                                  "--advertise-ip", "127.0.0.1"],
                                 {}, profile.runtime.workdir, http_port, distributed_port, kv_port, [], role))
    router_port = port()
    router = LaunchSpec("router", "router", "127.0.0.1", "127.0.0.1",
                        [str(router_binary), "--host", "127.0.0.1", "--port", str(router_port), "--policy", "round_robin",
                         "--vllm-pd-disaggregation", "--kv-connector", "nixl", "--prefill", workers[0].url,
                         "--decode", workers[1].url, "--health-check-endpoint", "/ready", "--disable-retries",
                         "--worker-startup-timeout-secs", "30", "--request-timeout-secs", "10"],
                        {}, profile.runtime.workdir, router_port, None, None, [], "router")
    if compatible:
        # Explicitly test-only profile injection, like the CPU argv/gpus above.
        # Production YAML rejects cpu-test; workers never pretend their transport is NIXL.
        profile = profile.model_copy(update={"kv": profile.kv.model_copy(update={"connector": "cpu-test", "expected_version": "cpu-contract-v1"})})
        with pytest.raises(ValueError):
            type(profile).model_validate(profile.model_dump())
    profile = profile.model_copy(update={"model_path": str(model), "served_model_name": "cpu-contract",
                                         "timeouts": profile.timeouts.model_copy(update={"startup_seconds": 45, "drain_seconds": 1})})
    plan = replace(source, name="ssh-pd", mode="disaggregated", workers=workers, router=router, profile=profile, unresolved=[])
    executor = SSHExecutor(profile)
    actions = []

    class ObservedSSH:
        def request(self, ip, request, timeout):
            actions.append((request["action"], request["role_id"]))
            return executor.request(ip, request, timeout)

    controller = Controller(plan, root=root / "pd-state", executor=ObservedSSH())
    remote = []
    try:
        if not compatible:
            with pytest.raises(DeploymentError, match="incompatible KV capabilities"):
                controller.up()
            assert [role for action, role in actions if action == "start"] == ["prefill"]
            assert read_json(controller.path)["phase"] == "STOPPED"
            return
        assert controller.up()["ready"]
        assert [role for action, role in actions if action == "start"] == ["prefill", "decode", "router"]
        remote = [controller.rpc(spec, "status") for spec in [*workers, router]]
        with httpx.Client(timeout=10, trust_env=False) as client:
            response = client.post(router.url + "/v1/chat/completions", json={
                "model": "cpu-contract", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 5})
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == "CPU CONTRACT decode"
            assert response.json()["usage"]["completion_tokens"] == 5
        assert controller.up()["ready"]
        for role, log in controller.logs().items():
            (tmp_path / f"pd-{role}.log").write_text(log)
        actions.clear()
        assert controller.down()["phase"] == "STOPPED"
        assert [role for action, role in actions if action == "stop"] == ["router", "decode", "prefill"]
        assert all(owned_process(item["worker"]) is None for item in remote)
    finally:
        try:
            if controller.path.exists():
                controller.down()
        finally:
            controller.close()
