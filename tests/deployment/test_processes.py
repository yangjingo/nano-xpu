"""Real helper/supervisor/HTTP-worker integration; router probes are explicitly simulated.

These tests are lifecycle evidence only. Linux real-router tests must separately prove forwarding.
"""
from dataclasses import replace
import sys

import httpx

from nanoxpu.deployment.controller import Controller
from nanoxpu.deployment.planner import LaunchSpec, plan_file
from nanoxpu.deployment.state import owned_process, read_json
from tests.helpers.support import write_model_fixture
from tests.helpers.process import ROOT, free_port, helper
from tests.helpers.http import payload
from tests.helpers.checkout_fixture import committed_checkout


class LocalHelper:
    def __init__(self, root):
        self.root = root
        self.starts = []

    def request(self, ip, request, timeout):
        assert ip == "127.0.0.1"
        if request["action"] == "preflight" and request["role_id"] == "router":
            # This fixture is an inert process, not a vllm-router binary.
            request = {key: value for key, value in request.items() if key != "router_revision"}
        response = helper(self.root, request, cwd=request.get("spec", {}).get("workdir"))
        if request["action"] == "start":
            self.starts.append(response)
        return response


class RouterProbeFixture:
    def __init__(self, plan):
        self.plan = plan
        self.client = httpx.Client(timeout=2, trust_env=False)

    def get(self, url, **kwargs):
        if url.startswith(self.plan.router.url + "/"):
            workers = [{"url": spec.url, "worker_type": "regular", "is_healthy": True} for spec in self.plan.workers]
            return httpx.Response(200, json={"workers": workers})
        return self.client.get(url, **kwargs)


def test_controller_manages_real_cpu_worker_after_helpers_exit(tmp_path, example_files):
    model = write_model_fixture(tmp_path / "model")
    checkout, revision = committed_checkout(tmp_path / "checkout")
    source = plan_file(example_files / "deployment/cluster-colocated.yaml")
    ports = set()
    while len(ports) != 3:
        ports.add(free_port())
    worker_port, distributed_port, router_port = ports
    worker = LaunchSpec("worker", "colocated", "127.0.0.1", "127.0.0.1",
                        [sys.executable, "-m", "tests.helpers.cpu_server", "_worker", str(model), "--served-model-name", "qwen-test",
                         "--port", str(worker_port), "--distributed-port", str(distributed_port), "--instance-id", "cpu-worker"],
                        {}, str(checkout), worker_port, distributed_port, None, [], "cpu-worker")
    router = LaunchSpec("router", "router", "127.0.0.1", "127.0.0.1",
                        [sys.executable, str(ROOT / "tests/helpers/process_fixture.py")], {}, str(ROOT), router_port,
                        None, None, [], "fixture-router")
    runtime = source.profile.runtime.model_copy(update={"expected_nano_xpu_revision": revision})
    profile = source.profile.model_copy(update={"runtime": runtime, "served_model_name": "qwen-test",
                                               "timeouts": source.profile.timeouts.model_copy(update={"drain_seconds": 1, "startup_seconds": 15})})
    plan = replace(source, name="cpu-processes", workers=[worker], router=router, profile=profile, unresolved=[])
    executor = LocalHelper(tmp_path / "remote-state")
    probes = RouterProbeFixture(plan)
    controller = Controller(plan, root=tmp_path / "local-state", executor=executor, http=probes)
    try:
        assert controller.up()["ready"]
        assert len(executor.starts) == 2
        # All start helpers have exited. HTTP is still served by the detached worker.
        response = probes.client.post(worker.url + "/v1/chat/completions", json=payload())
        assert response.status_code == 200
        assert "CPU TEST ENGINE" in response.text
        assert controller.up()["ready"]
        assert len(executor.starts) == 2
        assert "CPU TEST SERVER" in controller.logs("worker")["worker"]
        assert controller.down()["phase"] == "STOPPED"
        assert all(owned_process(record["worker"]) is None for record in executor.starts)
        assert read_json(controller.path)["phase"] == "STOPPED"
    finally:
        controller.down()
        probes.client.close()
