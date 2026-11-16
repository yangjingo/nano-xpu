"""Fault-injected orchestration tests; the injected RPC/HTTP are not router integration evidence."""
from copy import deepcopy
from dataclasses import replace

import httpx
import pytest

from nanoxpu.deployment.controller import Controller, DeploymentError
from nanoxpu.deployment.planner import plan_file
from nanoxpu.deployment.state import read_json
from tests.helpers.examples import materialize_examples

FINGERPRINTS = dict.fromkeys(("model", "tokenizer", "template", "rope"), "a" * 64)


class ControlFixture:
    def __init__(self, plan):
        self.plan = plan
        self.calls = []
        self.records = {}
        self.fail_preflight = None
        self.fail_start = None
        self.drop_reply = None
        self.unreachable = None
        self.bad_info = None
        self.router_healthy = True

    def request(self, ip, payload, timeout):
        self.calls.append(deepcopy(payload))
        role_id, action = payload["role_id"], payload["action"]
        if self.unreachable == role_id:
            raise OSError("host unavailable")
        if action == "preflight":
            if role_id == self.fail_preflight:
                raise RuntimeError("injected preflight failure")
            return {"model": FINGERPRINTS}
        if action == "start":
            if role_id == self.fail_start:
                raise RuntimeError("injected startup failure")
            self.records[role_id] = {**payload, "phase": "RUNNING"}
            if role_id == self.drop_reply:
                raise OSError("reply was lost after process startup")
            return self.records[role_id]
        if action == "stop":
            record = self.records.get(role_id)
            if record is None:
                return {"phase": "ABSENT"}
            assert record["owner"] == payload["owner"]
            record["phase"] = "STOPPED"
            return record
        if action == "logs":
            return {"text": "fixture output"}
        return self.records.get(role_id, {"phase": "ABSENT"})

    def http(self, request):
        spec = next(spec for spec in [*self.plan.workers, self.plan.router] if request.url.port == spec.http_port)
        if request.url.path == "/workers":
            workers = [{"url": worker.url, "worker_type": "regular" if worker.role == "colocated" else worker.role,
                        "is_healthy": self.router_healthy} for worker in self.plan.workers]
            return httpx.Response(200, json={"workers": workers})
        if request.url.path == "/server_info":
            record = self.records[spec.id]
            assert request.headers["authorization"] == "Bearer " + record["auth_token"]
            info = {"instance_id": spec.instance_id, "engine_generation": record["generation"],
                    "cluster_id": self.plan.cluster_id, "role": spec.role, "model": self.plan.profile.served_model_name,
                    "tensor_parallel_size": 1,
                    "capabilities": {"fingerprints": FINGERPRINTS, "protocol_version": 1,
                                     "transport": "nixl-pull" if spec.role != "colocated" else "none",
                                     "transport_version": self.plan.profile.kv.expected_version,
                                     "connector_ready": True, "layout_digest": "b" * 64}}
            if spec.id == self.bad_info:
                info["engine_generation"] = "stale"
            return httpx.Response(200, json=info)
        return httpx.Response(200, json={"status": "ready"})


def setup_controller(tmp_path, pd=False):
    example_files = materialize_examples(tmp_path / "examples")
    plan = plan_file(example_files / f"deployment/cluster-{'pd' if pd else 'colocated'}.yaml")
    runtime = plan.profile.runtime.model_copy(update={"expected_nano_xpu_revision": "a" * 40})
    profile = plan.profile.model_copy(update={"runtime": runtime,
                                            "timeouts": plan.profile.timeouts.model_copy(update={"startup_seconds": 1}),
                                            "kv": plan.profile.kv.model_copy(update={"expected_version": "pinned-test"})})
    plan = replace(plan, profile=profile, unresolved=[])
    fixture = ControlFixture(plan)
    http = httpx.Client(transport=httpx.MockTransport(fixture.http))
    return Controller(plan, root=tmp_path, executor=fixture, http=http), fixture


def test_all_preflight_precedes_start_router_last_and_repeat_up_reuses(tmp_path):
    controller, fixture = setup_controller(tmp_path)
    result = controller.up()
    assert result["ready"] and result["router_url"] == controller.plan.router.url
    starts = [call for call in fixture.calls if call["action"] == "start"]
    assert starts[-1]["role_id"] == "router"
    first_start = next(index for index, call in enumerate(fixture.calls) if call["action"] == "start")
    assert all(call["action"] == "preflight" for call in fixture.calls[:first_start])
    assert len(starts) == len(controller.plan.workers) + 1
    for call in starts:
        assert call["auth_token"] not in controller.path.read_text()
    assert controller.up()["ready"]
    assert len([call for call in fixture.calls if call["action"] == "start"]) == len(starts)
    reloaded = Controller.load(controller.plan.name, root=tmp_path, executor=fixture, http=controller.http)
    assert reloaded.status()["ready"]
    assert reloaded.logs("colocated")
    assert reloaded.down()["phase"] == "STOPPED"
    assert reloaded.down()["phase"] == "STOPPED"


def test_preflight_failure_starts_zero_workers(tmp_path):
    controller, fixture = setup_controller(tmp_path)
    fixture.fail_preflight = "router"
    with pytest.raises(RuntimeError, match="preflight"):
        controller.up()
    assert not fixture.records
    assert not controller.path.exists()


@pytest.mark.parametrize("lost_reply", [False, True])
def test_start_failure_or_lost_reply_rolls_back_owned_attempts(tmp_path, lost_reply):
    controller, fixture = setup_controller(tmp_path)
    if lost_reply:
        fixture.drop_reply = "router"
    else:
        fixture.fail_start = "router"
    with pytest.raises((OSError, RuntimeError)):
        controller.up()
    assert read_json(controller.path)["phase"] == "STOPPED"
    assert all(record["phase"] == "STOPPED" for record in fixture.records.values())
    stops = [call["role_id"] for call in fixture.calls if call["action"] == "stop"]
    assert stops[0] == "router"


def test_mismatched_worker_identity_never_starts_router(tmp_path):
    controller, fixture = setup_controller(tmp_path)
    fixture.bad_info = controller.plan.workers[0].id
    with pytest.raises(DeploymentError, match="generation"):
        controller.up()
    assert "router" not in fixture.records
    assert read_json(controller.path)["phase"] == "STOPPED"


def test_router_unhealthy_is_not_a_successful_up(tmp_path):
    controller, fixture = setup_controller(tmp_path)
    fixture.router_healthy = False
    with pytest.raises(DeploymentError, match="timed out"):
        controller.up()
    assert read_json(controller.path)["phase"] == "STOPPED"


def test_unreachable_down_preserves_resources_and_can_retry(tmp_path):
    controller, fixture = setup_controller(tmp_path)
    controller.up()
    fixture.unreachable = controller.plan.workers[0].id
    assert controller.status()["phase"] == "DEGRADED"
    with pytest.raises(DeploymentError, match="ledger retained"):
        controller.down()
    assert fixture.records["router"]["phase"] == "STOPPED"
    assert read_json(controller.path)["phase"] == "CLEANUP_FAILED"
    with pytest.raises(DeploymentError, match="incomplete/degraded"):
        controller.up()
    fixture.unreachable = None
    assert controller.down()["phase"] == "STOPPED"


def test_changed_config_cannot_replace_a_live_cluster(tmp_path):
    controller, fixture = setup_controller(tmp_path)
    controller.up()
    changed = Controller(replace(controller.plan, config_hash="c" * 64), root=tmp_path, executor=fixture, http=controller.http)
    with pytest.raises(DeploymentError, match="configuration changed"):
        changed.up()
    assert controller.status()["ready"]
    controller.down()


def test_pd_shutdown_order_and_generations_in_worker_args(tmp_path):
    controller, fixture = setup_controller(tmp_path, pd=True)
    assert controller.up()["ready"]
    starts = [call for call in fixture.calls if call["action"] == "start"]
    for call in starts:
        if call["spec"]["role"] == "decode":
            assert "--prefill-peer" in call["spec"]["argv"]
            assert "--engine-generation" in call["spec"]["argv"]
    controller.down()
    stops = [call["role_id"] for call in fixture.calls if call["action"] == "stop"]
    assert stops == ["router", *[worker.id for worker in controller.plan.workers if worker.role == "decode"],
                     *[worker.id for worker in controller.plan.workers if worker.role == "prefill"]]


def test_restart_generations_change(tmp_path):
    controller, fixture = setup_controller(tmp_path)
    controller.up()
    first = {key: value["generation"] for key, value in fixture.records.items()}
    controller.down()
    controller.up()
    assert all(fixture.records[key]["generation"] != generation for key, generation in first.items())
    controller.down()
