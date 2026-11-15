"""Transactional cluster control. SSH owns process actions; HTTP proves readiness."""
from dataclasses import asdict, replace
from pathlib import Path
import secrets
import time
import uuid

import httpx

from nanoxpu.deployment.planner import LaunchPlan, LaunchSpec
from nanoxpu.deployment.schema import RuntimeProfile
from nanoxpu.deployment.ssh import SSHExecutor
from nanoxpu.deployment.state import atomic_json, file_lock, read_json, safe_name, state_root
from nanoxpu.server.metadata import Capabilities, Fingerprints


class DeploymentError(RuntimeError):
    pass


class Controller:
    def __init__(self, plan, *, root=None, executor=None, http=None):
        self.plan = plan
        self.root = Path(root or state_root()) / "clusters"
        self.path = self.root / (safe_name(plan.name) + ".json")
        self.lock = self.path.with_suffix(".lock")
        self.executor = executor or SSHExecutor(plan.profile)
        self.http = http or httpx.Client(timeout=3, trust_env=False)
        self.owns_http = http is None

    def close(self):
        if self.owns_http:
            self.http.close()

    @classmethod
    def load(cls, name, *, root=None, **kwargs):
        path = Path(root or state_root()) / "clusters" / (safe_name(name) + ".json")
        record = read_json(path)
        if record is None:
            raise DeploymentError("no recorded deployment with this name")
        data = record["plan"]
        plan = LaunchPlan(data["name"], data["mode"], data["config_hash"], data["cluster_id"],
                          [LaunchSpec(**worker) for worker in data["workers"]], LaunchSpec(**data["router"]),
                          RuntimeProfile.model_validate(record["profile"]), data["unresolved"])
        return cls(plan, root=root, **kwargs)

    def rpc(self, target, action, **data):
        timeout = max(30, self.plan.profile.timeouts.drain_seconds * 2 + 20) if action == "stop" else 30
        if action == "preflight":
            timeout = max(timeout, self.plan.profile.timeouts.startup_seconds)
        return self.executor.request(target.ip, {"action": action, "name": self.plan.name, "role_id": target.id, **data}, timeout=timeout)

    def get(self, spec, path, token=None):
        headers = {"Authorization": "Bearer " + token} if token else {}
        response = self.http.get(spec.url + path, headers=headers)
        if response.status_code != 200:
            raise DeploymentError(f"{spec.id} {path} is not ready (HTTP {response.status_code})")
        return response

    def router_ready(self, spec):
        self.get(spec, "/health")
        workers = self.get(spec, "/workers").json().get("workers", [])
        expected = {(worker.url, "regular" if worker.role == "colocated" else worker.role) for worker in self.plan.workers}
        observed = {(worker.get("url"), worker.get("worker_type")) for worker in workers if worker.get("is_healthy") is True}
        if observed != expected or len(workers) != len(expected):
            raise DeploymentError("router worker pool is incomplete or unhealthy")

    def wait_ready(self, spec, token, generation, fingerprints):
        deadline = time.monotonic() + self.plan.profile.timeouts.startup_seconds
        while True:
            remote = self.rpc(spec, "status")
            if remote.get("phase") != "RUNNING":
                raise DeploymentError(f"{spec.id} process is not running")
            try:
                if spec.role == "router":
                    self.router_ready(spec)
                    return None
                self.get(spec, "/ready")
                info = self.get(spec, "/server_info", token).json()
                break
            except (httpx.TransportError, DeploymentError):
                if time.monotonic() >= deadline:
                    raise DeploymentError(f"{spec.id} readiness timed out") from None
                time.sleep(0.1)
        identity = (info.get("instance_id"), info.get("engine_generation"), info.get("cluster_id"), info.get("role"), info.get("model"))
        if identity != (spec.instance_id, generation, self.plan.cluster_id, spec.role, self.plan.profile.served_model_name):
            raise DeploymentError(f"{spec.id} reports a different role/model/generation")
        caps = Capabilities.model_validate(info.get("capabilities"))
        if caps.fingerprints != Fingerprints.model_validate(fingerprints):
            raise DeploymentError(f"{spec.id} model contents changed after preflight")
        if spec.role != "colocated":
            if (caps.transport != self.plan.profile.kv.connector
                or caps.transport_version != self.plan.profile.kv.expected_version
                or not caps.connector_ready or caps.layout_digest is None or info.get("tensor_parallel_size") != 1):
                raise DeploymentError(f"{spec.id} has incompatible KV capabilities")
        return info

    def up(self):
        with file_lock(self.lock):
            old = read_json(self.path)
            if old and old["phase"] != "STOPPED":
                if old["plan"]["config_hash"] != self.plan.config_hash:
                    raise DeploymentError("configuration changed; stop the recorded deployment first")
                result = self.status()
                if old["phase"] == "RUNNING" and result["ready"]:
                    return result
                raise DeploymentError("recorded deployment is incomplete/degraded; run down before retrying")
            if self.plan.unresolved:
                raise DeploymentError("; ".join(self.plan.unresolved))
            fingerprints = {}
            # Complete ALL preflight checks before the first worker is started.
            for spec in [*self.plan.workers, self.plan.router]:
                request = {"spec": asdict(spec)}
                if spec.role == "router":
                    request["router_revision"] = self.plan.profile.router.source_revision
                else:
                    request.update(expected_revision=self.plan.profile.runtime.expected_nano_xpu_revision,
                                   model_path=spec.argv[spec.argv.index("_worker") + 1])
                evidence = self.rpc(spec, "preflight", **request)
                if spec.role != "router":
                    fingerprints[spec.id] = Fingerprints.model_validate(evidence["model"]).model_dump()
            if any(value != next(iter(fingerprints.values())) for value in fingerprints.values()):
                raise DeploymentError("workers have different model/tokenizer/template/RoPE contents")
            generations = {spec.id: uuid.uuid4().hex for spec in [*self.plan.workers, self.plan.router]}
            peers = [f"{spec.instance_id}={generations[spec.id]}" for spec in self.plan.workers if spec.role == "prefill"]
            specs = []
            for spec in self.plan.workers:
                argv = [*spec.argv, "--cluster-id", self.plan.cluster_id, "--engine-generation", generations[spec.id]]
                if spec.role == "decode":
                    for peer in peers:
                        argv += ["--prefill-peer", peer]
                specs.append(replace(spec, argv=argv))
            specs.append(self.plan.router)
            record = {"phase": "STARTING", "plan": self.plan.public_dict(), "profile": self.plan.profile.model_dump(),
                      "roles": [], "started_at": time.time()}
            atomic_json(self.path, record)
            token = secrets.token_urlsafe(32)
            infos = []
            try:
                for spec in specs:
                    entry = {"spec": asdict(spec), "generation": generations[spec.id], "owner": uuid.uuid4().hex,
                             "phase": "STARTING"}
                    record["roles"].append(entry)
                    # Persist the ownership nonce before sending start, including for lost replies.
                    atomic_json(self.path, record)
                    remote = self.rpc(spec, "start", spec=asdict(spec), config_hash=self.plan.config_hash,
                                      generation=entry["generation"], owner=entry["owner"], auth_token=token,
                                      drain_seconds=float(self.plan.profile.timeouts.drain_seconds))
                    if remote.get("owner") != entry["owner"] or remote.get("generation") != entry["generation"]:
                        raise DeploymentError("start response belongs to a different owner/generation")
                    info = self.wait_ready(spec, token, entry["generation"], fingerprints.get(spec.id))
                    if info:
                        infos.append(info)
                        if self.plan.mode == "disaggregated" and any(
                            item["capabilities"]["layout_digest"] != infos[0]["capabilities"]["layout_digest"] for item in infos
                        ):
                            raise DeploymentError("P/D KV layouts differ")
                    entry["phase"] = "RUNNING"
                    entry["metadata"] = info
                    atomic_json(self.path, record)
                record["phase"] = "RUNNING"
                atomic_json(self.path, record)
                result = self.status()
                if not result["ready"]:
                    raise DeploymentError("cluster lost readiness before startup completed")
                return result
            except BaseException as exc:
                # Never persist arbitrary exception text that might carry bootstrap credentials.
                record["failure"] = type(exc).__name__
                self._down(record)
                raise

    def status(self):
        record = read_json(self.path)
        if record is None:
            return {"name": self.plan.name, "phase": "ABSENT", "ready": False, "roles": []}
        roles = []
        for entry in record["roles"]:
            spec = LaunchSpec(**entry["spec"])
            result = {"id": spec.id, "role": spec.role, "phase": entry["phase"], "ready": False}
            if entry["phase"] == "STOPPED":
                roles.append(result)
                continue
            try:
                remote = self.rpc(spec, "status")
                result["phase"] = remote["phase"]
                if (remote.get("owner") != entry["owner"] or remote.get("generation") != entry["generation"]):
                    raise DeploymentError("remote ownership changed")
                if result["phase"] == "RUNNING":
                    if spec.role == "router":
                        self.router_ready(spec)
                    else:
                        self.get(spec, "/ready")
                    result["ready"] = True
            except Exception as exc:
                result["phase"] = "UNREACHABLE" if isinstance(exc, (OSError, httpx.TransportError)) else "DEGRADED"
                result["error"] = type(exc).__name__
            roles.append(result)
        ready = record["phase"] == "RUNNING" and len(roles) == len(self.plan.workers) + 1 and all(role["ready"] for role in roles)
        phase = "RUNNING" if ready else ("STOPPED" if record["phase"] == "STOPPED" else "DEGRADED")
        return {"name": self.plan.name, "phase": phase, "ready": ready, "router_url": self.plan.router.url, "roles": roles}

    def _down(self, record):
        record["phase"] = "STOPPING"
        atomic_json(self.path, record)
        order = {"router": 0, "decode": 1, "colocated": 1, "prefill": 2}
        for entry in sorted(record["roles"], key=lambda item: order[item["spec"]["role"]]):
            if entry["phase"] == "STOPPED":
                continue
            spec = LaunchSpec(**entry["spec"])
            try:
                result = self.rpc(spec, "stop", owner=entry["owner"], drain_seconds=float(self.plan.profile.timeouts.drain_seconds))
                if result["phase"] not in ("STOPPED", "ABSENT"):
                    raise DeploymentError("remote cleanup did not finish")
                entry["phase"] = "STOPPED"
                entry.pop("cleanup_error", None)
            except Exception as exc:
                entry["phase"] = "CLEANUP_FAILED"
                entry["cleanup_error"] = type(exc).__name__
            atomic_json(self.path, record)
        record["phase"] = "STOPPED" if all(entry["phase"] == "STOPPED" for entry in record["roles"]) else "CLEANUP_FAILED"
        atomic_json(self.path, record)
        return record["phase"] == "STOPPED"

    def down(self):
        with file_lock(self.lock):
            record = read_json(self.path)
            if record and not self._down(record):
                raise DeploymentError("some resources could not be cleaned; ledger retained, retry down")
            return self.status()

    def logs(self, role=None):
        record = read_json(self.path)
        if record is None:
            raise DeploymentError("no recorded deployment")
        selected = [entry for entry in record["roles"] if role is None or role in (entry["spec"]["id"], entry["spec"]["role"])]
        if not selected:
            raise DeploymentError("no recorded worker matches this role or ID")
        return {entry["spec"]["id"]: self.rpc(LaunchSpec(**entry["spec"]), "logs")["text"] for entry in selected}
