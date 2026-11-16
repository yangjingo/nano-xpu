"""Deterministic launch plans. No network, subprocesses, state writes or GPU discovery."""
import hashlib
import json
import re
from dataclasses import asdict, dataclass

from nanoxpu.deployment.schema import ClusterConfig, RuntimeProfile, load_config


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class LaunchSpec:
    id: str
    role: str
    ip: str
    advertise_ip: str
    argv: list[str]
    env: dict[str, str]
    workdir: str
    http_port: int
    distributed_port: int | None
    kv_port: int | None
    gpus: list[int]
    instance_id: str

    @property
    def url(self):
        return f"http://{self.advertise_ip}:{self.http_port}"


@dataclass(frozen=True)
class LaunchPlan:
    name: str
    mode: str
    config_hash: str
    cluster_id: str
    workers: list[LaunchSpec]
    router: LaunchSpec
    profile: RuntimeProfile
    unresolved: list[str]

    def public_dict(self):
        return {"schema_version": 1, "name": self.name, "mode": self.mode,
                "config_hash": self.config_hash, "cluster_id": self.cluster_id,
                "workers": [asdict(worker) for worker in self.workers],
                "router": asdict(self.router), "router_url": self.router.url,
                "unresolved": self.unresolved}


def plan(config: ClusterConfig, profile: RuntimeProfile) -> LaunchPlan:
    config_hash = digest({"cluster": config.model_dump(), "profile": profile.model_dump()})
    cluster_id = f"{config.name}-{config_hash[:12]}"
    ports = set()

    def reserve(ip, port):
        if not 1 <= port <= 65535 or (ip, port) in ports:
            raise ValueError(f"invalid or conflicting port {ip}:{port}")
        ports.add((ip, port))
        return port

    reserve(config.router.ip, profile.router.port)
    workers = []
    for index, node in enumerate(config.workers):
        http_port = reserve(node.ip, node.http_port if node.http_port is not None else 8100 + index)
        distributed_port = reserve(node.ip, node.distributed_port if node.distributed_port is not None else 23000 + index)
        kv_port = None
        if node.role != "colocated":
            kv_port = reserve(node.ip, node.kv_port if node.kv_port is not None else 24000 + index)
        instance_id = f"{cluster_id}-{node.id}"
        argv = [profile.runtime.nano_xpu_bin, "_worker", node.model_path or profile.model_path,
                "--served-model-name", profile.served_model_name,
                "--role", node.role, "--instance-id", instance_id,
                "--host", "0.0.0.0", "--advertise-ip", node.advertise_ip or node.ip,
                "--port", str(http_port), "--distributed-port", str(distributed_port),
                "--tensor-parallel-size", str(len(node.gpus))]
        for key, value in profile.engine.model_dump().items():
            if isinstance(value, bool):
                if value:
                    argv.append("--" + key.replace("_", "-"))
            else:
                argv.extend(["--" + key.replace("_", "-"), str(value)])
        if kv_port is not None:
            argv.extend(["--kv-port", str(kv_port), "--kv-connector", profile.kv.connector])
        workers.append(LaunchSpec(node.id, node.role, node.ip, node.advertise_ip or node.ip,
                                  argv, {"CUDA_VISIBLE_DEVICES": ",".join(map(str, node.gpus))},
                                  profile.runtime.workdir, http_port, distributed_port, kv_port,
                                  node.gpus, instance_id))
    argv = [profile.router.executable, "--host", "0.0.0.0", "--port", str(profile.router.port),
            "--policy", profile.router.policy, "--health-check-endpoint", "/ready", "--disable-retries",
            "--request-timeout-secs", str(profile.timeouts.request_seconds),
            "--worker-startup-timeout-secs", str(profile.timeouts.startup_seconds)]
    if config.mode == "colocated":
        argv += ["--worker-urls", *[worker.url for worker in workers]]
    else:
        argv += ["--vllm-pd-disaggregation", "--kv-connector", "nixl"]
        # --prefill / --decode are repeatable in the upstream launcher contract.
        for worker in workers:
            argv += ["--" + worker.role, worker.url]
    router = LaunchSpec("router", "router", config.router.ip, config.router.advertise_ip or config.router.ip,
                        argv, {}, profile.runtime.workdir, profile.router.port, None, None, [], cluster_id + "-router")
    unresolved = []
    if not re.fullmatch(r"[0-9a-f]{40}", profile.runtime.expected_nano_xpu_revision):
        unresolved.append("pin runtime.expected_nano_xpu_revision before nano-xpu serve -f")
    if config.mode == "disaggregated" and profile.kv.expected_version.startswith("<"):
        unresolved.append("pin kv.expected_version after the NIXL device spike before nano-xpu serve -f")
    return LaunchPlan(config.name, config.mode, config_hash, cluster_id, workers, router, profile, unresolved)


def plan_file(path) -> LaunchPlan:
    return plan(*load_config(path))
