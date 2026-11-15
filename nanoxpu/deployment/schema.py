"""RFC-0002 configuration. Validation does not inspect or connect to remote hosts."""
from ipaddress import IPv4Address
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml

ROUTER_REVISION = "07c20da66c255ba65a97ab531f9fb5f60413a7d4"
Port = Annotated[int, Field(ge=1, le=65535)]
Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,62}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    @field_validator("schema_version", check_fields=False, mode="before")
    @classmethod
    def integer_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value


def ipv4(value: str) -> str:
    address = IPv4Address(value)
    if address.is_unspecified or address.is_multicast or str(address) == "255.255.255.255":
        raise ValueError("endpoint must be a reachable unicast IPv4 address")
    return str(address)


def absolute_path(value: str) -> str:
    if not PurePosixPath(value).is_absolute() or "\x00" in value or "\n" in value:
        raise ValueError("remote paths must be absolute Linux paths without NUL/newline")
    return value


class Runtime(StrictModel):
    nano_xpu_bin: str
    workdir: str
    expected_nano_xpu_revision: str = Field(min_length=1)
    # This is the only variable remote command token; it cannot contain shell syntax.
    helper_python: str = Field(default="/usr/bin/python3", pattern=r"^/[A-Za-z0-9_./-]+$")

    _paths = field_validator("nano_xpu_bin", "workdir")(absolute_path)


class SSH(StrictModel):
    user: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_-]*$")
    port: Port = 22
    identity_file: str | None = None
    known_hosts_file: str | None = None

    @field_validator("identity_file", "known_hosts_file")
    @classmethod
    def identity(cls, value):
        if value is not None and (not value or any(char in value for char in "\x00\n\r")):
            raise ValueError("invalid SSH file path")
        return value


class RouterProfile(StrictModel):
    executable: str
    source_revision: Literal[ROUTER_REVISION]
    policy: Literal["round_robin"] = "round_robin"
    port: Port = 8000

    _path = field_validator("executable")(absolute_path)


class EngineProfile(StrictModel):
    max_model_len: int = Field(default=4096, gt=0)
    max_num_batched_tokens: int = Field(default=2048, gt=0)
    max_num_seqs: int = Field(default=1, gt=0)
    gpu_memory_utilization: float = Field(default=0.9, gt=0, lt=1, allow_inf_nan=False)
    enforce_eager: bool = True


class KVProfile(StrictModel):
    connector: Literal["nixl-pull"] = "nixl-pull"
    expected_version: str = Field(min_length=1)
    lease_seconds: int = Field(default=180, gt=0)


class Timeouts(StrictModel):
    startup_seconds: int = Field(default=600, gt=0)
    request_seconds: int = Field(default=120, gt=0)
    drain_seconds: int = Field(default=30, gt=0)


class RuntimeProfile(StrictModel):
    schema_version: Literal[1]
    model_path: str
    served_model_name: str = Field(min_length=1)
    runtime: Runtime
    ssh: SSH
    router: RouterProfile
    engine: EngineProfile = Field(default_factory=EngineProfile)
    kv: KVProfile
    timeouts: Timeouts = Field(default_factory=Timeouts)

    _model_path = field_validator("model_path")(absolute_path)

    @field_validator("served_model_name")
    @classmethod
    def model_name(cls, value):
        if not value.strip():
            raise ValueError("served_model_name must not be blank")
        return value

    @model_validator(mode="after")
    def lease_deadline(self):
        if self.kv.lease_seconds < self.timeouts.request_seconds + self.timeouts.drain_seconds:
            raise ValueError("lease_seconds must cover request_seconds + drain_seconds")
        return self


class RouterNode(StrictModel):
    ip: str
    advertise_ip: str | None = None

    _ip = field_validator("ip")(ipv4)

    @field_validator("advertise_ip")
    @classmethod
    def advertised(cls, value):
        return ipv4(value) if value is not None else None


class WorkerNode(RouterNode):
    id: Name
    role: Literal["colocated", "prefill", "decode"]
    gpus: list[Annotated[int, Field(ge=0)]] = Field(min_length=1, max_length=8)
    tensor_parallel_size: int | None = Field(default=None, ge=1, le=8)
    model_path: str | None = None
    http_port: Port | None = None
    distributed_port: Port | None = None
    kv_port: Port | None = None

    @field_validator("model_path")
    @classmethod
    def model_override(cls, value):
        return absolute_path(value) if value is not None else None

    @model_validator(mode="after")
    def gpu_mapping(self):
        if len(set(self.gpus)) != len(self.gpus):
            raise ValueError("duplicate GPU within worker")
        if self.tensor_parallel_size is not None and self.tensor_parallel_size != len(self.gpus):
            raise ValueError("tensor_parallel_size must equal number of assigned GPUs")
        if self.role != "colocated" and len(self.gpus) != 1:
            raise ValueError("P/D workers require TP=1 in protocol v1")
        if self.role == "colocated" and self.kv_port is not None:
            raise ValueError("colocated workers must not expose a KV port")
        return self


class ClusterConfig(StrictModel):
    schema_version: Literal[1]
    name: Name
    profile: str = Field(min_length=1)
    mode: Literal["colocated", "disaggregated"]
    router: RouterNode
    workers: list[WorkerNode] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def roles_and_ownership(self):
        ids = [worker.id for worker in self.workers]
        if len(set(ids)) != len(ids) or "router" in ids:
            raise ValueError("worker IDs must be unique and cannot be 'router'")
        roles = {worker.role for worker in self.workers}
        expected = {"colocated"} if self.mode == "colocated" else {"prefill", "decode"}
        if roles != expected:
            raise ValueError(f"{self.mode} requires exactly the role set {sorted(expected)}")
        owners = set()
        for worker in self.workers:
            for gpu in worker.gpus:
                key = (worker.ip, gpu)
                if key in owners:
                    raise ValueError(f"GPU {gpu} on {worker.ip} is assigned more than once")
                owners.add(key)
        return self


class UniqueKeyLoader(yaml.SafeLoader):
    """A repeated YAML key would otherwise silently replace the reviewed value."""


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"YAML mapping keys must be unique strings: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def read_yaml(path: Path) -> dict:
    # Configs are small control inputs, never model tensors or executable YAML tags.
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("deployment config exceeds 1 MiB")
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid deployment YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("deployment config must be a mapping")
    return value


def load_config(path: str | Path) -> tuple[ClusterConfig, RuntimeProfile]:
    source = Path(path).resolve()
    config = ClusterConfig.model_validate(read_yaml(source))
    profile = RuntimeProfile.model_validate(read_yaml(source.parent / config.profile))
    return config, profile
