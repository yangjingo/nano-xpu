from dataclasses import dataclass, field
from pathlib import Path
import math
from ipaddress import IPv4Address
from nanoxpu.engine.identity import new_identity, shared_memory_name


@dataclass(frozen=True)
class ServerConfig:
    model: str
    served_model_name: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    tensor_parallel_size: int = 1
    distributed_port: int = 2333
    max_model_len: int = 4096
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    instance_id: str = field(default_factory=new_identity)
    engine_generation: str = field(default_factory=new_identity)
    role: str = "colocated"
    advertise_ip: str | None = None
    kv_port: int | None = None
    kv_connector: str | None = None
    cluster_id: str | None = None
    prefill_peers: tuple[str, ...] = ()

    def __post_init__(self):
        shared_memory_name(self.instance_id, self.engine_generation)
        if self.cluster_id is not None:
            shared_memory_name(self.cluster_id, self.engine_generation)
        peers = {}
        for entry in self.prefill_peers:
            instance, separator, generation = entry.partition("=")
            if not separator or instance in peers:
                raise ValueError("prefill peers must be unique INSTANCE=GENERATION entries")
            shared_memory_name(instance, generation)
            peers[instance] = generation
        if self.prefill_peers and self.role != "decode":
            raise ValueError("only decode workers accept prefill peers")
        if self.role not in ("colocated", "prefill", "decode"):
            raise ValueError("unknown worker role")
        if self.advertise_ip is not None:
            address = IPv4Address(self.advertise_ip)
            if address.is_unspecified or address.is_multicast or str(address) == "255.255.255.255":
                raise ValueError("advertise_ip must be a unicast IPv4 address")
        if self.role == "colocated":
            if self.kv_port is not None or self.kv_connector is not None:
                raise ValueError("colocated workers do not use a KV connector")
        elif self.kv_connector != "nixl-pull" or self.kv_port is None or not 1 <= self.kv_port <= 65535 or self.tensor_parallel_size != 1:
            raise ValueError("P/D roles require nixl-pull, a valid KV port and TP=1")
        if not Path(self.model).is_dir():
            raise ValueError(f"model directory does not exist: {self.model}")
        if not self.host.strip():
            raise ValueError("host must not be empty")
        for name in ("port", "distributed_port"):
            if not 1 <= getattr(self, name) <= 65535:
                raise ValueError(f"{name} must be between 1 and 65535")
        ports = [self.port, self.distributed_port] + ([self.kv_port] if self.kv_port is not None else [])
        if len(set(ports)) != len(ports):
            raise ValueError("HTTP, rendezvous and KV ports must be distinct")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")
        for name in ("max_model_len", "max_num_batched_tokens", "max_num_seqs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.gpu_memory_utilization) or not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be finite and between 0 and 1")
        model_name = Path(self.model).resolve().name if self.served_model_name is None else self.served_model_name
        if not model_name.strip():
            raise ValueError("served_model_name must not be empty")
        object.__setattr__(self, "served_model_name", model_name)
