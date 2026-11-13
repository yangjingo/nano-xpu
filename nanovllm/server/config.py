from dataclasses import dataclass
from pathlib import Path
import math


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

    def __post_init__(self):
        if not Path(self.model).is_dir():
            raise ValueError(f"model directory does not exist: {self.model}")
        if not self.host.strip():
            raise ValueError("host must not be empty")
        for name in ("port", "distributed_port"):
            if not 1 <= getattr(self, name) <= 65535:
                raise ValueError(f"{name} must be between 1 and 65535")
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
