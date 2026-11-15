from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from nanoxpu.engine.identity import new_identity, shared_memory_name

if TYPE_CHECKING:
    from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    distributed_port: int = 2333
    instance_id: str = field(default_factory=new_identity)
    engine_generation: str = field(default_factory=new_identity)
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        from transformers import AutoConfig

        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        shared_memory_name(self.instance_id, self.engine_generation)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
