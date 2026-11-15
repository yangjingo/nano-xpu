"""Versioned KV metadata, independent of transport handles and accelerator imports."""
import hashlib
import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Fingerprint = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(min_length=1, max_length=256)]


class StrictMetadata(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    @field_validator("protocol_version", "tensor_parallel_size", "layout_version", check_fields=False, mode="before")
    @classmethod
    def integer_versions(cls, value):
        if type(value) is not int:
            raise ValueError("protocol, layout and TP versions must be integers")
        return value


class ModelLayout(StrictMetadata):
    model_fingerprint: Fingerprint
    tokenizer_fingerprint: Fingerprint
    template_fingerprint: Fingerprint
    rope_fingerprint: Fingerprint
    dtype: Literal["float16", "bfloat16", "float32"]
    layers: int = Field(gt=0)
    kv_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    block_size: int = Field(gt=0)
    tensor_parallel_size: Literal[1] = 1
    layout_version: Literal[1] = 1

    @property
    def element_bytes(self):
        return 4 if self.dtype == "float32" else 2

    def shape(self, num_blocks):
        return [2, self.layers, num_blocks, self.block_size, self.kv_heads, self.head_dim]

    def strides(self, num_blocks):
        stride = self.element_bytes
        strides = []
        for length in reversed(self.shape(num_blocks)):
            strides.append(stride)
            stride *= length
        return tuple(reversed(strides))


class KVDescriptor(StrictMetadata):
    protocol_version: Literal[1] = 1
    cluster_id: Identifier
    lease_id: Identifier
    request_id: str = Field(min_length=1, max_length=1024)
    generation: Identifier
    layout: ModelLayout
    source_num_blocks: int = Field(gt=0)
    source_block_ids: tuple[Annotated[int, Field(ge=0)], ...]
    strides_bytes: tuple[Annotated[int, Field(ge=0)], ...] = Field(min_length=6, max_length=6)
    prompt_tokens: int = Field(gt=0)
    cached_tokens: int = Field(ge=0)
    expires_at: float = Field(gt=0, allow_inf_nan=False)
    registration_id: Identifier
    content_digest: Fingerprint
    prompt_digest: Fingerprint

    @model_validator(mode="after")
    def bounds(self):
        if self.cached_tokens != self.prompt_tokens - 1:
            raise ValueError("protocol v1 imports exactly L-1 prompt tokens")
        blocks = (self.cached_tokens + self.layout.block_size - 1) // self.layout.block_size
        if len(self.source_block_ids) != blocks or len(set(self.source_block_ids)) != blocks:
            raise ValueError("source block mapping must contain one unique block per logical block")
        if any(index >= self.source_num_blocks for index in self.source_block_ids):
            raise ValueError("source block mapping is outside the registered layout")
        if self.strides_bytes != self.layout.strides(self.source_num_blocks):
            raise ValueError("unsupported strides for [2,layers,blocks,block_size,kv_heads,head_dim]")
        return self

    @property
    def pinned_bytes(self):
        return 2 * self.layout.layers * len(self.source_block_ids) * self.layout.block_size * self.layout.kv_heads * self.layout.head_dim * self.layout.element_bytes

    @property
    def effective_bytes(self):
        return 2 * self.layout.layers * self.cached_tokens * self.layout.kv_heads * self.layout.head_dim * self.layout.element_bytes

    def digest(self):
        raw = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def validate_for(self, *, cluster_id, request_id, generation, layout, now, digest):
        if not math.isfinite(now) or now >= self.expires_at:
            raise ValueError("lease expired or invalid clock")
        if (cluster_id, request_id, generation) != (self.cluster_id, self.request_id, self.generation):
            raise ValueError("KV identity or generation mismatch")
        if layout != self.layout:
            raise ValueError("model, tokenizer, template, RoPE or KV layout mismatch")
        if digest != self.digest():
            raise ValueError("descriptor digest mismatch")

    def copy_spans(self, destination_block_ids: list[int], destination_num_blocks: int):
        """Return (source block, D-owned block, valid tokens), never a borrowed block table."""
        if len(destination_block_ids) != len(self.source_block_ids):
            raise ValueError("destination reservation has the wrong number of blocks")
        if len(set(destination_block_ids)) != len(destination_block_ids) or any(
            type(block) is not int or not 0 <= block < destination_num_blocks for block in destination_block_ids
        ):
            raise ValueError("invalid destination block reservation")
        return [(source, target, min(self.layout.block_size, self.cached_tokens - index * self.layout.block_size))
                for index, (source, target) in enumerate(zip(self.source_block_ids, destination_block_ids))]


def parse_descriptor(data: bytes) -> KVDescriptor:
    if len(data) > 64 * 1024:
        raise ValueError("KV descriptor exceeds 64 KiB")
    return KVDescriptor.model_validate_json(data)
