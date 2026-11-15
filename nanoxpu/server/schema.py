from typing import Literal
from ipaddress import IPv4Address

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from nanoxpu.kv.descriptor import Fingerprint, Identifier, StrictMetadata


class Message(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(default=0.6, gt=1e-10, allow_inf_nan=False)
    max_tokens: int = Field(default=64, gt=0)
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1)
    # The pinned router serializes these defaults even when the client omits them.
    # Only neutral values are accepted; this does not enable reasoning/logprob/hidden-state APIs.
    logprobs: Literal[False] = False
    no_stop_trim: Literal[False] = False
    ignore_eos: Literal[False] = False
    continue_final_message: Literal[False] = False
    skip_special_tokens: Literal[True] = True
    separate_reasoning: Literal[True] = True
    stream_reasoning: Literal[True] = True
    return_hidden_states: Literal[False] = False
    include_reasoning: Literal[True] = True

    @field_validator("logprobs", "no_stop_trim", "ignore_eos", "continue_final_message", "skip_special_tokens",
                     "separate_reasoning", "stream_reasoning", "return_hidden_states", "include_reasoning", mode="before")
    @classmethod
    def router_default_booleans(cls, value):
        if type(value) is not bool:
            raise ValueError("router default flags must be booleans")
        return value

    @field_validator("stream")
    @classmethod
    def non_streaming_only(cls, value):
        if value:
            raise ValueError("stream=true is not supported")
        return value


class PrefillKV(StrictMetadata):
    do_remote_decode: Literal[True]
    do_remote_prefill: Literal[False]
    remote_engine_id: None
    remote_block_ids: None
    remote_host: None
    remote_port: None

    @field_validator("do_remote_decode", "do_remote_prefill", mode="before")
    @classmethod
    def booleans(cls, value):
        if type(value) is not bool:
            raise ValueError("stage flags must be booleans")
        return value


class NanoTransfer(StrictMetadata):
    protocol_version: Literal[1]
    lease_id: Identifier
    generation: Identifier
    expires_at: float = Field(gt=0, allow_inf_nan=False)
    cached_tokens: int = Field(ge=0)
    descriptor_digest: Fingerprint


class DecodeKV(StrictMetadata):
    do_remote_decode: Literal[False]
    do_remote_prefill: Literal[True]
    remote_engine_id: Identifier
    remote_block_ids: list[int] = Field(max_length=4096)
    remote_host: str
    remote_port: int = Field(ge=1, le=65535)
    nano: NanoTransfer

    _booleans = field_validator("do_remote_decode", "do_remote_prefill", mode="before")(PrefillKV.booleans.__func__)
    @field_validator("remote_host")
    @classmethod
    def host(cls, value):
        address = IPv4Address(value)
        if address.is_unspecified or address.is_multicast or str(address) == "255.255.255.255":
            raise ValueError("remote host must be a unicast IPv4 address")
        return str(address)

    @field_validator("remote_block_ids")
    @classmethod
    def block_ids(cls, value):
        if any(block < 0 for block in value) or len(set(value)) != len(value):
            raise ValueError("remote blocks must be unique nonnegative IDs")
        return value


class StageRequest(ChatRequest):
    _request_id: str = PrivateAttr(default="")

    @property
    def request_id(self):
        return self._request_id

    def bind_request_id(self, value):
        if not value or len(value) > 1024 or any(ord(char) < 33 or ord(char) > 126 for char in value):
            raise ValueError("P/D requires a bounded printable X-Request-Id")
        self._request_id = value
        return self


class PrefillRequest(StageRequest):
    kv_transfer_params: PrefillKV
    min_tokens: int = Field(ge=1, le=1)

    @model_validator(mode="after")
    def one_internal_token(self):
        if self.max_tokens != 1:
            raise ValueError("prefill requires max_tokens=1")
        return self


class DecodeRequest(StageRequest):
    kv_transfer_params: DecodeKV
