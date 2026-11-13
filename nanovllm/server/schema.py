from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    @field_validator("stream")
    @classmethod
    def non_streaming_only(cls, value):
        if value:
            raise ValueError("stream=true is not supported")
        return value
