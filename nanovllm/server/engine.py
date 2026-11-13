from dataclasses import dataclass
from typing import Protocol

from nanovllm.server.config import ServerConfig
from nanovllm.server.schema import ChatRequest


class InvalidRequest(ValueError):
    """Rejected before any engine generation state is modified."""


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class Engine(Protocol):
    def start(self) -> None: ...
    def complete(self, request: ChatRequest) -> Completion: ...
    def close(self) -> None: ...


class NanoEngine:
    """Thin CUDA adapter. CPU tests inject their own Engine, never a fallback."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.llm = None

    def start(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("nano-xpu inference requires CUDA; no CPU inference backend is available")
        from transformers import AutoConfig
        from nanovllm import LLM

        config = self.config
        hf_config = AutoConfig.from_pretrained(config.model, local_files_only=True)
        if hf_config.model_type != "qwen3":
            raise ValueError("serve currently supports Qwen3 models only")
        if config.max_model_len > hf_config.max_position_embeddings:
            raise ValueError("max_model_len exceeds the model context limit")
        if torch.cuda.device_count() < config.tensor_parallel_size:
            raise ValueError("not enough visible CUDA devices for tensor_parallel_size")
        for name in ("num_attention_heads", "num_key_value_heads", "hidden_size", "intermediate_size", "vocab_size"):
            if getattr(hf_config, name) % config.tensor_parallel_size:
                raise ValueError(f"model {name} must be divisible by tensor_parallel_size")
        self.llm = LLM(
            config.model,
            tensor_parallel_size=config.tensor_parallel_size,
            distributed_port=config.distributed_port,
            max_model_len=config.max_model_len,
            max_num_batched_tokens=config.max_num_batched_tokens,
            max_num_seqs=config.max_num_seqs,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
        )

    def complete(self, request):
        from nanovllm import SamplingParams

        tokens = self.llm.tokenizer.apply_chat_template(
            [message.model_dump() for message in request.messages],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if not tokens:
            raise InvalidRequest("chat template produced an empty prompt")
        total = len(tokens) + request.max_tokens
        if total > self.config.max_model_len:
            raise InvalidRequest("prompt plus max_tokens exceeds max_model_len")
        blocks = self.llm.scheduler.block_manager
        required_blocks = (total + blocks.block_size - 1) // blocks.block_size
        if required_blocks > len(blocks.blocks):
            raise InvalidRequest("request exceeds total KV cache block capacity")
        params = SamplingParams(temperature=request.temperature, max_tokens=request.max_tokens)
        output = self.llm.generate([tokens], params, use_tqdm=False)[0]
        generated = output["token_ids"]
        reason = "stop" if generated[-1] == self.llm.tokenizer.eos_token_id else "length"
        text = self.llm.tokenizer.decode(generated, skip_special_tokens=True)
        return Completion(text, len(tokens), len(generated), reason)

    def close(self):
        if self.llm is not None:
            self.llm.exit()
            self.llm = None
