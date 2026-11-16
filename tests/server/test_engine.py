"""CPU tests of adapter boundaries; all accelerator/model calls are stubbed."""
from types import SimpleNamespace
import sys

import pytest
import nanoxpu

from nanoxpu.server.config import ServerConfig
from nanoxpu.server.engine import InvalidRequest, NanoEngine
from nanoxpu.server.schema import ChatRequest


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.options = kwargs
        return [1, 2, 3]

    def decode(self, tokens, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return "stubbed completion"


class LLMStub:
    def __init__(self, path, **kwargs):
        self.path, self.options = path, kwargs
        self.tokenizer = Tokenizer()
        self.scheduler = SimpleNamespace(block_manager=SimpleNamespace(block_size=4, blocks=[None] * 4))
        self.generated = []
        self.closed = 0

    def generate(self, prompts, params, use_tqdm):
        self.generated.append((prompts, params, use_tqdm))
        return [{"text": "unused", "token_ids": [8, 99]}]

    def exit(self):
        self.closed += 1


@pytest.fixture
def engine(tmp_path, monkeypatch):
    hf = SimpleNamespace(model_type="qwen3", max_position_embeddings=8192,
                         num_attention_heads=16, num_key_value_heads=8,
                         hidden_size=1024, intermediate_size=4096, vocab_size=1000)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: hf)))
    monkeypatch.setitem(nanoxpu.__dict__, "LLM", LLMStub)
    config = ServerConfig(str(tmp_path), tensor_parallel_size=2,
                          distributed_port=2444, max_num_batched_tokens=128, max_num_seqs=32)
    adapter = NanoEngine(config)
    adapter.start()
    yield adapter
    adapter.close()


def request(**changes):
    return ChatRequest(model="test", messages=[{"role": "user", "content": "hello"}], max_tokens=4, **changes)


def test_tp_chunk_budget_and_template_reach_engine(engine):
    assert engine.llm.options["tensor_parallel_size"] == 2
    assert engine.llm.options["max_num_batched_tokens"] == 128
    assert engine.llm.options["max_num_seqs"] == 32
    assert engine.llm.options["distributed_port"] == 2444
    result = engine.complete(request())
    assert result.prompt_tokens == 3 and result.completion_tokens == 2
    assert result.finish_reason == "stop"
    assert engine.llm.generated[0][0] == [[1, 2, 3]]
    assert engine.llm.tokenizer.options == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}


def test_context_overflow_rejected_before_generate(engine):
    value = request()
    value.max_tokens = engine.config.max_model_len
    with pytest.raises(InvalidRequest, match="max_model_len"):
        engine.complete(value)
    assert not engine.llm.generated


def test_kv_capacity_rejected_before_generate(engine):
    engine.llm.scheduler.block_manager.blocks = [None]
    with pytest.raises(InvalidRequest, match="KV cache"):
        engine.complete(request())
    assert not engine.llm.generated


def test_close_is_idempotent(engine):
    llm = engine.llm
    engine.close()
    engine.close()
    assert llm.closed == 1


def test_no_automatic_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        NanoEngine(ServerConfig(str(tmp_path))).start()


def test_bad_tp_model_dimensions_fail_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)))
    hf = SimpleNamespace(model_type="qwen3", max_position_embeddings=8192, num_attention_heads=15)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: hf)))
    monkeypatch.setitem(nanoxpu.__dict__, "LLM", LLMStub)
    with pytest.raises(ValueError, match="num_attention_heads"):
        NanoEngine(ServerConfig(str(tmp_path), tensor_parallel_size=2)).start()
