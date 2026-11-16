"""CPU-only test engine. No model is loaded and no model output is claimed."""
import threading
import json
from pathlib import Path

from nanoxpu.server.engine import Completion, InvalidRequest


def write_model_fixture(root):
    """Small byte fixtures for fingerprint tests, never loadable model weights."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": "qwen3", "rope_theta": 10000,
                                                "max_position_embeddings": 4096}), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}), encoding="utf-8")
    (root / "tokenizer.json").write_text('{"fixture": true}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"CPU fingerprint fixture; not a model")
    return root


class TestEngine:
    __test__ = False

    def __init__(self, config):
        self.config = config
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls = []
        self.closed = 0
        self.fail_start = False
        self.fail_generation = False

    def start(self):
        self.calls.append(("start", threading.get_ident()))
        if self.fail_start:
            raise RuntimeError("injected startup failure")
        if not hasattr(self, "capabilities") and (Path(self.config.model) / "model.safetensors").exists():
            from nanoxpu.server.metadata import Capabilities, Fingerprints, model_fingerprints
            self.capabilities = Capabilities(fingerprints=Fingerprints(**model_fingerprints(self.config.model)))

    def complete(self, request):
        self.calls.append(("complete", threading.get_ident()))
        self.started.set()
        if not self.release.wait(10):
            raise RuntimeError("CPU test timed out")
        if self.fail_generation:
            raise RuntimeError("injected generation failure")
        # Synthetic character counts are only for testing HTTP accounting.
        prompt_tokens = sum(len(message.content) for message in request.messages)
        if prompt_tokens + request.max_tokens > self.config.max_model_len:
            raise InvalidRequest("test prompt exceeds max_model_len")
        return Completion("[CPU TEST ENGINE: no Qwen model loaded]", prompt_tokens, 1, "stop")

    def close(self):
        self.calls.append(("close", threading.get_ident()))
        self.closed += 1
