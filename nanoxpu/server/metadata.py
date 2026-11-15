"""Content identities without importing a tokenizer, tensor library or accelerator."""
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from nanoxpu.kv.descriptor import Fingerprint, StrictMetadata
from nanoxpu.utils.hashing import sha256_file


def digest_files(root, names):
    digest = hashlib.sha256()
    for name in sorted(names):
        path = root / name
        if not path.is_file():
            raise ValueError(f"required model file is missing: {name}")
        content = sha256_file(path)
        digest.update(json.dumps([name, content], separators=(",", ":")).encode())
    return digest.hexdigest()


class Fingerprints(StrictMetadata):
    model: Fingerprint
    tokenizer: Fingerprint
    template: Fingerprint
    rope: Fingerprint


def model_fingerprints(directory):
    """Hash actual local weights, not the model alias or modification timestamps."""
    root = Path(directory)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
    weights = [path.name for path in root.glob("*.safetensors")]
    if not weights:
        raise ValueError("model has no local safetensors weights")
    index = root / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        if not shards <= set(weights):
            raise ValueError("weight index references missing or nonlocal shards")
    tokenizer_files = [name for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                                        "added_tokens.json", "vocab.json", "merges.txt") if (root / name).is_file()]
    if "tokenizer.json" not in tokenizer_files:
        raise ValueError("model requires a local tokenizer.json")
    template_file = root / "chat_template.jinja"
    template = template_file.read_text(encoding="utf-8") if template_file.is_file() else tokenizer_config.get("chat_template")
    if not template:
        raise ValueError("model requires an explicit chat template")

    def canonical(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

    return Fingerprints(
        model=digest_files(root, ["config.json", *weights]),
        tokenizer=digest_files(root, tokenizer_files),
        template=canonical({"template": template, "add_generation_prompt": True, "enable_thinking": False}),
        rope=canonical({key: value for key, value in config.items() if key.startswith("rope_") or key == "max_position_embeddings"}),
    ).model_dump()


class Capabilities(StrictMetadata):
    protocol_version: Literal[1] = 1
    transport: Literal["none", "cpu-test", "nixl-pull"] = "none"
    transport_version: str = Field(default="none", min_length=1)
    connector_ready: bool = False
    fingerprints: Fingerprints
    layout_digest: Fingerprint | None = None


def server_info(config, capabilities):
    return {"instance_id": config.instance_id, "engine_generation": config.engine_generation,
            "cluster_id": config.cluster_id, "role": config.role, "model": config.served_model_name,
            "tensor_parallel_size": config.tensor_parallel_size,
            "capabilities": capabilities.model_dump() if capabilities is not None else None}
