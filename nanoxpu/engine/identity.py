"""Names scoped to one engine lifetime, including offline engine instances."""
import hashlib
import re
import uuid


def new_identity():
    return uuid.uuid4().hex


def shared_memory_name(instance_id: str, generation: str) -> str:
    for value in (instance_id, generation):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", value):
            raise ValueError("instance_id and generation must be safe identifiers, at most 160 characters")
    return "nanoxpu_" + hashlib.sha256(f"{instance_id}:{generation}".encode()).hexdigest()[:32]
