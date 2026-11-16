"""Required Linux integration prerequisites fail visibly; they are never pytest skips."""
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from nanoxpu.deployment.schema import ROUTER_REVISION


@pytest.fixture(scope="session")
def router_binary():
    if sys.platform != "linux":
        pytest.fail("real router/SSH integration requires Linux; run the required CI job")
    path = Path(os.environ.get("NANO_XPU_ROUTER_BIN", ""))
    if not path.is_file():
        pytest.fail("NANO_XPU_ROUTER_BIN must identify the pinned upstream build")
    receipt = json.loads(Path(str(path) + ".source.json").read_text(encoding="utf-8"))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert receipt["revision"] == ROUTER_REVISION and receipt["sha256"] == digest
    return path.resolve()
