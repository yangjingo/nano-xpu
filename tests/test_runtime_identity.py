import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.helpers.checkout_fixture import committed_checkout
from tests.helpers.process import helper, free_port


def identity(checkout):
    result = subprocess.run([sys.executable, "-m", "nanoxpu", "runtime-info"], cwd=checkout,
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_actual_clean_checkout_and_dirty_runtime_are_distinguished(tmp_path):
    checkout, revision = committed_checkout(tmp_path / "source")
    result = identity(checkout)
    assert result["revision"] == revision and result["clean"]
    assert Path(result["package_root"]) == checkout / "nanoxpu"
    (checkout / "README.md").write_text("uncommitted docs", encoding="utf-8")
    assert identity(checkout)["clean"]
    with (checkout / "nanoxpu/cli.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# uncommitted runtime change\n")
    assert not identity(checkout)["clean"]


@pytest.mark.parametrize("case", ["clean", "revision", "different_package", "dirty"])
def test_preflight_checks_actual_worker_and_helper_not_only_workdir(tmp_path, case):
    checkout, revision = committed_checkout(tmp_path / "source")
    prefix = [sys.executable, "-m", "tests.helpers.cpu_server"]
    if case == "different_package":
        # A launcher points at a different import root while the configured workdir is valid.
        other, _ = committed_checkout(tmp_path / "other")
        launcher = tmp_path / "launcher.py"
        launcher.write_text("import sys\nsys.path.insert(0, " + repr(str(other)) + ")\nfrom nanoxpu.cli import main\nmain()\n", encoding="utf-8")
        prefix = [sys.executable, str(launcher)]
    if case == "dirty":
        (checkout / "nanoxpu/new_module.py").write_text("value = 1\n", encoding="utf-8")
    payload = {"action": "preflight", "name": "identity", "role_id": "worker",
               "expected_revision": "0" * 40 if case == "revision" else revision,
               "spec": {"id": "worker", "role": "colocated", "ip": "127.0.0.1", "advertise_ip": "127.0.0.1",
                        "argv": [*prefix, "_worker", str(tmp_path)], "env": {}, "workdir": str(checkout),
                        "http_port": free_port(), "distributed_port": None, "kv_port": None, "gpus": [], "instance_id": "worker"}}
    result = helper(tmp_path / "state", payload, check=False, cwd=checkout)
    assert result["ok"] is (case == "clean"), result
    if case == "clean":
        assert result["result"]["runtime"]["worker"] == result["result"]["runtime"]["helper"]
    else:
        assert "clean profile checkout" in result["error"]
    assert not list((tmp_path / "state").rglob("state.json"))
