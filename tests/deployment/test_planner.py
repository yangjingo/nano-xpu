import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from nanoxpu.deployment.planner import plan, plan_file
from nanoxpu.deployment.schema import ClusterConfig, RuntimeProfile, load_config




@pytest.fixture
def inputs(example_files):
    config, profile = load_config(example_files / "deployment" / "cluster-colocated.yaml")
    return config.model_dump(), profile.model_dump()


def make_plan(inputs):
    config, profile = inputs
    return plan(ClusterConfig.model_validate(config), RuntimeProfile.model_validate(profile))


def test_both_rfc_examples_and_stable_plan(example_files):
    for filename in ("cluster-colocated.yaml", "cluster-pd.yaml"):
        first = plan_file(example_files / "deployment" / filename)
        second = plan_file(example_files / "deployment" / filename)
        assert first.public_dict() == second.public_dict()
        assert first.router.http_port == 8000
        assert [worker.http_port for worker in first.workers] == [8100, 8101]
        assert [worker.distributed_port for worker in first.workers] == [23000, 23001]
        assert first.unresolved  # Examples are intentionally not ready to launch real hosts.
        assert "--disable-retries" in first.router.argv
        if first.mode == "colocated":
            assert "--worker-urls" in first.router.argv
            assert all(worker.kv_port is None for worker in first.workers)
        else:
            assert "--vllm-pd-disaggregation" in first.router.argv
            assert [worker.kv_port for worker in first.workers] == [24000, 24001]


def test_one_worker_still_has_router(inputs):
    inputs[0]["workers"] = inputs[0]["workers"][:1]
    result = make_plan(inputs)
    assert result.router.url == "http://10.0.0.10:8000"
    assert result.router.argv[-2:] == ["--worker-urls", result.workers[0].url]


def test_argv_owns_each_value_and_maps_physical_gpus(inputs):
    node = inputs[0]["workers"][0]
    node.update(gpus=[3, 7], tensor_parallel_size=2, model_path="/models/a b;$(touch nope)")
    result = make_plan(inputs)
    worker = result.workers[0]
    assert worker.env == {"CUDA_VISIBLE_DEVICES": "3,7"}
    assert worker.argv[2] == "/models/a b;$(touch nope)"
    assert worker.argv[worker.argv.index("--tensor-parallel-size") + 1] == "2"
    assert worker.instance_id != result.workers[1].instance_id


def test_no_network_processes_or_writes(inputs, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("plan attempted an external side effect")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    make_plan(inputs)


@pytest.mark.parametrize("field,value", [
    ("ip", "0.0.0.0"), ("ip", "example.com"), ("ip", "::1"),
    ("ip", "224.0.0.1"), ("ip", "255.255.255.255"),
    ("gpus", []), ("gpus", [0, 0]), ("gpus", [-1]), ("gpus", [True]),
    ("tensor_parallel_size", 2), ("http_port", 0), ("http_port", 65536),
    ("http_port", True), ("kv_port", 24000), ("id", "../escape"),
    ("id", "router"), ("model_path", "relative/path"), ("advertise_ip", "0.0.0.0"),
])
def test_worker_validation(inputs, field, value):
    inputs[0]["workers"][0][field] = value
    with pytest.raises(ValueError):
        make_plan(inputs)


def test_duplicate_worker_id_and_same_host_gpu(inputs):
    original = copy.deepcopy(inputs)
    inputs[0]["workers"][1]["id"] = inputs[0]["workers"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        make_plan(inputs)
    original[0]["workers"][1]["ip"] = original[0]["workers"][0]["ip"]
    with pytest.raises(ValueError, match="assigned more than once"):
        make_plan(original)


@pytest.mark.parametrize("port_field,port", [("http_port", 8000), ("distributed_port", 8000), ("distributed_port", 8100)])
def test_ports_across_roles_and_uses(inputs, port_field, port):
    inputs[0]["workers"][0]["ip"] = inputs[0]["router"]["ip"]
    inputs[0]["workers"][0][port_field] = port
    with pytest.raises(ValueError, match="conflicting port"):
        make_plan(inputs)


def test_pd_requires_both_roles_and_tp_one(example_files):
    config, profile = load_config(example_files / "deployment" / "cluster-pd.yaml")
    data = config.model_dump()
    data["workers"] = data["workers"][:1]
    with pytest.raises(ValueError, match="role set"):
        ClusterConfig.model_validate(data)
    data = config.model_dump()
    data["workers"][0]["gpus"] = [0, 1]
    with pytest.raises(ValueError, match="TP=1"):
        ClusterConfig.model_validate(data)


@pytest.mark.parametrize("group,field,value", [
    ("timeouts", "request_seconds", 151),
    ("timeouts", "startup_seconds", 0),
    ("router", "source_revision", "latest"),
    ("engine", "gpu_memory_utilization", float("nan")),
    ("runtime", "helper_python", "/usr/bin/python;touch /tmp/nope"),
    ("ssh", "user", "-oProxyCommand=bad"),
    ("ssh", "port", "22"),
])
def test_profile_constraints(inputs, group, field, value):
    inputs[1][group][field] = value
    with pytest.raises(ValueError):
        make_plan(inputs)


def test_unknown_fields_rejected_everywhere(inputs):
    locations = [inputs[0], inputs[0]["router"], inputs[0]["workers"][0], inputs[1], inputs[1]["engine"]]
    for location in locations:
        location["typo"] = "must not be ignored"
        with pytest.raises(ValueError, match="Extra inputs"):
            make_plan(inputs)
        del location["typo"]


def test_profile_relative_to_config_and_duplicate_yaml(inputs, tmp_path):
    topology, profile = inputs
    topology["profile"] = "profile.yaml"
    (tmp_path / "profile.yaml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    source = tmp_path / "cluster.yaml"
    source.write_text(yaml.safe_dump(topology), encoding="utf-8")
    assert plan_file(source).name == topology["name"]
    source.write_text("name: a\nname: b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_config(source)


@pytest.mark.parametrize("mode", ["colocated", "pd"])
def test_cli_emits_json_and_imports_no_accelerator_stack(mode, example_files):
    code = "from nanoxpu.cli import main; import sys; main(sys.argv[1:]); assert not any(n in sys.modules for n in ('torch','transformers','triton','flash_attn'))"
    result = subprocess.run([sys.executable, "-c", code, "serve", "plan", "-f", str(example_files / "deployment" / f"cluster-{mode}.yaml")], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["mode"] == ("colocated" if mode == "colocated" else "disaggregated")


def test_invalid_cli_exits_without_traceback(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: missing-fields", encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "nanoxpu", "serve", "plan", "-f", str(path)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert "required" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("text", ["[invalid", "!!python/object/apply:os.system ['bad']", "schema_version: true\nname: demo"])
def test_malformed_unsafe_and_boolean_version_yaml_rejected(tmp_path, text):
    source = tmp_path / "invalid.yaml"
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(source)
