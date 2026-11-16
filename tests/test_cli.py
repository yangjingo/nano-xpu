"""Public configuration entry and lifecycle dispatch, without SSH or model execution."""
import json
import subprocess
import sys

import pytest

from nanoxpu.cli import main
from nanoxpu.deployment import cli


CONFIG = "cluster.yaml"  # Invalid combinations must fail before opening the file.


@pytest.mark.parametrize("arguments", [
    [], ["deploy", "plan"], ["serve"], ["serve", "/models/qwen"], ["serve", "up"],
    ["serve", "-f", str(CONFIG), "--name", "wrong"],
    ["serve", "plan", "-f", str(CONFIG), "--detach"],
    ["serve", "plan", "-f", str(CONFIG), "--state-dir", "state"],
    ["serve", "status"], ["serve", "status", "--name", "demo", "-f", str(CONFIG)],
    ["serve", "down", "--name", "demo", "--detach"],
    ["serve", "status", "--name", "demo", "--role", "prefill"],
])
def test_invalid_or_retired_entry_never_creates_controller(arguments, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid CLI reached controller")
    monkeypatch.setattr(cli, "Controller", unexpected)
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "Traceback" not in error
    if arguments == ["serve", "/models/qwen"] or arguments[:1] == ["deploy"]:
        assert "nano-xpu serve -f" in error


@pytest.mark.parametrize("mode", ["detach", "foreground", "degraded"])
def test_config_start_and_foreground_cleanup(mode, tmp_path, monkeypatch, capsys, example_files):
    events = []
    class Controller:
        def __init__(self, plan, *, root):
            assert plan.mode == "colocated" and plan.router is not None
            assert all(spec.argv[1] == "_worker" for spec in plan.workers)
            assert root == str(tmp_path)
        def up(self):
            events.append("up")
            return {"ready": True, "router_url": "http://127.0.0.1:8000"}
        def status(self):
            events.append("status")
            return {"ready": False}
        def down(self):
            events.append("down")
            return {"phase": "STOPPED"}
        def close(self):
            events.append("close")
    def sleep(_):
        if mode == "foreground":
            raise KeyboardInterrupt
    monkeypatch.setattr(cli, "Controller", Controller)
    monkeypatch.setattr(cli.time, "sleep", sleep)
    args = ["serve", "-f", str(example_files / "deployment/cluster-colocated.yaml"), "--state-dir", str(tmp_path)]
    if mode == "detach":
        args.append("--detach")
    if mode == "degraded":
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == 1
    else:
        main(args)
    assert events == {"detach": ["up", "close"], "foreground": ["up", "down", "close"],
                      "degraded": ["up", "status", "down", "close"]}[mode]
    response, _ = json.JSONDecoder().raw_decode(capsys.readouterr().out)
    assert response["router_url"] == "http://127.0.0.1:8000"


@pytest.mark.parametrize("action", ["status", "logs", "down"])
def test_management_loads_recorded_deployment(action, tmp_path, monkeypatch, capsys):
    calls = []
    class Controller:
        @classmethod
        def load(cls, name, *, root):
            assert (name, root) == ("demo", str(tmp_path))
            calls.append("load")
            return cls()
        def status(self):
            calls.append("status")
            return {"phase": "RUNNING"}
        def logs(self, role):
            assert role == "prefill"
            calls.append("logs")
            return {"text": "log fixture"}
        def down(self):
            calls.append("down")
            return {"phase": "STOPPED"}
        def close(self):
            calls.append("close")
    monkeypatch.setattr(cli, "Controller", Controller)
    args = ["serve", action, "--name", "demo", "--state-dir", str(tmp_path)]
    if action == "logs":
        args += ["--role", "prefill"]
    main(args)
    assert calls == ["load", action, "close"]
    assert isinstance(json.loads(capsys.readouterr().out), dict)


def test_public_help_has_one_start_entry_and_no_model_imports():
    for args in (["--help"], ["serve", "--help"], ["_worker", "--help"]):
        code = "from nanoxpu.cli import main; import sys\ntry: main(sys.argv[1:])\nfinally: assert not any(n in sys.modules for n in ('torch','transformers','triton','flash_attn'))"
        run = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=15)
        assert run.returncode == 0, run.stderr
        if args == ["--help"]:
            assert "serve" in run.stdout and "_worker" not in run.stdout and "deploy" not in run.stdout
