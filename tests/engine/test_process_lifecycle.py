import importlib.util
import multiprocessing as mp
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import pytest

from nanoxpu.engine.process_lifecycle import WorkerProcesses, stop_all_workers


def sleeper(ready, ignore_term=False):
    if ignore_term and sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        time.sleep(0.1)


def cooperative(ready, release):
    ready.set()
    release.wait(10)


def launch(group, target=sleeper, extra=()):
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    process = group.start(ctx.Process(target=target, args=(ready, *extra)))
    assert ready.wait(10)
    return process


def test_cooperative_exit_and_idempotent_stop():
    group = WorkerProcesses()
    release = mp.get_context("spawn").Event()
    try:
        process = launch(group, cooperative, (release,))
        release.set()
        group.stop(timeout=2, graceful=True)
        assert process.exitcode == 0
        group.stop(timeout=0.1)
        assert not group.processes
    finally:
        group.stop(timeout=0.2)


@pytest.mark.parametrize("ignore_term", [False, True])
def test_stuck_worker_is_reaped_with_deadline(ignore_term):
    group = WorkerProcesses()
    try:
        process = launch(group, extra=(ignore_term,))
        started = time.monotonic()
        group.stop(timeout=0.2, graceful=True)
        assert time.monotonic() - started < 4
        assert not process.is_alive()
        assert process.exitcode != 0
        if ignore_term and sys.platform != "win32":
            assert process.exitcode == -signal.SIGKILL
    finally:
        group.stop(timeout=0.2)


def test_failed_spawn_reaps_previous_workers():
    class FailedProcess:
        def start(self):
            raise RuntimeError("injected start failure")
    group = WorkerProcesses()
    try:
        process = launch(group)
        with pytest.raises(RuntimeError, match="injected"):
            group.start(FailedProcess())
        assert not process.is_alive()
        assert not group.processes
    finally:
        group.stop(timeout=0.2)


def test_watchdog_cleanup_only_targets_registered_workers():
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    unrelated = ctx.Process(target=sleeper, args=(ready,))
    unrelated.start()
    group = WorkerProcesses()
    try:
        assert ready.wait(10)
        owned = launch(group)
        stop_all_workers(timeout=0.2)
        assert not owned.is_alive()
        assert unrelated.is_alive()
    finally:
        group.stop(timeout=0.2)
        unrelated.terminate()
        unrelated.join(timeout=5)
        if unrelated.is_alive():
            unrelated.kill()
            unrelated.join(timeout=5)


class FailingModelRunner:
    def __init__(self, config, rank, events):
        if rank == 0:
            assert events[0].wait(10)
            raise RuntimeError("injected rank-zero initialization failure")
        sleeper(events)


def test_actual_llm_constructor_cleans_workers_on_initialization_failure(tmp_path, monkeypatch):
    # Execute the actual constructor; only CUDA/model dependencies are substituted.
    torch = SimpleNamespace(multiprocessing=mp)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", mp)
    monkeypatch.setitem(sys.modules, "tqdm.auto", SimpleNamespace(tqdm=lambda *args, **kwargs: None))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *args: SimpleNamespace(max_position_embeddings=4096)),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)))
    monkeypatch.setitem(sys.modules, "nanoxpu.engine.model_runner", SimpleNamespace(ModelRunner=FailingModelRunner))
    path = Path(__file__).parents[2] / "nanoxpu/engine/llm_engine.py"
    spec = importlib.util.spec_from_file_location("test_llm_lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = {process.pid for process in mp.active_children()}
    with pytest.raises(RuntimeError, match="rank-zero"):
        module.LLMEngine(str(tmp_path), tensor_parallel_size=2)
    assert {process.pid for process in mp.active_children()} == before
