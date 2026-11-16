"""Atomic publication keeps the previous ledger valid during Windows sharing conflicts."""
import os
import threading

import pytest
import psutil

from nanoxpu.deployment import state


@pytest.mark.parametrize("after", [psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD, "gone", psutil.STATUS_RUNNING])
def test_identity_read_racing_exit_does_not_hide_live_access_denial(monkeypatch, after):
    class Process:
        pid = 100

        def __init__(self):
            self.reading_identity = False

        def is_running(self):
            return True

        def status(self):
            if not self.reading_identity:
                return psutil.STATUS_RUNNING
            if after == "gone":
                raise psutil.NoSuchProcess(self.pid)
            return after

        def environ(self):
            self.reading_identity = True
            raise psutil.AccessDenied(self.pid)

    process = Process()
    monkeypatch.setattr(state.psutil, "Process", lambda pid: process)
    identity = {"pid": process.pid, "owner": "test-owner"}
    if after == psutil.STATUS_RUNNING:
        with pytest.raises(psutil.AccessDenied):
            state.owned_process(identity)
    else:
        assert state.owned_process(identity) is None


@pytest.mark.parametrize("child_exits", [True, False])
def test_shutdown_continues_past_zombie_but_rejects_live_survivor(monkeypatch, child_exits):
    identities = {pid: {"pid": pid, "owner": "test-owner"} for pid in (100, 101)}
    killed = []

    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.dead = False

        def children(self, recursive):
            return [processes[101]]

        def status(self):
            return psutil.STATUS_ZOMBIE if self.dead else psutil.STATUS_RUNNING

        def terminate(self):
            pass  # Both processes ignore TERM, as in the real descendant test.

        def kill(self):
            killed.append(self.pid)
            self.dead = self.pid == 100 or child_exits

        def wait(self, timeout):
            if self.pid == 100 and self.dead:
                return 0
            # A grandchild may remain a zombie until its parent reaps it.
            raise psutil.TimeoutExpired(timeout, pid=self.pid)

    processes = {pid: Process(pid) for pid in identities}
    monkeypatch.setattr(state, "owned_process", lambda entry: None if processes[entry["pid"]].dead else processes[entry["pid"]])
    monkeypatch.setattr(state, "process_identity", lambda pid, owner: identities[pid])
    if child_exits:
        state.stop_owned_tree(identities[100], 0.1)
        assert killed == [101, 100]
    else:
        with pytest.raises(RuntimeError, match="survived forced shutdown"):
            state.stop_owned_tree(identities[100], 0.1)
        assert killed == [101]


@pytest.mark.parametrize("zombie", [True, False])
def test_descendant_identity_denial_is_only_ignored_for_zombies(monkeypatch, zombie):
    class Child:
        pid = 101

        def status(self):
            return psutil.STATUS_ZOMBIE if zombie else psutil.STATUS_RUNNING

    class Parent:
        def children(self, recursive):
            return [Child()]

        def terminate(self):
            pass

        def wait(self, timeout):
            pass

    visits = iter([Parent(), None])
    monkeypatch.setattr(state, "owned_process", lambda entry: next(visits))
    def denied(pid, owner):
        raise psutil.AccessDenied(pid)
    monkeypatch.setattr(state, "process_identity", denied)
    if zombie:
        state.stop_owned_tree({"pid": 100, "owner": "test-owner"}, 0.1)
    else:
        with pytest.raises(psutil.AccessDenied):
            state.stop_owned_tree({"pid": 100, "owner": "test-owner"}, 0.1)


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_transient_sharing_conflict_preserves_old_ledger_until_publish(tmp_path, monkeypatch, winerror):
    path = tmp_path / "state.json"
    state.atomic_json(path, {"phase": "RUNNING"})
    replace = state.os.replace
    attempts = []
    def sharing_conflict(source, target):
        attempts.append(target)
        assert state.read_json(path) == {"phase": "RUNNING"}
        if len(attempts) < 3:
            error = PermissionError("sharing denied")
            error.winerror = winerror
            raise error
        replace(source, target)
    monkeypatch.setattr(state.os, "replace", sharing_conflict)
    state.atomic_json(path, {"phase": "STOPPED"})
    assert state.read_json(path) == {"phase": "STOPPED"}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("winerror", [5, None])
def test_permanent_or_unrelated_denial_is_reported_and_old_ledger_survives(tmp_path, monkeypatch, winerror):
    path = tmp_path / "state.json"
    state.atomic_json(path, {"phase": "RUNNING"})
    error = PermissionError("denied")
    if winerror is not None:
        error.winerror = winerror
    def denied(*args):
        raise error
    clock = iter([0, 2])
    monkeypatch.setattr(state.os, "replace", denied)
    monkeypatch.setattr(state.time, "monotonic", lambda: next(clock))
    with pytest.raises(PermissionError) as caught:
        state.atomic_json(path, {"phase": "STOPPED"})
    assert caught.value is error
    assert state.read_json(path) == {"phase": "RUNNING"}
    assert not list(tmp_path.glob("*.tmp"))


def test_reader_can_observe_old_file_while_writer_publishes(tmp_path):
    path = tmp_path / "state.json"
    state.atomic_json(path, {"phase": "RUNNING"})
    reader = path.open("rb")
    timer = threading.Timer(0.1, reader.close)
    # Windows requires the reader to close first; POSIX permits replacing the name.
    timer.start()
    try:
        state.atomic_json(path, {"phase": "STOPPED"})
        assert state.read_json(path) == {"phase": "STOPPED"}
        if os.name == "nt":
            assert reader.closed
    finally:
        timer.join()
        reader.close()
