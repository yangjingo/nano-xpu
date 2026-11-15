"""Atomic publication keeps the previous ledger valid during Windows sharing conflicts."""
import os
import threading

import pytest

from nanoxpu.deployment import state


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
