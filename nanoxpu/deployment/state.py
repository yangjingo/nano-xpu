"""Atomic, user-private ledgers and verifiable process identities."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

import psutil

OWNER_ENV = "NANO_XPU_PROCESS_OWNER"


def state_root():
    return Path(os.environ.get("NANO_XPU_STATE_DIR", Path.home() / ".local/state/nano-xpu"))


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", value):
        raise ValueError("invalid deployment state name")
    return value


def private_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.chmod(0o700)
    return path


def atomic_json(path, data):
    path = Path(path)
    private_dir(path.parent)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, sort_keys=True, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        deadline = time.monotonic() + 1
        while True:
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                # Windows readers may briefly deny rename/delete sharing. Keep the
                # old ledger intact; permanent denial must still reach the caller.
                if getattr(exc, "winerror", None) not in (5, 32, 33) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


@contextmanager
def file_lock(path, timeout=10):
    path = Path(path)
    private_dir(path.parent)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    stream = os.fdopen(descriptor, "r+b", buffering=0)
    if path.stat().st_size == 0:
        stream.write(b"0")
    deadline = time.monotonic() + timeout
    acquired = False
    try:
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("deployment state is locked by another operation")
                time.sleep(0.05)
        yield
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def command_hash(argv):
    return hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()


def process_identity(pid, owner):
    process = psutil.Process(pid)
    try:
        if process.environ().get(OWNER_ENV) != owner:
            raise RuntimeError("process ownership marker does not match")
        return {"pid": pid, "created_at": process.create_time(), "user": process.username(),
                "command_hash": command_hash(process.cmdline()), "owner": owner}
    except psutil.AccessDenied:
        # A process can exit after our status check, making /proc identity fields
        # unreadable. Only a confirmed exit overrides denial for a live process.
        if process.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
            raise psutil.ZombieProcess(pid) from None
        raise


def owned_process(identity):
    """Return None for an exited process; refuse to signal a reused or foreign PID."""
    if not identity:
        return None
    try:
        process = psutil.Process(identity["pid"])
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        if process_identity(process.pid, identity["owner"]) != identity:
            raise RuntimeError("process identity changed; refusing to signal this PID")
        return process
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None


def stop_owned_tree(identity, timeout):
    parent = owned_process(identity)
    if parent is None:
        return
    children = []
    for child in parent.children(recursive=True):
        try:
            # Exited descendants can remain in /proc until their parent reaps
            # them; their environment may already be unreadable.
            if child.status() == psutil.STATUS_ZOMBIE:
                continue
            children.append(process_identity(child.pid, identity["owner"]))
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
    # Give the worker time to stop computation and its own TP children first.
    parent.terminate()
    try:
        parent.wait(timeout=timeout)
    except psutil.TimeoutExpired:
        pass
    survivors = []
    for entry in [*children, identity]:
        process = owned_process(entry)
        if process is not None:
            process.terminate()
            survivors.append((entry, process))
    deadline = time.monotonic() + min(timeout, 2)
    for _, process in survivors:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except psutil.TimeoutExpired:
            pass
    for entry, _ in survivors:
        process = owned_process(entry)
        if process is not None:
            process.kill()
            try:
                process.wait(timeout=2)
            except psutil.TimeoutExpired as exc:
                # wait() cannot reap a grandchild. A zombie has stopped executing
                # and must not prevent us from reaching its still-live parent.
                if owned_process(entry) is not None:
                    raise RuntimeError("owned process survived forced shutdown; keep ledger for recovery") from exc
