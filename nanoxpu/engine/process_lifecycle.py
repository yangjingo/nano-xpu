"""Bounded cleanup of owned TP processes, also callable from the HTTP watchdog.

No accelerator imports: these paths are tested using real CPU child processes.
"""
import threading
import time

_groups = set()
_registry_lock = threading.Lock()


class WorkerProcesses:
    def __init__(self):
        self.processes = []
        self._lock = threading.RLock()
        with _registry_lock:
            _groups.add(self)

    def start(self, process):
        with self._lock:
            try:
                process.start()
                self.processes.append(process)
            except BaseException:
                self.stop(timeout=1)
                raise
        return process

    def stop(self, timeout=5, graceful=False):
        """Use shared deadlines per phase; retain handles if cleanup fails."""
        if timeout < 0:
            raise ValueError("timeout must be nonnegative")
        with self._lock:
            if graceful:
                self._join_until(time.monotonic() + timeout)
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            self._join_until(time.monotonic() + timeout)
            for process in self.processes:
                if process.is_alive():
                    process.kill()
            self._join_until(time.monotonic() + max(timeout, 1))
            self.processes[:] = [p for p in self.processes if p.is_alive()]
            if self.processes:
                raise RuntimeError("TP worker cleanup failed; processes are still alive")
            with _registry_lock:
                _groups.discard(self)

    def _join_until(self, deadline):
        for process in self.processes:
            process.join(timeout=max(0, deadline - time.monotonic()))


def stop_all_workers(timeout=1):
    # Do not hold the registry lock while joining a group's children.
    with _registry_lock:
        groups = list(_groups)
    failures = []
    for group in groups:
        try:
            group.stop(timeout=timeout)
        except Exception as exc:
            failures.append(str(exc))
    if failures:
        raise RuntimeError("; ".join(failures))
