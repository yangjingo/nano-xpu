"""Source-side leases. All pin/release operations run on the owner engine thread."""
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue, Empty
import threading
import time
from typing import Callable

from nanoxpu.kv.descriptor import KVDescriptor


class LeaseCapacityError(RuntimeError):
    pass


@dataclass
class Lease:
    descriptor: KVDescriptor
    release: Callable[[], None]
    fence: Callable[[], bool]
    state: str = "EXPORTED"
    reading: bool = False
    acknowledged: bool = False


class LeaseTable:
    def __init__(self, max_leases: int, max_pinned_bytes: int, clock=time.time):
        if max_leases < 1 or max_pinned_bytes < 1:
            raise ValueError("lease budgets must be positive")
        self.max_leases = max_leases
        self.max_pinned_bytes = max_pinned_bytes
        self.clock = clock
        self.owner = threading.get_ident()
        self.leases = {}
        self.released = OrderedDict()
        self.events = Queue(maxsize=128)
        self.healthy = True

    def _owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("KV ownership can only change on the engine thread")

    @property
    def pinned_bytes(self):
        return sum(lease.descriptor.pinned_bytes for lease in self.leases.values())

    def export(self, descriptor: KVDescriptor, *, pin, release, fence):
        self._owner()
        self.require_capacity(descriptor.pinned_bytes)
        if descriptor.lease_id in self.leases or descriptor.lease_id in self.released:
            raise ValueError("lease ID cannot be reused")
        if descriptor.expires_at <= self.clock():
            raise ValueError("cannot export an expired lease")
        pin()
        self.leases[descriptor.lease_id] = Lease(descriptor, release, fence)

    def require_capacity(self, pinned_bytes):
        self._owner()
        if not self.healthy:
            raise RuntimeError("worker is quarantined; restart or validated transport shutdown required")
        if len(self.leases) >= self.max_leases or self.pinned_bytes + pinned_bytes > self.max_pinned_bytes:
            raise LeaseCapacityError("pinned KV budget exhausted")

    def post(self, event: str, lease_id: str, generation: str):
        if event not in ("begin_read", "read_finished", "ack", "cancel"):
            raise ValueError("unknown lease event")
        # The transport callback never changes allocator references itself.
        future = Future()
        self.events.put_nowait(((event, lease_id, generation), future))
        return future

    def handle(self, event: str, lease_id: str, generation: str):
        self._owner()
        if event not in ("begin_read", "read_finished", "ack", "cancel"):
            raise ValueError("unknown lease event")
        if lease_id in self.released:
            if self.released[lease_id] != generation:
                raise ValueError("stale generation")
            if event == "begin_read":
                raise ValueError("lease was released")
            return
        lease = self.leases.get(lease_id)
        if lease is None or lease.descriptor.generation != generation:
            raise ValueError("unknown lease or stale generation")
        if event == "begin_read":
            if not self.healthy or lease.state != "EXPORTED" or self.clock() >= lease.descriptor.expires_at:
                raise ValueError("lease is not available for a new read")
            lease.reading = True
            lease.state = "READING"
        elif event == "read_finished":
            if lease.state not in ("READING", "ACKED", "QUARANTINED"):
                raise ValueError("read completion without an active transfer")
            lease.reading = False
            if lease.acknowledged:
                self._release(lease_id)
        elif event == "ack":
            if lease.state == "EXPORTED":
                raise ValueError("ACK before transfer was admitted")
            lease.acknowledged = True
            if lease.reading:
                lease.state = "ACKED"
            else:
                self._release(lease_id)
        else:
            self._cancel(lease_id)

    def _release(self, lease_id):
        lease = self.leases[lease_id]
        if lease.reading:
            raise RuntimeError("cannot release source KV while a transfer may read it")
        try:
            lease.release()
        except Exception:
            lease.state = "QUARANTINED"
            self.healthy = False
            raise
        del self.leases[lease_id]
        self.released[lease_id] = lease.descriptor.generation
        while len(self.released) > 1024:
            self.released.popitem(last=False)

    def _cancel(self, lease_id):
        lease = self.leases[lease_id]
        if lease.reading:
            try:
                fenced = lease.fence()
            except Exception:
                fenced = False
            if not fenced:
                lease.state = "QUARANTINED"
                self.healthy = False
                return
            lease.reading = False
        self._release(lease_id)

    def tick(self):
        """Called periodically by the engine owner, including when no HTTP request arrives."""
        self._owner()
        errors = []
        for _ in range(128):
            try:
                event, future = self.events.get_nowait()
            except Empty:
                break
            if not future.set_running_or_notify_cancel():
                continue
            try:
                self.handle(*event)
                future.set_result(None)
            except ValueError as exc:
                errors.append(str(exc))
                future.set_exception(exc)
            except BaseException as exc:
                future.set_exception(exc)
                raise
        for lease_id, lease in list(self.leases.items()):
            if self.clock() >= lease.descriptor.expires_at:
                self._cancel(lease_id)
        return errors

    def close(self):
        self._owner()
        for lease_id in list(self.leases):
            self._cancel(lease_id)
        if self.leases:
            raise RuntimeError("KV readers could not be fenced; pinned allocations remain quarantined")
