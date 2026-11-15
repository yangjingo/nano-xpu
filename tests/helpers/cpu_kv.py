"""Explicit CPU model/transport fixtures. Neither implements Qwen nor NIXL."""
from dataclasses import dataclass
from types import SimpleNamespace
from nanoxpu.engine.scheduler import Scheduler
from nanoxpu.engine.sequence import Sequence
from nanoxpu.sampling_params import SamplingParams
import hashlib
import threading

import numpy as np

from nanoxpu.kv.descriptor import ModelLayout, parse_descriptor


class CPUModel:
    eos = 0

    def __init__(self, blocks=32, eos_after=None):
        self.num_blocks, self.eos_after = blocks, eos_after
        self.layout = ModelLayout(model_fingerprint="1" * 64, tokenizer_fingerprint="2" * 64,
                                  template_fingerprint="3" * 64, rope_fingerprint="4" * 64,
                                  dtype="float32", layers=2, kv_heads=1, head_dim=2, block_size=4)
        self.runs = []
        self.closed = False
        self.compute_error = False
        self.fence_allowed = True

    def start(self, config):
        self.owner = threading.get_ident()
        self.storage = np.full(self.layout.shape(self.num_blocks), -12345, dtype=np.float32)

    def tokenize(self, request):
        return [ord(char) for message in request.messages for char in message.content]

    def run(self, seq, is_prefill):
        assert threading.get_ident() == self.owner
        if self.compute_error:
            raise RuntimeError("injected compute failure")
        self.runs.append((seq.num_cached_tokens, seq.num_scheduled_tokens, is_prefill, list(seq.block_table)))
        for position in range(seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens):
            block, offset = seq.block_table[position // self.layout.block_size], position % self.layout.block_size
            self.storage[:, :, block, offset] = seq.token_ids[position]
        # A result depends on every prior KV value: missing or stale imported bytes change the answer.
        checksum = 0
        for position in range(seq.num_cached_tokens + seq.num_scheduled_tokens):
            block, offset = seq.block_table[position // self.layout.block_size], position % self.layout.block_size
            checksum += int(self.storage[0, 0, block, offset, 0, 0])
        if self.eos_after is not None and seq.num_completion_tokens + 1 >= self.eos_after:
            return self.eos
        return checksum % 97 + 1

    def synchronize(self):
        assert threading.get_ident() == self.owner

    def content_digest(self, blocks, cached):
        digest = hashlib.sha256()
        for index, block in enumerate(blocks):
            count = min(self.layout.block_size, cached - index * self.layout.block_size)
            digest.update(self.storage[:, :, block, :count].tobytes())
        return digest.hexdigest()

    def decode(self, tokens):
        return ",".join(map(str, tokens))

    def close(self):
        self.closed = True

    def cancel_and_fence(self, timeout):
        return self.fence_allowed


class CPUTransfer:
    def __init__(self, published, destination, spans, options):
        self.published, self.destination, self.spans = published, destination, spans
        self.options = options
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.written = threading.Event()
        self.visible = options.get("visible_gate", threading.Event())
        if "visible_gate" not in options:
            self.visible.set()
        self.fence_allowed = options.get("fence_allowed", True)
        self.error = None
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        with self.lock:
            if self.cancelled.is_set():
                self.finished.set()
                return
            self.thread = threading.Thread(target=self.copy, daemon=True)
            self.thread.start()

    def copy(self):
        try:
            with self.lock:
                if self.cancelled.is_set():
                    return
                if self.options.get("copy_error"):
                    raise RuntimeError("injected copy failure")
                for source, target, count in self.spans:
                    self.destination[:, :, target, :count] = self.published.storage[:, :, source, :count]
                if self.options.get("corrupt") and self.spans:
                    self.destination[:, :, self.spans[0][1], 0] += 1
            self.written.set()
            while not self.visible.wait(0.01):
                if self.cancelled.is_set():
                    raise RuntimeError("copy cancelled before compute visibility")
        except Exception as exc:
            self.error = exc
        finally:
            # No wait on the P owner from a transfer thread: P may be fencing this very thread.
            self.published.post("read_finished", self.published.descriptor.lease_id, self.published.descriptor.generation)
            self.finished.set()

    def wait_compute_visible(self, timeout):
        if not self.finished.wait(timeout):
            raise TimeoutError("copy is not compute-visible")
        if self.error:
            raise self.error
        if self.cancelled.is_set():
            raise RuntimeError("copy was cancelled")

    def cancel_and_fence(self, timeout):
        self.cancelled.set()
        with self.lock:
            if self.thread is None:
                self.finished.set()
        return self.fence_allowed and self.finished.wait(timeout)


@dataclass
class Published:
    descriptor: object
    storage: object
    post: object
    config: object
    transfer: CPUTransfer | None = None


class CPUHub:
    def __init__(self):
        self.regions = {}
        self.lock = threading.RLock()


class CPUConnector:
    kind = "cpu-test"
    version = "cpu-test-v1"

    def __init__(self, hub, **options):
        self.hub, self.options = hub, options
        self.transfers = {}
        self.last_transfer = None

    def start(self, config):
        self.config = config

    def publish(self, descriptor, storage, post):
        with self.hub.lock:
            self.hub.regions[descriptor.lease_id] = Published(descriptor, storage, post, self.config)

    def unpublish(self, lease_id):
        with self.hub.lock:
            self.hub.regions.pop(lease_id, None)

    def fence_source(self, lease_id):
        with self.hub.lock:
            region = self.hub.regions.get(lease_id)
            ticket = region.transfer if region else None
        return ticket is None or ticket.cancel_and_fence(0.1)

    def describe(self, metadata):
        region = self.hub.regions[metadata.nano.lease_id]
        if (metadata.remote_engine_id, metadata.remote_host, metadata.remote_port) != (
            region.config.instance_id, region.config.advertise_ip or region.config.host, region.config.kv_port
        ):
            raise ValueError("endpoint does not identify this registered peer")
        return parse_descriptor(region.descriptor.model_dump_json().encode())

    def pull(self, descriptor, storage, spans):
        with self.hub.lock:
            region = self.hub.regions[descriptor.lease_id]
            if region.transfer is not None:
                raise ValueError("attempt already has a reader; no replay")
            ticket = CPUTransfer(region, storage, spans, self.options)
            region.transfer = ticket
        future = region.post("begin_read", descriptor.lease_id, descriptor.generation)
        try:
            future.result(timeout=3)
        except BaseException:
            future.cancel()
            ticket.cancel_and_fence(0)
            raise
        self.transfers[descriptor.lease_id] = ticket
        self.last_transfer = ticket
        ticket.start()
        return ticket

    def ack(self, descriptor):
        region = self.hub.regions[descriptor.lease_id]
        if self.options.get("ack_error"):
            raise OSError("injected ACK delivery failure")
        region.post("ack", descriptor.lease_id, descriptor.generation).result(timeout=3)
        self.transfers.pop(descriptor.lease_id, None)

    def cancel(self, descriptor):
        region = self.hub.regions.get(descriptor.lease_id)
        if region:
            region.post("cancel", descriptor.lease_id, descriptor.generation).result(timeout=3)
        ticket = self.transfers.get(descriptor.lease_id)
        if ticket is not None and ticket.cancel_and_fence(0):
            self.transfers.pop(descriptor.lease_id, None)

    def close(self):
        for ticket in self.transfers.values():
            if not ticket.cancel_and_fence(1):
                raise RuntimeError("CPU fixture transfer cannot be fenced")
        self.transfers.clear()


def baseline(text, budget, eos_after=None):
    model = CPUModel(eos_after=eos_after)
    model.start(None)
    scheduler = Scheduler(SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=3, eos=0,
                                          kvcache_block_size=4, num_kvcache_blocks=32))
    seq = Sequence([ord(char) for char in text], SamplingParams(max_tokens=budget))
    seq.block_size = 4
    scheduler.add(seq)
    while not seq.is_finished:
        batch, prefill = scheduler.schedule()
        scheduler.postprocess(batch, [model.run(seq, prefill)], prefill)
    return model.decode(seq.completion_token_ids)
