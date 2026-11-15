"""P/D state and allocation control, shared by injected CPU tests and future device adapters.

No device adapter is selected here; production NanoEngine still refuses P/D until its device gate.
"""
import hashlib
import json
import threading
import time
from types import SimpleNamespace
from typing import Protocol
import uuid

from nanoxpu.engine.scheduler import Scheduler
from nanoxpu.engine.sequence import Sequence, SequenceStatus
from nanoxpu.kv.descriptor import KVDescriptor
from nanoxpu.kv.connector import KVConnector
from nanoxpu.kv.leases import LeaseTable
from nanoxpu.sampling_params import SamplingParams
from nanoxpu.server.engine import Completion, InvalidRequest
from nanoxpu.server.metadata import Capabilities, Fingerprints
from nanoxpu.server.schema import PrefillRequest, DecodeRequest, DecodeKV


def prompt_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


class StageModel(Protocol):
    """run returns a completed sample; exceptions require a positive fence before releasing KV."""
    layout: object
    storage: object
    num_blocks: int
    eos: int

    def start(self, config) -> None: ...
    def tokenize(self, request) -> list[int]: ...
    def run(self, seq: Sequence, is_prefill: bool) -> int: ...
    def synchronize(self) -> None: ...
    def content_digest(self, blocks, cached_tokens: int) -> str: ...
    def cancel_and_fence(self, timeout: float) -> bool: ...
    def decode(self, tokens: list[int]) -> str: ...
    def close(self) -> None: ...


class StageEngine:
    def __init__(self, config, model: StageModel, connector: KVConnector, *, lease_seconds=180, transfer_timeout=120,
                 max_leases=16, max_pinned_bytes=1024 * 1024 * 1024):
        if config.role not in ("prefill", "decode") or config.tensor_parallel_size != 1 or not config.cluster_id:
            raise ValueError("stage engine requires a deployed P/D role with TP=1")
        if lease_seconds <= transfer_timeout or transfer_timeout <= 0:
            raise ValueError("lease lifetime must exceed the positive transfer deadline")
        self.config, self.model, self.connector = config, model, connector
        self.lease_seconds, self.transfer_timeout = lease_seconds, transfer_timeout
        self.max_leases, self.max_pinned_bytes = max_leases, max_pinned_bytes
        self.scheduler = None
        self.leases = None
        self.quarantine = {}
        self.healthy = True
        self.started = False
        self.owner = None
        self.closed = False

    def _owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("stage allocations belong to the engine thread")

    def start(self):
        self.owner = threading.get_ident()
        self.model.start(self.config)
        layout = self.model.layout
        self.scheduler = Scheduler(SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=self.config.max_num_batched_tokens,
                                                   eos=self.model.eos, kvcache_block_size=layout.block_size,
                                                   num_kvcache_blocks=self.model.num_blocks))
        self.leases = LeaseTable(self.max_leases, self.max_pinned_bytes)
        self.connector.start(self.config)
        self.capabilities = Capabilities(
            transport=self.connector.kind, transport_version=self.connector.version, connector_ready=True,
            fingerprints=Fingerprints(model=layout.model_fingerprint, tokenizer=layout.tokenizer_fingerprint,
                                      template=layout.template_fingerprint, rope=layout.rope_fingerprint),
            layout_digest=hashlib.sha256(layout.model_dump_json().encode()).hexdigest(),
        )
        self.started = True

    def _sequence(self, tokens, request):
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise InvalidRequest("tokenizer must return nonempty nonnegative token IDs")
        if len(tokens) + request.max_tokens > self.config.max_model_len:
            raise InvalidRequest("prompt plus max_tokens exceeds max_model_len")
        seq = Sequence(tokens, SamplingParams(temperature=request.temperature, max_tokens=request.max_tokens))
        seq.block_size = self.model.layout.block_size
        return seq

    def _reserve(self, seq, capacity):
        try:
            self.scheduler.block_manager.reserve_private(seq, capacity)
        except ValueError as exc:
            raise InvalidRequest("request exceeds private KV capacity") from exc

    def _release(self, seq):
        for queue in (self.scheduler.waiting, self.scheduler.running):
            if seq in queue:
                queue.remove(seq)
        if seq.block_table:
            self.scheduler.block_manager.deallocate(seq)
        seq.status = SequenceStatus.FINISHED

    def complete(self, request):
        self._owner()
        if not self.started or not self.healthy or not self.leases.healthy:
            raise RuntimeError("stage engine is not available")
        if not request.request_id or request.model != self.config.served_model_name:
            raise InvalidRequest("missing attempt identity or wrong model")
        seq = self._sequence(self.model.tokenize(request), request)
        if self.config.role == "prefill" and isinstance(request, PrefillRequest):
            return self.prefill_export(seq, request)
        if self.config.role == "decode" and isinstance(request, DecodeRequest):
            return self.import_and_resume(seq, request)
        raise InvalidRequest("request stage does not match worker role")

    def prefill_export(self, seq, request):
        layout = self.model.layout
        cached = len(seq) - 1
        count = (cached + layout.block_size - 1) // layout.block_size
        pinned_bytes = 2 * layout.layers * count * layout.block_size * layout.kv_heads * layout.head_dim * layout.element_bytes
        self.leases.require_capacity(pinned_bytes)
        self._reserve(seq, len(seq))
        manager = self.scheduler.block_manager
        lease_id = uuid.uuid4().hex
        exported = False
        compute_safe = True
        try:
            self.scheduler.add(seq)
            while seq.num_cached_tokens < len(seq):
                batch, is_prefill = self.scheduler.schedule()
                compute_safe = False
                sampled = self.model.run(seq, is_prefill)
                compute_safe = True
                if seq.num_cached_tokens + seq.num_scheduled_tokens == len(seq):
                    # Do not enter normal max_tokens=1 completion: it would release the export source.
                    seq.num_cached_tokens += seq.num_scheduled_tokens
                    seq.num_scheduled_tokens = 0
                    break
                self.scheduler.postprocess(batch, [sampled], is_prefill)
            compute_safe = False
            self.model.synchronize()
            compute_safe = True
            blocks = tuple(seq.block_table[:count])
            descriptor = KVDescriptor(
                cluster_id=self.config.cluster_id, lease_id=lease_id, request_id=request.request_id,
                generation=self.config.engine_generation, layout=layout, source_num_blocks=self.model.num_blocks,
                source_block_ids=blocks, strides_bytes=layout.strides(self.model.num_blocks),
                prompt_tokens=len(seq), cached_tokens=cached, expires_at=time.time() + self.lease_seconds,
                registration_id=uuid.uuid4().hex, content_digest=self.model.content_digest(blocks, cached),
                prompt_digest=prompt_digest(seq.token_ids),
            )

            def release_export():
                self.connector.unpublish(lease_id)
                manager.release(blocks)

            self.leases.export(descriptor, pin=lambda: manager.retain(blocks), release=release_export,
                               fence=lambda: self.connector.fence_source(lease_id))
            exported = True
            self.connector.publish(descriptor, self.model.storage, self.leases.post)
            metadata = DecodeKV.model_validate({
                "do_remote_decode": False, "do_remote_prefill": True, "remote_engine_id": self.config.instance_id,
                "remote_block_ids": list(blocks), "remote_host": self.config.advertise_ip or self.config.host,
                "remote_port": self.config.kv_port,
                "nano": {"protocol_version": 1, "lease_id": lease_id, "generation": descriptor.generation,
                         "expires_at": descriptor.expires_at, "cached_tokens": cached, "descriptor_digest": descriptor.digest()},
            })
            return Completion("", len(seq), 1, "length", metadata.model_dump())
        except BaseException:
            if not compute_safe:
                compute_safe = self._fence_model()
                if not compute_safe:
                    self.healthy = False
                    self.quarantine[lease_id] = (seq, None, True)
            if exported:
                self.leases.handle("cancel", lease_id, self.config.engine_generation)
            raise
        finally:
            if compute_safe:
                self._release(seq)  # An export lease retains only the effective source blocks.

    def _fence_model(self):
        try:
            return self.model.cancel_and_fence(self.transfer_timeout)
        except Exception:
            return False

    def import_and_resume(self, seq, request):
        metadata = request.kv_transfer_params
        peers = dict(entry.split("=", 1) for entry in self.config.prefill_peers)
        if peers.get(metadata.remote_engine_id) != metadata.nano.generation:
            raise InvalidRequest("unknown prefill peer or stale generation")
        try:
            descriptor = self.connector.describe(metadata)
            descriptor.validate_for(cluster_id=self.config.cluster_id, request_id=request.request_id,
                                    generation=metadata.nano.generation, layout=self.model.layout,
                                    now=time.time(), digest=metadata.nano.descriptor_digest)
            if (descriptor.lease_id != metadata.nano.lease_id or descriptor.cached_tokens != metadata.nano.cached_tokens
                or descriptor.expires_at != metadata.nano.expires_at
                or list(descriptor.source_block_ids) != metadata.remote_block_ids
                or descriptor.prompt_tokens != len(seq) or descriptor.prompt_digest != prompt_digest(seq.token_ids)):
                raise ValueError("HTTP metadata or prompt tokens differ from the registered export")
        except (ValueError, LookupError) as exc:
            raise InvalidRequest("incompatible KV descriptor or prompt") from exc
        self._reserve(seq, len(seq) + request.max_tokens)
        transfer = None
        safe = True
        compute_safe = True
        try:
            count = len(descriptor.source_block_ids)
            spans = descriptor.copy_spans(seq.block_table[:count], self.model.num_blocks)
            transfer = self.connector.pull(descriptor, self.model.storage, spans)
            safe = False
            transfer.wait_compute_visible(self.transfer_timeout)
            safe = True
            if self.model.content_digest(seq.block_table[:count], descriptor.cached_tokens) != descriptor.content_digest:
                raise RuntimeError("imported KV content digest does not match")
            self.connector.ack(descriptor)
            seq.num_cached_tokens = descriptor.cached_tokens
            self.scheduler.add(seq)
            while not seq.is_finished:
                batch, is_prefill = self.scheduler.schedule()
                compute_safe = False
                sampled = self.model.run(seq, is_prefill)
                compute_safe = True
                self.scheduler.postprocess(batch, [sampled], is_prefill)
            generated = seq.completion_token_ids
            return Completion(self.model.decode(generated), seq.num_prompt_tokens, len(generated),
                              "stop" if generated[-1] == self.model.eos else "length")
        except BaseException:
            if not compute_safe:
                compute_safe = self._fence_model()
            if transfer is not None and not safe:
                try:
                    safe = transfer.cancel_and_fence(self.transfer_timeout)
                except Exception:
                    safe = False
            try:
                self.connector.cancel(descriptor)
            except Exception:
                pass  # Source TTL still owns its pin; failure does not prove destination safety.
            if not safe or not compute_safe:
                self.healthy = False
                self.quarantine[descriptor.lease_id] = (seq, None if safe else transfer, not compute_safe)
            raise
        finally:
            if safe and compute_safe:
                self._release(seq)

    def tick(self):
        self._owner()
        self.leases.tick()
        for lease_id, (seq, transfer, compute_pending) in list(self.quarantine.items()):
            if (transfer is None or transfer.cancel_and_fence(0)) and (not compute_pending or self.model.cancel_and_fence(0)):
                self._release(seq)
                del self.quarantine[lease_id]
        return self.healthy and self.leases.healthy

    def close(self):
        self._owner()
        if self.closed:
            return
        if self.leases is not None:
            self.leases.close()
        for lease_id, (seq, transfer, compute_pending) in list(self.quarantine.items()):
            if (transfer is None or transfer.cancel_and_fence(self.transfer_timeout)) and (not compute_pending or self._fence_model()):
                self._release(seq)
                del self.quarantine[lease_id]
        if self.quarantine:
            raise RuntimeError("destination transfers are not fenced; allocations remain quarantined")
        self.connector.close()
        self.model.close()
        self.started = False
        self.closed = True
