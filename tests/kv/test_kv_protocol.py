import hashlib
import threading
from queue import Full

import numpy as np
import pytest

from nanoxpu.kv.descriptor import KVDescriptor, ModelLayout, parse_descriptor
from nanoxpu.kv.leases import LeaseTable, LeaseCapacityError


@pytest.fixture
def layout():
    return ModelLayout(model_fingerprint="1" * 64, tokenizer_fingerprint="2" * 64,
                       template_fingerprint="3" * 64, rope_fingerprint="4" * 64,
                       dtype="float16", layers=2, kv_heads=2, head_dim=3, block_size=4)


@pytest.fixture
def descriptor(layout):
    return KVDescriptor(cluster_id="cluster", lease_id="lease", request_id="request", generation="generation",
                        layout=layout, source_num_blocks=8, source_block_ids=(5, 2, 7),
                        strides_bytes=layout.strides(8), prompt_tokens=10, cached_tokens=9,
                        expires_at=200.0, registration_id="opaque-registration", content_digest=hashlib.sha256(b"test").hexdigest(),
                        prompt_digest=hashlib.sha256(b"prompt").hexdigest())


def test_descriptor_round_trip_and_identity(descriptor, layout):
    restored = parse_descriptor(descriptor.model_dump_json().encode())
    assert restored == descriptor
    assert restored.digest() == descriptor.digest()
    restored.validate_for(cluster_id="cluster", request_id="request", generation="generation",
                          layout=layout, now=100, digest=descriptor.digest())
    assert descriptor.pinned_bytes > descriptor.effective_bytes


@pytest.mark.parametrize("field,value", [
    ("protocol_version", 2), ("protocol_version", True),
    ("cached_tokens", 10), ("source_block_ids", (5, 2, 8)),
    ("source_block_ids", (5, 5, 7)), ("source_block_ids", (5,)),
    ("strides_bytes", (1, 2, 3, 4, 5, 6)), ("expires_at", float("inf")),
    ("content_digest", "unverified"), ("prompt_tokens", 0),
])
def test_bad_descriptor_rejected(descriptor, field, value):
    data = descriptor.model_dump()
    data[field] = value
    with pytest.raises(ValueError):
        KVDescriptor.model_validate(data)


@pytest.mark.parametrize("field,value", [
    ("cluster_id", "other"), ("request_id", "other"), ("generation", "previous"),
    ("now", 200), ("now", float("nan")), ("digest", "0" * 64),
])
def test_descriptor_cannot_be_used_for_another_request(descriptor, layout, field, value):
    args = dict(cluster_id="cluster", request_id="request", generation="generation",
                layout=layout, now=100, digest=descriptor.digest())
    args[field] = value
    with pytest.raises(ValueError):
        descriptor.validate_for(**args)


def test_model_template_and_layout_must_match(descriptor, layout):
    for field, value in (("template_fingerprint", "5" * 64), ("dtype", "float32"), ("block_size", 8)):
        changed = ModelLayout.model_validate({**layout.model_dump(), field: value})
        with pytest.raises(ValueError, match="mismatch"):
            descriptor.validate_for(cluster_id="cluster", request_id="request", generation="generation",
                                    layout=changed, now=100, digest=descriptor.digest())


def test_copy_mapping_preserves_destination_ownership_and_tail(descriptor, layout):
    source = np.arange(np.prod(layout.shape(8)), dtype=np.uint16).reshape(layout.shape(8))
    destination = np.full(layout.shape(6), 65535, dtype=np.uint16)
    mapping = descriptor.copy_spans([1, 4, 0], destination_num_blocks=6)
    assert mapping == [(5, 1, 4), (2, 4, 4), (7, 0, 1)]
    for src, dst, tokens in mapping:
        destination[:, :, dst, :tokens] = source[:, :, src, :tokens]
    assert np.array_equal(destination[:, :, 1], source[:, :, 5])
    assert np.all(destination[:, :, 0, 1:] == 65535)
    assert np.all(destination[:, :, 2] == 65535)


def test_single_prompt_token_imports_no_prefix(descriptor, layout):
    data = descriptor.model_dump()
    data.update(prompt_tokens=1, cached_tokens=0, source_block_ids=())
    empty = KVDescriptor.model_validate(data)
    assert empty.copy_spans([], 1) == []
    assert empty.pinned_bytes == empty.effective_bytes == 0


@pytest.mark.parametrize("blocks,capacity", [([1, 1, 2], 8), ([1, 2], 8), ([1, 2, 8], 8), ([True, 2, 3], 8)])
def test_destination_reservation_bounds(descriptor, blocks, capacity):
    with pytest.raises(ValueError):
        descriptor.copy_spans(blocks, capacity)


def test_descriptor_size_limit():
    with pytest.raises(ValueError, match="64 KiB"):
        parse_descriptor(b" " * (65536 + 1))


def publish(table, descriptor, events, fenced=True):
    table.export(descriptor, pin=lambda: events.append("pin"), release=lambda: events.append("release"),
                 fence=lambda: events.append("fence") or fenced)


@pytest.mark.parametrize("ack_first", [False, True])
def test_release_requires_ack_and_completed_read(descriptor, ack_first):
    events = []
    table = LeaseTable(2, 100000, clock=lambda: 100)
    publish(table, descriptor, events)
    table.handle("begin_read", "lease", "generation")
    ordered = ["ack", "read_finished"] if ack_first else ["read_finished", "ack"]
    table.handle(ordered[0], "lease", "generation")
    assert events == ["pin"]
    table.handle(ordered[1], "lease", "generation")
    table.handle("ack", "lease", "generation")
    table.handle("read_finished", "lease", "generation")
    assert events == ["pin", "release"]
    assert table.pinned_bytes == 0


def test_lease_budget_rejected_before_pin(descriptor):
    events = []
    table = LeaseTable(1, descriptor.pinned_bytes - 1, clock=lambda: 100)
    with pytest.raises(LeaseCapacityError):
        publish(table, descriptor, events)
    assert not events
    table = LeaseTable(1, 100000, clock=lambda: 100)
    publish(table, descriptor, events)
    other = KVDescriptor.model_validate({**descriptor.model_dump(), "lease_id": "another"})
    with pytest.raises(LeaseCapacityError):
        publish(table, other, events)
    table.close()


def test_expiration_without_new_requests_releases_only_idle_kv(descriptor):
    clock = [100]
    events = []
    table = LeaseTable(1, 100000, clock=lambda: clock[0])
    publish(table, descriptor, events)
    clock[0] = 200
    assert table.tick() == []
    assert events == ["pin", "release"]


def test_expiration_while_reading_requires_fence(descriptor):
    clock = [100]
    events = []
    table = LeaseTable(1, 100000, clock=lambda: clock[0])
    publish(table, descriptor, events, fenced=False)
    table.handle("begin_read", "lease", "generation")
    clock[0] = 200
    table.tick()
    assert "release" not in events
    assert table.leases["lease"].state == "QUARANTINED"
    assert table.pinned_bytes == descriptor.pinned_bytes
    assert not table.healthy
    with pytest.raises(RuntimeError, match="quarantined"):
        table.close()
    # A later transport completion proves that the memory is no longer being read.
    table.handle("read_finished", "lease", "generation")
    table.tick()
    assert events.count("release") == 1
    assert not table.healthy  # A transport failure still requires a restart.


def test_cancel_fences_before_release(descriptor):
    events = []
    table = LeaseTable(1, 100000, clock=lambda: 100)
    publish(table, descriptor, events)
    table.handle("begin_read", "lease", "generation")
    table.handle("cancel", "lease", "generation")
    assert events == ["pin", "fence", "release"]


def test_wrong_generation_and_out_of_order_ack(descriptor):
    table = LeaseTable(1, 100000, clock=lambda: 100)
    events = []
    publish(table, descriptor, events)
    with pytest.raises(ValueError, match="generation"):
        table.handle("cancel", "lease", "old")
    with pytest.raises(ValueError, match="ACK before"):
        table.handle("ack", "lease", "generation")
    assert events == ["pin"]
    table.close()


def test_transport_callback_queues_to_owner_thread(descriptor):
    events = []
    table = LeaseTable(1, 100000, clock=lambda: 100)
    publish(table, descriptor, events)
    wrong_thread = []
    def producer():
        try:
            table.handle("cancel", "lease", "generation")
        except RuntimeError:
            wrong_thread.append(True)
        table.post("cancel", "lease", "generation")
    thread = threading.Thread(target=producer)
    thread.start()
    thread.join(timeout=2)
    assert wrong_thread == [True]
    assert events == ["pin"]
    table.tick()
    assert events == ["pin", "release"]


def test_control_queue_is_bounded():
    table = LeaseTable(1, 1000)
    for _ in range(128):
        table.post("ack", "unknown", "generation")
    with pytest.raises(Full):
        table.post("ack", "unknown", "generation")
    assert len(table.tick()) == 128
