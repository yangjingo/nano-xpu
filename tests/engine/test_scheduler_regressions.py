from types import SimpleNamespace

import pytest

from nanoxpu.engine.block_manager import BlockManager
from nanoxpu.engine.scheduler import Scheduler
from nanoxpu.engine.sequence import Sequence, SequenceStatus
from nanoxpu.sampling_params import SamplingParams


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)


def scheduler(blocks=16, budget=4, sequences=2):
    return Scheduler(SimpleNamespace(max_num_seqs=sequences, max_num_batched_tokens=budget,
                                    eos=99, kvcache_block_size=4, num_kvcache_blocks=blocks))


def run_to_completion(engine, token=42):
    batches = []
    for _ in range(100):
        if engine.is_finished():
            break
        batch, prefill = engine.schedule()
        batches.append((list(batch), prefill))
        assert sum(seq.num_scheduled_tokens for seq in batch) <= engine.max_num_batched_tokens
        engine.postprocess(batch, [token] * len(batch), prefill)
    assert engine.is_finished(), "scheduler did not make bounded progress"
    blocks = engine.block_manager
    assert not blocks.used_block_ids
    assert len(blocks.free_block_ids) == len(blocks.blocks)
    assert all(block.ref_count == 0 for block in blocks.blocks)
    return batches


@pytest.mark.parametrize("length", [1, 3, 4, 5, 8, 9])
def test_prompt_boundaries_and_chunk_budget(length):
    engine = scheduler()
    seq = Sequence(list(range(length)), SamplingParams(max_tokens=3))
    engine.add(seq)
    run_to_completion(engine)
    assert seq.completion_token_ids == [42, 42, 42]


def test_multiple_sequences_finish_and_release_all_references():
    engine = scheduler(blocks=12, budget=8)
    sequences = [Sequence([index] * 7, SamplingParams(max_tokens=3)) for index in range(6)]
    for seq in sequences:
        engine.add(seq)
    run_to_completion(engine)
    assert all(seq.status == SequenceStatus.FINISHED and seq.num_completion_tokens == 3 for seq in sequences)


def test_memory_pressure_preempts_and_resumes_without_duplicate_output():
    engine = scheduler(blocks=3, budget=8)
    first = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=2))
    second = Sequence([5, 6, 7, 8], SamplingParams(max_tokens=2))
    engine.add(first)
    engine.add(second)
    batch, prefill = engine.schedule()
    assert batch == [first, second] and prefill
    engine.postprocess(batch, [41, 51], prefill)
    batch, prefill = engine.schedule()
    assert batch == [first] and not prefill
    assert second.status == SequenceStatus.WAITING
    assert not second.block_table
    engine.postprocess(batch, [42], prefill)
    run_to_completion(engine, token=52)
    assert first.completion_token_ids == [41, 42]
    assert second.completion_token_ids == [51, 52]


@pytest.mark.parametrize("ignore_eos,expected", [(False, 1), (True, 3)])
def test_eos_and_length_both_release_blocks(ignore_eos, expected):
    engine = scheduler()
    seq = Sequence([1, 2], SamplingParams(max_tokens=3, ignore_eos=ignore_eos))
    engine.add(seq)
    run_to_completion(engine, token=99)
    assert seq.num_completion_tokens == expected


def test_active_prefix_is_shared_until_last_owner_releases():
    blocks = BlockManager(4, 4)
    first, second = Sequence(list(range(9))), Sequence(list(range(9)))
    blocks.allocate(first, blocks.can_allocate(first))
    first.num_scheduled_tokens = 8
    blocks.hash_blocks(first)
    assert blocks.can_allocate(second) == 2
    blocks.allocate(second, 2)
    shared = first.block_table[:2]
    assert second.block_table[:2] == shared
    assert all(blocks.blocks[index].ref_count == 2 for index in shared)
    blocks.deallocate(first)
    assert all(blocks.blocks[index].ref_count == 1 and index in blocks.used_block_ids for index in shared)
    blocks.deallocate(second)
    assert not blocks.used_block_ids
    assert len(blocks.free_block_ids) == 4


def test_idle_prefix_reuse_and_hash_collision_cannot_return_wrong_tokens(monkeypatch):
    blocks = BlockManager(4, 4)
    monkeypatch.setattr(blocks, "compute_hash", lambda tokens, prefix=-1: 123)
    first = Sequence([1, 2, 3, 4, 5])
    blocks.allocate(first, 0)
    first.num_scheduled_tokens = 4
    blocks.hash_blocks(first)
    blocks.deallocate(first)
    identical = Sequence([1, 2, 3, 4, 9])
    different = Sequence([5, 6, 7, 8, 9])
    assert blocks.can_allocate(identical) == 1
    assert blocks.can_allocate(different) == 0
    blocks.allocate(identical, 1)
    blocks.deallocate(identical)
    blocks.allocate(different, 0)
    assert different.num_cached_tokens == 0
    blocks.deallocate(different)


def test_rejected_allocation_does_not_mutate_block_ownership():
    blocks = BlockManager(2, 4)
    seq = Sequence(list(range(9)))
    assert blocks.can_allocate(seq) == -1
    assert list(blocks.free_block_ids) == [0, 1]
    assert not blocks.used_block_ids
    assert not seq.block_table
