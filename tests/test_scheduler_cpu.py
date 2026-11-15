"""Exercise the original scheduler without importing transformer/CUDA code."""
from types import SimpleNamespace

from nanoxpu.engine.scheduler import Scheduler
from nanoxpu.engine.sequence import Sequence
from nanoxpu.sampling_params import SamplingParams


def test_chunked_prefill_and_block_release():
    config = SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=4,
                             eos=-1, kvcache_block_size=4, num_kvcache_blocks=16)
    original_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        scheduler = Scheduler(config)
        seq = Sequence(list(range(10)), SamplingParams(max_tokens=2))
        scheduler.add(seq)
        chunks = []
        for expected in (4, 4, 2):
            seqs, is_prefill = scheduler.schedule()
            assert is_prefill
            chunks.append(seqs[0].num_scheduled_tokens)
            assert chunks[-1] == expected
            scheduler.postprocess(seqs, [42], is_prefill)
        assert seq.num_cached_tokens == 10
        assert seq.num_completion_tokens == 1
        seqs, is_prefill = scheduler.schedule()
        assert not is_prefill
        scheduler.postprocess(seqs, [43], is_prefill)
        assert scheduler.is_finished()
        assert not scheduler.block_manager.used_block_ids
        assert all(block.ref_count == 0 for block in scheduler.block_manager.blocks)
    finally:
        Sequence.block_size = original_size
