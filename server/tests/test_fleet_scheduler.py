"""Fleet engine bookkeeping tests: the vendored nano-vllm scheduler core.

Exercises the vendored scheduler/block_manager/sequence standalone — no model,
no tensors: prefill→decode cycling, chunked prefill budgets, prefix-cache
block sharing, and preemption when the block pool runs out.
"""

from __future__ import annotations

import pytest

from server.fleet.config import Config
from server.fleet.engine import Sequence, SequenceStatus, Scheduler
from server.fleet.sampling_params import SamplingParams

MODEL_CONFIG = {
    "num_key_value_heads": 8,
    "num_hidden_layers": 2,
    "max_position_embeddings": 4096,
    "torch_dtype": "float16",
}


def make_config(num_blocks: int = 16, max_batched: int = 4096, eos: int = -1) -> Config:
    return Config(
        model_config=MODEL_CONFIG,
        num_kvcache_blocks=num_blocks,
        max_num_batched_tokens=max_batched,
        eos=eos,
    )


def test_prefill_then_decode_cycle_finishes_at_max_tokens() -> None:
    scheduler = Scheduler(make_config())
    seq = Sequence(list(range(10)), SamplingParams(temperature=1.0, max_tokens=3))
    scheduler.add(seq)

    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and seqs == [seq]
    assert seq.num_scheduled_tokens == 10  # whole prompt in one prefill
    scheduler.postprocess(seqs, [99], True)
    assert seq.status == SequenceStatus.RUNNING
    assert seq.completion_token_ids == [99]

    for step in range(2):
        seqs, is_prefill = scheduler.schedule()
        assert not is_prefill and seqs == [seq]
        assert seq.num_scheduled_tokens == 1
        scheduler.postprocess(seqs, [100 + step], False)
    assert seq.is_finished
    assert seq.completion_token_ids == [99, 100, 101]
    assert scheduler.is_finished()
    # Blocks are released on finish; the pool is fully free again.
    assert len(scheduler.block_manager.free_block_ids) == 16


def test_prefill_respects_token_budget_chunked() -> None:
    scheduler = Scheduler(make_config(max_batched=100))
    seq = Sequence(list(range(300)), SamplingParams(temperature=1.0, max_tokens=2))
    scheduler.add(seq)

    seqs, is_prefill = scheduler.schedule()
    assert is_prefill
    assert seq.num_scheduled_tokens == 100
    scheduler.postprocess(seqs, [123], True)  # incomplete: token not appended
    assert seq.completion_token_ids == []

    while True:
        seqs, is_prefill = scheduler.schedule()
        if not is_prefill:
            break
        scheduler.postprocess(seqs, [124], True)
    assert seq.num_cached_tokens == seq.num_prompt_tokens
    # The final prefill chunk's sampled token is the first completion token.
    assert seq.completion_token_ids == [124]


def test_prefix_cache_shares_blocks_between_sequences() -> None:
    scheduler = Scheduler(make_config(num_blocks=8))
    shared = list(range(512))  # exactly two full 256-token blocks
    a = Sequence(shared + [900, 901], SamplingParams(max_tokens=1))
    scheduler.add(a)
    seqs, _ = scheduler.schedule()
    a_blocks = list(a.block_table)  # deallocate on finish clears the table
    scheduler.postprocess(seqs, [950], True)
    assert a.is_finished

    b = Sequence(shared + [800, 801], SamplingParams(max_tokens=1))
    scheduler.add(b)
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill
    # The two shared blocks hit the prefix cache: only the tail is scheduled.
    assert b.num_cached_tokens == 512
    assert b.num_scheduled_tokens == 2
    # b reuses a's block ids for the shared prefix (refcounted, not copied).
    assert b.block_table[:2] == a_blocks[:2]
    scheduler.postprocess(seqs, [960], True)
    assert b.is_finished
    # After both finish, the shared blocks return to the free pool exactly once.
    assert len(scheduler.block_manager.free_block_ids) == 8


def test_decode_preempts_when_block_pool_exhausted() -> None:
    # Two 256-token prompts = one block each; a 2-block pool fits both prefills
    # but not the extra block the first decode step needs.
    scheduler = Scheduler(make_config(num_blocks=2))
    a = Sequence(list(range(256)), SamplingParams(max_tokens=4))
    b = Sequence(list(range(256)), SamplingParams(max_tokens=4))
    scheduler.add(a)
    scheduler.add(b)
    seqs, is_prefill = scheduler.schedule()
    assert is_prefill and [s.seq_id for s in seqs] == [a.seq_id, b.seq_id]
    scheduler.postprocess(seqs, [10, 20], True)

    seqs, is_prefill = scheduler.schedule()
    assert not is_prefill
    # b was preempted (least-recently-used pop) to free a block for a's decode.
    assert [s.seq_id for s in seqs] == [a.seq_id]
    assert b.status == SequenceStatus.WAITING
    assert b.block_table == []
    assert a.status == SequenceStatus.RUNNING
    scheduler.postprocess(seqs, [11], False)
    # a holds both blocks now; finishing it releases them for b's recompute.
    while not scheduler.is_finished():
        seqs, is_prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [12], is_prefill)
    assert a.is_finished and b.is_finished


def test_config_view_maps_model_config_dict() -> None:
    import torch

    config = make_config()
    assert config.hf_config.num_key_value_heads == 8
    assert config.hf_config.max_position_embeddings == 4096
    assert config.hf_config.dtype is torch.float16
    with pytest.raises(AttributeError):
        _ = config.hf_config.nonexistent_field
