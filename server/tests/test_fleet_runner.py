"""Fleet runner tests: the seam drives a mock context-driven model.

The mock mirrors the nano-vllm model contract that Axon-generated serving
models will speak (see qwen3.py upstream and the 2b lowering plan):
``forward(input_ids, positions)`` reads ``get_context()`` and returns hidden
states; ``compute_logits`` maps them to ``[N, V]`` logits; attention modules
carry ``k_cache``/``v_cache`` attributes the runner binds to the pool.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from server.fleet.engine import LLMEngine
from server.fleet.sampling_params import SamplingParams
from server.fleet.utils.context import get_context

MODEL_CONFIG = {
    "num_key_value_heads": 8,
    "num_attention_heads": 8,
    "num_hidden_layers": 2,
    "hidden_size": 16,
    "head_dim": 8,
    "max_position_embeddings": 4096,
    "torch_dtype": "float32",
}


class MockAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])


class MockLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = MockAttention()


class MockContextModel(torch.nn.Module):
    """Hidden state = token id broadcast; logits favor (id + 1) % vocab."""

    def __init__(self, vocab: int = 8, num_layers: int = 2) -> None:
        super().__init__()
        self.vocab = vocab
        self.layers = torch.nn.ModuleList(MockLayer() for _ in range(num_layers))
        self.calls: list[dict] = []

    def forward(self, input_ids, positions):  # noqa: ANN001
        ctx = get_context()
        self.calls.append(
            {
                "is_prefill": ctx.is_prefill,
                "ids": input_ids.tolist(),
                "positions": positions.tolist(),
                "slot_mapping": ctx.slot_mapping.tolist(),
                "context_lens": None if ctx.context_lens is None else ctx.context_lens.tolist(),
                "block_tables": None if ctx.block_tables is None else ctx.block_tables.tolist(),
                "cu_seqlens_q": None if ctx.cu_seqlens_q is None else ctx.cu_seqlens_q.tolist(),
            }
        )
        return input_ids.to(torch.float32).unsqueeze(1).repeat(1, 4)

    def compute_logits(self, hidden):  # noqa: ANN001
        ids = hidden[:, 0].long()
        logits = torch.full((hidden.shape[0], self.vocab), -10.0)
        logits[torch.arange(len(ids)), (ids + 1) % self.vocab] = 10.0
        return logits


def make_engine(model, **kwargs) -> LLMEngine:
    return LLMEngine(model, model_config=dict(MODEL_CONFIG), **kwargs)


def drain(engine: LLMEngine) -> dict[int, list[int]]:
    finished: dict[int, list[int]] = {}
    while not engine.is_finished():
        for seq_id, token_ids in engine.step()[0]:
            finished[seq_id] = token_ids
    return finished


def test_multi_seq_prefill_batch_samples_each_seq_correctly() -> None:
    # All three prompts prefill in ONE batch (upstream sampler crashes here:
    # it divides [N, V] logits by [B] temperatures). The fleet runner gathers
    # each seq's last scheduled row before sampling, so every completion is
    # the per-sequence chain: favored token = (last id + 1) % vocab.
    engine = make_engine(MockContextModel(), num_kvcache_blocks=16)
    prompts = [[0, 1], [3, 4], [6, 7]]  # favored: 2.., 5.., 0..
    for prompt in prompts:
        engine.add_request(prompt, SamplingParams(temperature=1.0, max_tokens=2))
    finished = drain(engine)
    assert len(finished) == 3
    assert sorted(finished.values()) == [[0, 1], [2, 3], [5, 6]]
    # Exactly one prefill forward for the whole batch, then decodes.
    prefills = [call for call in engine.model_runner.model.calls if call["is_prefill"]]
    assert len(prefills) == 1
    assert prefills[0]["ids"] == [0, 1, 3, 4, 6, 7]


def test_runner_builds_paged_context_contract() -> None:
    engine = make_engine(MockContextModel(), num_kvcache_blocks=16)
    engine.add_request([10, 11, 12, 13], SamplingParams(temperature=1.0, max_tokens=2))
    engine.step()  # prefill
    call = engine.model_runner.model.calls[-1]
    assert call["is_prefill"] is True
    assert call["ids"] == [10, 11, 12, 13]
    assert call["positions"] == [0, 1, 2, 3]
    assert call["cu_seqlens_q"] == [0, 4]
    assert call["block_tables"] is None  # no prefix cache: flat contiguous path
    assert call["slot_mapping"] == [0, 1, 2, 3]  # block 0, one slot per token

    engine.step()  # decode: the sampled token (13+1)%8=6 is fed back at position 4
    call = engine.model_runner.model.calls[-1]
    assert call["is_prefill"] is False
    assert call["ids"] == [6]
    assert call["positions"] == [4]
    assert call["context_lens"] == [5]
    assert call["slot_mapping"] == [4]  # last_block_num_tokens - 1
    assert call["block_tables"] == [[0]]

    # slot addresses agree with block-table arithmetic for every scheduled token
    bs = engine.model_runner.block_size
    prefill_call = engine.model_runner.model.calls[0]
    for position, slot in zip(prefill_call["positions"], prefill_call["slot_mapping"]):
        assert slot == 0 * bs + position  # single block table [0]


def test_prefill_with_prefix_cache_sets_block_tables() -> None:
    engine = make_engine(MockContextModel(), num_kvcache_blocks=16)
    # Seed the cache with a >256-token prompt so block 0 completes and hashes.
    seed = list(range(300))
    engine.add_request(seed, SamplingParams(temperature=1.0, max_tokens=1))
    engine.step()
    seed_call = engine.model_runner.model.calls[-1]
    assert seed_call["block_tables"] is None  # first seq: no cache hit yet

    # Second seq shares the first 256 tokens (a full block) with the seed.
    engine.add_request(list(range(256)) + [777, 778], SamplingParams(temperature=1.0, max_tokens=1))
    engine.step()
    call = engine.model_runner.model.calls[-1]
    assert call["is_prefill"] is True
    # Only the tail is forwarded; positions continue from the cached prefix.
    assert call["ids"] == [777, 778]
    assert call["positions"] == [256, 257]
    assert call["cu_seqlens_q"] == [0, 2]
    assert call["block_tables"] is not None
    table = call["block_tables"][0]
    assert len(table) >= 2 and table[0] == 0  # reuses the seed's first block
    # The shared prefix block covers cached positions; the tail tokens land in
    # the seq's OWN next block. Assert paged addressing by the invariant:
    # slot(position) == block_table[position // block_size] * block_size
    #                 + position % block_size
    bs = engine.model_runner.block_size
    for position, slot in zip(call["positions"], call["slot_mapping"]):
        assert slot == table[position // bs] * bs + position % bs
    assert call["slot_mapping"][0] >= bs  # past the shared block's 256 slots


def test_eos_finishes_and_releases_blocks() -> None:
    engine = make_engine(MockContextModel(), num_kvcache_blocks=8, eos=5)
    engine.add_request([4], SamplingParams(temperature=1.0, max_tokens=10))
    finished, _ = engine.step()
    assert len(finished) == 1
    seq_id, token_ids = finished[0]
    # NOTE: the fleet path appends the EOS token to the completion (upstream
    # semantics), unlike the Session engine which consumes it silently.
    assert token_ids == [5]
    assert engine.is_finished()
    assert len(engine.scheduler.block_manager.free_block_ids) == 8


def test_pool_autosizes_when_not_configured() -> None:
    engine = make_engine(MockContextModel())  # num_kvcache_blocks defaults to -1
    blocks_per_seq = -(-4096 // 256)
    assert engine.config.num_kvcache_blocks == blocks_per_seq * engine.config.max_num_seqs
