# Scripts

Durable documentation for agent-maintained scripts under `scripts/`: purpose,
CLI, inputs/outputs, ownership. One section per script; keep entries current
when the script changes (`wiki/AGENTS.md` policy at the repo root).

## `materialize_privacy_filter_opf.py`

- **Purpose**: write the privacy-filter checkpoint in the opf key layout the
  materialized `.axon` binds (`embedding.weight`, `block.N.attn.qkv.*`,
  `norm.scale`, `unembedding.weight`, ...), so the generic server loaders
  (`server/loading.load_mlx_model` → `from_safetensors`) consume it without
  model-specific code. The HF download ships raw HF names with separate
  q/k/v projections; the rename+concat mapping is
  `stream_privacy_filter_axon._convert_hf_to_opf_state_dict` (the
  parity-tested loader).
- **CLI**: `python scripts/materialize_privacy_filter_opf.py
  [--model-dir DIR] [--out DIR]` — defaults: model dir resolved by the
  stream harness (`models/privacy-filter`), output `<model-dir>/opf/`.
- **Inputs**: the HF checkpoint dir (`model.safetensors`, `config.json`,
  tokenizer files, `viterbi_calibration.json`).
- **Outputs**: `<out>/model.safetensors` (opf keys, bf16 preserved) plus
  copied metadata (`config.json`, `tokenizer*.json`, `chat_template*`,
  `special_tokens_map.json`, `viterbi_calibration.json`).
- **Depends-on**: `stream_privacy_filter_axon.py` (mapping + model-dir
  resolution), `mx.save_safetensors`.
- **Run when**: after re-downloading or replacing the HF checkpoint, or when
  the `.axon`'s bindings change.
- **Ownership**: agent-maintained; introduced 2026-11-20 with the
  `/v1/classify` serving work (see `log.md` same-day entry). The output dir
  is a build artifact — safe to delete and regenerate.
