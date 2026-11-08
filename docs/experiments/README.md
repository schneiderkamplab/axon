# Experiments

Artifacts of the Ettin trace study. Two agents each wrote an implementation of the Ettin
encoder and decoder suite ("Seq vs Seq: An Open Suite of Paired Encoders and Decoders",
Weller et al., 2025) on 2026-10-28: one from scratch in MLX, one in Axon compiled to MLX.
The two implementations were then benchmarked against each other and against
HF transformers.

| Entry | What it is |
|---|---|
| `traces/` | Anonymized session transcripts of both runs, added by PR #10. |
| `ettin-mlx/` | The raw-MLX implementation, copied from its own git repository at commit `c2d87fd`. |

The Axon-MLX implementation is part of the repository itself:
`synapse/models/ettin/` and `scripts/ettin_axon/`, added by PR #11.

## `ettin-mlx/`

- **Source.** The local `ettin-mlx` git repository at commit
  `c2d87fd2ebe4ce04297089b2d77ed8ed24fa22f9` (2026-10-28). This is the revision that the Ettin
  benchmark ran against.
- **How it was copied.** `git archive c2d87fd`, so the folder is the tracked tree of that commit
  (32 files). The single difference is one removed `.gitignore` line that named a local tool.
- **Not included.** Files that are git-ignored in the source: the HF reference outputs
  `tests/data/*.npz` (about 678 MB; regenerate with `tools/make_reference.py`),
  `tests/data/parity_report.json` and `bench_results/`. The project's own `README.md` refers to
  these files.
- **Using it.** The folder is a self-contained `uv` project (Python 3.13, `mlx>=0.32.4`, its own
  `uv.lock`):

  ```bash
  cd docs/experiments/ettin-mlx
  uv sync
  uv run pytest
  ```

  To import it from another environment, put its `src/` on the path:
  `PYTHONPATH=docs/experiments/ettin-mlx/src`.
- **Isolation.** It is not part of the `axon` package, the repository test suite
  (`testpaths = ["tests"]`) or the mypy configuration (`files = ["synapse"]`).
- **License.** Covered by the repository's MIT `LICENSE`.
