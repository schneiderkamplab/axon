# brainsurgery — Legacy Repository

`confidence: high` `source: local checkout + git log` `last-confirmed: 2026-09-25`

## Overview

`brainsurgery` is the predecessor repository to `axon`. It was the original
home of the Axon DSL compiler and the SYNAPSE runner before the project was
split into the standalone `axon` repo.

- **Local path**: `/work/dfm/jacobwashere/brainsurgery`
- **Remote**: `github.com/schneiderkamplab/brainsurgery.git`
- **Active branch**: `clean-push-2` (HEAD: `c4ff5e1`, 2026-07-30)
- **Commit span**: 2026-03-06 → 2026-07-30 (347 commits on HEAD, 400 across all branches)
- **superseded-by**: `axon` repo (`github.com/schneiderkamplab/axon.git`)

## Relationship to axon

| Aspect | brainsurgery (legacy) | axon (current) |
|--------|----------------------|----------------|
| Top-level package | `brainsurgery/` | `synapse/` |
| Axon compiler package | `brainsurgery/synapse/axon/` | `synapse/axon/` |
| Runner/benchmark | `brainsurgery/synapse/axon_test.py` etc. | `synapse/axon_test.py` etc. |
| Last commit | 2026-07-30 | 2026-09-20 |
| Repo size (excl. .git) | ~4.2 TB (contains model weights) | ~7.4 MB |

The `axon` repo was carved out of `brainsurgery` — the `synapse/axon/`
compiler tree and `synapse/` runner modules were promoted to the top-level
`synapse/` package. The `brainsurgery/` package itself (core, CLI, transforms,
expressions, web UI, etc.) was dropped in the `axon` split.

## Structure

```
brainsurgery/                  # top-level Python package
  __init__.py
  core/                        # tensor surgery engine
  cli/                         # CLI entrypoint
  transforms/                  # checkpoint transform pipelines
  expressions/                 # OLY expression language
  io/                          # model I/O
  engine/                      # execution engine
  algorithms/                  # surgical algorithms
  web/                         # web UI
  synapse/                     # Axon DSL + runner (→ became axon's synapse/)
    axon/                      # compiler: parse, typecheck, codegen, graph_ir, …
    axon_test.py               # model integration test harness
    axon_benchmark.py           # benchmark runner
    builtins/                  # Axon builtin definitions
    models/                    # Axon model definitions
    ops/                       # op definitions
```

## Backends (shared with axon)

torch, triton, jax, mlx, tinygrad, vllm — all present under
`brainsurgery/synapse/axon/codegen2_<backend>/`.

## Notes

- The `brainsurgery` checkout is very large (~4.2 TB) due to model weights
  stored under `models/` and `log/`. Do not `du` or glob it casually.
- The `brainsurgery` repo has a `wiki/` directory of its own; it is separate
  from this wiki and should not be treated as canonical for `axon`.
- The remote URL in the brainsurgery checkout contains an embedded token;
  do not copy it into any wiki page or config file.
- Branches of interest in the legacy repo: `origin/axon`, `origin/axon-refactor`,
  `origin/axon-reimplement`, `origin/synapse` — these capture intermediate
  states of the brainsurgery→axon transition.
