<p align="center">
  <img src="docs/logos/axon-logo.png" alt="axon" width="480" />
</p>

<h3 align="center">axon</h3>

<p align="center">
  A typed declarative DSL and compiler for neural network model definitions.<br/>
  Parse → Resolve → Normalize → Elaborate → Flatten → Typecheck → Graph IR → Optimize → Codegen
</p>

---

## Quick Start

```bash
pip install -e .
```

### Optional backends

```bash
pip install -e ".[mlx]"    # Apple MLX backend
pip install -e ".[dev]"   # Development tools
```

## Usage

### Compiling GPT-2 across all backends

```bash
# 1. Resolve imports → self-contained Axon file
synapse axon-resolve synapse/models/gpt2/generic-gpt2.axon resolved.axon --force

# 2. Dump any compiler stage
synapse axon-stage-dump synapse/models/gpt2/generic-gpt2.axon parse.axon         --stage parse
synapse axon-stage-dump synapse/models/gpt2/generic-gpt2.axon resolved.axon      --stage resolve
synapse axon-stage-dump synapse/models/gpt2/generic-gpt2.axon normalized.axon   --stage normalize
synapse axon-stage-dump synapse/models/gpt2/generic-gpt2.axon flat.axon          --stage flatten
synapse axon-stage-dump synapse/models/gpt2/generic-gpt2.axon typed.axon        --stage typecheck --show-types
synapse axon-stage-dump synapse/models/gpt2/generic-gpt2.axon graph-ir.axon     --stage graph-ir-axon --optimize-graph
synapse axon-stage-dump synapse/models/gpt2/generic-gpt2.axon final.axon        --stage final --optimize-ast --optimize-graph

# 3. Visualize Graph IR as Graphviz DOT
synapse axon-graph-ir-dot synapse/models/gpt2/generic-gpt2.axon gpt2.dot --optimize-graph --force

# 4. Generate code for each backend
synapse axon-codegen-dump synapse/models/gpt2/generic-gpt2.axon gpt2_torch.py      --backend codegen2-torch
synapse axon-codegen-dump synapse/models/gpt2/generic-gpt2.axon gpt2_tinygrad.py  --backend codegen2-tinygrad
synapse axon-codegen-dump synapse/models/gpt2/generic-gpt2.axon gpt2_mlx.py       --backend codegen2-mlx
synapse axon-codegen-dump synapse/models/gpt2/generic-gpt2.axon gpt2_jax.py       --backend codegen2-jax
synapse axon-codegen-dump synapse/models/gpt2/generic-gpt2.axon gpt2_triton.py    --backend codegen2-triton

# 5. Run HF vs Axon parity test
synapse axon-test synapse/models/gpt2/generic-gpt2.axon models/gpt2 --dtype float32

# 6. Benchmark across all checkpoints
synapse axon-benchmark synapse/models/gpt2/ --axon-backend codegen2-torch --log-dir log/gpt2
```

### Compiling Gemma-3 270M

```bash
# Resolve + generate optimized PyTorch code
synapse axon-resolve synapse/models/gemma3/gemma-3-270m.axon resolved.axon --force
synapse axon-codegen-dump synapse/models/gemma3/gemma-3-270m.axon gemma3.py \
  --backend codegen2-torch --optimize-ast --optimize-graph

# Parity test against HF
synapse axon-test synapse/models/gemma3/gemma-3-270m.axon models/gemma-3-270m \
  --dtype float32 --max-len 128
```

---

## The Axon Language

An Axon file declares model architecture as typed functions with explicit
tensor shapes, parameter paths, and config-driven dimensions.

```axon
{-# CHECKPOINTS ["openai-community/gpt2"] #-}
{-# TASK "causal_lm" #-}

import Activations (gelu_new)
import Attention (attention, merge_heads, reshape_heads)
import Masking (Mask, causal_mask_for_input, mask_for_input)
import Positions (position_ids)
import Cache
import Config
import NN
import Tensor

-- Config-driven dimensions with defaults
CONTEXT_SIZE <- (Config.dim @@n_positions default=1024 :: Dim)
MODEL_DIM    <- (Config.dim @@n_embd default=768 :: Dim)
NUM_LAYERS   <- (Config.dim @@n_layer default=12 :: Dim)
NUM_HEADS    <- (Config.dim @@n_head default=12 :: Dim)

-- Type signature: input_ids, optional mask, optional KV cache, use_cache flag
gpt2_block :: Tensor[B,S,D] -> Tensor[B,1,S,K]
  -> ?CacheLayer[B,H,P,CONTEXT_SIZE,DH]
  -> (Tensor[B,S,D], ?CacheLayer[B,H,K,CONTEXT_SIZE,DH])
gpt2_block x mask past_kv = do
  x1 <- NN.layernorm@ln_1 x
  a, new_kv <- scope@attn do
    q_lin, k_lin, v_lin <- Tensor.chunk
      (NN.linear@c_attn x1 dim=(3 * MODEL_DIM) bias=true transpose=true) parts=3
    q <- reshape_heads q_lin heads=NUM_HEADS
    k <- reshape_heads k_lin heads=NUM_HEADS
    v <- reshape_heads v_lin heads=NUM_HEADS
    k, v, new_kv <- Cache.update past_kv k v
    a <- attention q k v mask
      |> merge_heads
      |> NN.linear@c_proj dim=MODEL_DIM bias=true transpose=true
    return a, new_kv
  x <- x + a
  x3 <- NN.layernorm@ln_2 x
  m <- scope@mlp do
    return NN.linear@c_fc x3 dim=(4 * MODEL_DIM) bias=true transpose=true
      |> gelu_new
      |> NN.linear@c_proj dim=MODEL_DIM bias=true transpose=true
  return x + m, new_kv

gpt2 :: Tensor.TokenIds[B,S] -> ?Mask[B,K,CONTEXT_SIZE]
  -> ?Cache[B,NUM_HEADS,P,CONTEXT_SIZE,MODEL_DIM / NUM_HEADS]
  -> ?Bool
  -> (Tensor[B,S,V], ?Cache[B,NUM_HEADS,K,CONTEXT_SIZE,MODEL_DIM / NUM_HEADS])
gpt2 input_ids attn_mask past_kv use_cache = do
  tok <- NN.embedding@wte input_ids dim=MODEL_DIM
  pos <- position_ids input_ids |> NN.embedding@wpe
  x <- tok + pos
  mask <- causal_mask_for_input input_ids attn_mask window=CONTEXT_SIZE
  work_kv <- Cache.prepare past_kv use_cache CONTEXT_SIZE
  read_kv <- Cache.read past_kv work_kv
  x, work_kv <- for@h i <- [0..NUM_LAYERS) carry (x, work_kv) do
    past_i <- Cache.index read_kv i
    x, new_i <- gpt2_block x mask past_i
    work_kv <- Cache.append work_kv new_i
    yield x, work_kv
  work_kv <- Cache.finish work_kv (S + 0)
  new_kv <- (use_cache ? work_kv : null)
  logits <- NN.layernorm@ln_f x |> NN.linear@wte
  return logits, new_kv
```

Key language features:

| Feature | Syntax | Example |
|---|---|---|
| Config-driven dim | `<- (Config.dim @@key default=N :: Dim)` | `MODEL_DIM <- (Config.dim @@n_embd default=768 :: Dim)` |
| Type signature | `name :: type -> type` | `fn :: Tensor[B,S,D] -> Tensor[B,S,V]` |
| Parameter path | `@path` on calls | `NN.linear@c_attn x dim=D bias=true` |
| Pipeline | `\|>` | `attention q k v mask |> merge_heads |> NN.linear@c_proj` |
| For loop with carry | `for@scope i <- [0..N) carry (x) do ...` | Layer loop with KV cache threading |
| Optional type | `?Type` | `?Mask[B,K,W]`, `?Cache[...]` |
| Tuple | `(a, b)` | `return a, new_kv` |
| Ternary | `cond ? a : b` | `use_cache ? work_kv : null` |
| Scope bind | `x <- scope@name do ...` | Named sub-scopes for parameter namespacing |
| Yield | `yield a, b` | Loop body output |

---

## Compiler Pipeline

Each stage transforms the Axon program toward a lower-level representation:

```
  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
  │  Parse  │───▶│  Resolve │───▶│ Normalize│───▶│ Elaborate│───▶│ Flatten │
  │         │    │ imports  │    │ main mod │    │ helpers  │    │  loops  │
  └─────────┘    └──────────┘    └──────────┘    └──────────┘    └─────────┘
                                                                        │
                                                                        ▼
  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────────┐
  │ Codegen  │◀───│ Graph IR │◀───│ Optimize │◀───│     Typecheck        │
  │  torch   │    │ lowering │    │  graph   │    │  (typecheck2)        │
  │  jax     │    └──────────┘    └──────────┘    └──────────────────────┘
  │  mlx     │
  │  tinygrad│
  │  triton  │
  │  vllm    │
  └──────────┘
```

### Stage 1: Parse

Raw text → CST → AST. Handles indentation, `do` blocks, pipelines, for-loops, imports, pragmas.

```bash
synapse axon-stage-dump model.axon out.axon --stage parse
```

### Stage 2: Resolve

All `import` statements are resolved against the builtins library. The output is a
single self-contained file with no imports.

```bash
synapse axon-resolve model.axon resolved.axon
synapse axon-stage-dump model.axon out.axon --stage resolve
```

### Stage 3: Normalize

Selects the main module (via `{-# MAIN "name" #-}` pragma or last module). Resolves
module-level scoping and binding forms.

```bash
synapse axon-stage-dump model.axon out.axon --stage normalize
```

### Stage 4: Elaborate

Expands high-level builtin calls (e.g. `NN.linear`, `attention`) into their definitions,
resolving default parameters and optional arguments.

```bash
synapse axon-stage-dump model.axon out.axon --stage flatten  # includes elaboration
```

### Stage 5: Flatten

Transforms `for` loops with `carry`/`yield` into explicit state-threading form.
Scope binds become let-bindings. The program is now a flat set of definitions.

```bash
synapse axon-stage-dump model.axon out.axon --stage flatten
```

### Stage 6: Typecheck

Inference of tensor shapes, dimension variables, and function types. Catches
shape mismatches, unbound symbols, and arity errors.

```bash
synapse axon-stage-dump model.axon out.axon --stage typecheck --show-types
```

### Stage 7: Graph IR

Lowers typed AST to a typed graph IR with data-flow edges, provenance facts, and
domain analysis. Optional `--optimize-graph` runs SDPA fusion, dead-code elimination,
and backend-specific intrinsic introduction.

```bash
synapse axon-stage-dump model.axon out.axon --stage graph-ir-axon --optimize-graph
synapse axon-graph-ir-dot model.axon graph.dot --optimize-graph
```

### Stage 8: Codegen

Generates Python code for the target backend from the (optimized) Graph IR.

| Backend | `--backend` flag | Description |
|---|---|---|
| PyTorch | `codegen2-torch` | Full op coverage, profiling, device alignment |
| JAX | `codegen2-jax` | JAX arrays and primitives |
| MLX | `codegen2-mlx` | Apple Silicon backend |
| TinyGrad | `codegen2-tinygrad` | Lightweight GPU backend |
| Triton | `codegen2-triton` | Triton kernel fusion on top of torch ABI |
| vLLM | `codegen2-vllm` | vLLM serving backend (benchmark only) |

```bash
synapse axon-codegen-dump model.axon model_torch.py --backend codegen2-torch --optimize-graph
```

---

## Testing & Benchmarking

```bash
# Single model parity test (Axon vs HuggingFace)
synapse axon-test synapse/models/gpt2/generic-gpt2.axon models/gpt2 --dtype float32 --max-len 128

# Benchmark a model family across all declared checkpoints
synapse axon-benchmark synapse/models/gemma3/ --axon-backend codegen2-torch

# Benchmark with graph optimization across multiple backends
synapse axon-benchmark synapse/models/gpt2/ --axon-backends codegen2-torch,codegen2-triton \
  --optimize-graph --log-dir log/gpt2

# Materialize config-specific model files from generic definitions
synapse axon-materialize synapse/models/gemma3/generic-gemma-3.axon --checkpoint google/gemma-3-270m
```

---

## Structure

```
synapse/
├── axon/
│   ├── parse/          # Lark grammar, parser, CST
│   ├── ast/            # AST nodes, renderer, type annotations
│   ├── resolve/        # Import resolution
│   ├── normalize/      # Main-module selection
│   ├── elaborate/      # Builtin expansion
│   ├── flatten/        # Loop flattening, scope desugaring
│   ├── typecheck2/     # Shape inference, type checking
│   ├── graph_ir/       # Typed graph IR, optimizer, DOT renderer
│   ├── codegen2_torch/ # PyTorch backend
│   ├── codegen2_jax/   # JAX backend
│   ├── codegen2_mlx/   # MLX backend
│   ├── codegen2_tinygrad/  # TinyGrad backend
│   ├── codegen2_triton/    # Triton backend
│   └── codegen2_vllm/      # vLLM backend
├── ops/               # Primitive op implementations (_linear, _matmul, _softmax, ...)
├── builtins/          # Axon standard library (.axon files)
│   ├── NN.axon        # linear, embedding, layernorm, rmsnorm, ...
│   ├── Attention.axon # attention, reshape_heads, merge_heads, ...
│   ├── Masking.axon   # causal masks, padding masks
│   ├── Positions.axon # position_ids, RoPE factors
│   ├── Cache.axon     # KV cache management
│   ├── MoE.axon       # Mixture-of-Experts helpers
│   ├── Tensor.axon    # chunk, concat, reshape, ...
│   └── ...
├── models/            # 270+ model definitions (.axon files)
│   ├── gpt2/          # GPT-2 family
│   ├── llama3/        # Llama 3 family
│   ├── gemma3/        # Gemma 3 family
│   ├── deepseekv3/    # DeepSeek V3
│   ├── t5/            # T5 encoder-decoder
│   ├── bert/          # BERT family
│   └── ...
├── cli/               # Typer CLI (axon-resolve, axon-test, axon-benchmark, ...)
├── axon_test.py       # HF vs Axon test harness
├── axon_benchmark.py  # Multi-model benchmark runner
├── mxfp4.py           # MXFP4 dequant-on-load
└── int8.py            # int8 dequant-on-load
```

## License

MIT

<br/>

<p align="center">
  <img src="docs/logos/onlp_logo_processed.png" alt="OdenseNLP" width="200" />
</p>

<p align="center">
  Created by <a href="https://github.com/ordbogen">OdenseNLP</a>
</p>
