# Synapse

Axon: a domain-specific language and compiler for neural network model definitions.

## Overview

Synapse provides the **Axon** language — a typed, declarative DSL for defining
neural network architectures — along with a full compiler pipeline:

- **Parser** → AST → Resolver → Normalizer → Elaborator → Flattener → Typechecker → Graph IR → Optimizer → Codegen
- **Backend codegen**: PyTorch, MLX, JAX, TinyGrad, Triton, vLLM
- **Builtins**: standard library of tensor ops, attention, masking, positions, NN, MoE
- **Model definitions**: 270+ model specs covering Llama, GPT, Gemma, T5, BERT, DeepSeek, Qwen, and more

## Installation

```bash
pip install -e .
```

### Optional backends

```bash
pip install -e ".[mlx]"    # Apple MLX backend
pip install -e ".[dev]"   # Development tools
```

## Usage

```bash
# Resolve imports in an Axon file
synapse axon-resolve synapse/models/gpt2/generic-gpt2.axon output.axon

# Dump a compiler stage
synapse axon-stage-dump synapse/models/gpt2/generic-gpt2.axon dump.axon --stage graph-ir-axon

# Generate PyTorch code
synapse axon-codegen-dump synapse/models/gpt2/generic-gpt2.axon model.py --backend codegen2-torch

# Run HF vs Axon benchmark
synapse axon-test synapse/models/gpt2/generic-gpt2.axon models/gpt2

# Run benchmark across declared checkpoints
synapse axon-benchmark synapse/models/
```

## Structure

```
synapse/
├── axon/          # Compiler: parser, AST, typecheck, lowering, graph IR, codegen
├── ops/           # Primitive op implementations (_xyz)
├── builtins/      # Axon standard library (.axon files)
├── models/        # Model definitions (.axon files)
├── cli/           # CLI commands
├── mxfp4.py       # MXFP4 dequant-on-load
├── int8.py        # int8 dequant-on-load
├── axon_test.py   # HF vs Axon test harness
└── ...
```

## License

MIT
