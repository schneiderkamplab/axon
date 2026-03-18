from __future__ import annotations

from typing import Any

from brainsurgery.synapse import (
    lower_axon_module_to_synapse_spec,
    lower_axon_program_to_synapse_spec,
    parse_axon_module,
    parse_axon_program,
    synapse_spec_to_axon_module_text,
)


def _node_specs(graph: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in graph:
        assert isinstance(item, dict) and len(item) == 1
        _, node_spec = next(iter(item.items()))
        assert isinstance(node_spec, dict)
        out.append(node_spec)
    return out


def test_parse_axon_module_header_and_bindings() -> None:
    source = """
module tiny(x, cache?) -> (y) do
  y <- x |> linear@proj |> act::gelu_new
  return y
"""
    module = parse_axon_module(source)
    assert module.name == "tiny"
    assert [param.name for param in module.params] == ["x", "cache"]
    assert [param.optional for param in module.params] == [False, True]
    assert module.returns == ("y",)
    assert len(module.statements) == 2


def test_parse_repeat_block_statements() -> None:
    source = """
module tiny(x) -> (y) do
  repeat loop: i in 3 do
    y <- add(x, x)
  return y
"""
    module = parse_axon_module(source)
    assert len(module.statements) == 2


def test_parse_for_at_range_loop_sugar() -> None:
    source = """
module tiny(x) -> (y) do
  for@model.layers i <- [0..3] do
    y <- add(x, x)
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    model = spec["model"]
    repeat_node = model["graph"][0]["model"]["graph"][0]["layers"]
    assert repeat_node["op"] == "repeat"
    assert repeat_node["var"] == "i"
    assert repeat_node["range"] == "(3) + 1"


def test_parse_for_at_range_loop_sugar_with_nonzero_start() -> None:
    source = """
module tiny(x) -> (y) do
  for@model.layers i <- [1..4] do
    y <- add(x, x)
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    model = spec["model"]
    repeat_node = model["graph"][0]["model"]["graph"][0]["layers"]
    assert repeat_node["op"] == "repeat"
    assert repeat_node["var"] == "i"
    assert repeat_node["start"] == "1"
    assert repeat_node["range"] == "((4) + 1) - (1)"


def test_parse_for_at_range_loop_sugar_half_open_with_paren() -> None:
    source = """
module tiny(x) -> (y) do
  for@model.layers i <- [1..4) do
    y <- add(x, x)
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    model = spec["model"]
    repeat_node = model["graph"][0]["model"]["graph"][0]["layers"]
    assert repeat_node["op"] == "repeat"
    assert repeat_node["var"] == "i"
    assert repeat_node["start"] == "1"
    assert repeat_node["range"] == "(4) - (1)"


def test_parse_for_at_range_loop_sugar_left_open_right_closed() -> None:
    source = """
module tiny(x) -> (y) do
  for@model.layers i <- (0..4] do
    y <- add(x, x)
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    model = spec["model"]
    repeat_node = model["graph"][0]["model"]["graph"][0]["layers"]
    assert repeat_node["op"] == "repeat"
    assert repeat_node["var"] == "i"
    assert repeat_node["start"] == "(0) + 1"
    assert repeat_node["range"] == "((4) + 1) - ((0) + 1)"


def test_parse_top_level_haskell_constants_across_modules() -> None:
    source = """
D = 768

id_block :: Tensor -> Tensor
id_block x = do
  return x

eps = 1e-05

main :: Tensor -> Tensor
main x = do
  y <- layernorm x dim=D eps=eps
  return y
"""
    modules = parse_axon_program(source)
    assert [m.name for m in modules] == ["id_block", "main"]
    spec = lower_axon_program_to_synapse_spec(modules)
    symbols = spec["model"].get("symbols")
    assert symbols == {"D": 768, "eps": 1e-05}
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["op"] == "layernorm"
    assert node_specs[0]["dim"] == "D"
    assert node_specs[0]["eps"] == "eps"


def test_type_shape_annotations_expose_symbols_and_infer_layernorm_dim() -> None:
    source = """
gpt2_block :: Tensor[B,T,D] -> Tensor[B,T,D]
gpt2_block x = do
  y <- layernorm@ln_1 x eps=1e-05
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    symbols = spec["model"].get("symbols")
    assert symbols == {"B": None, "T": None, "D": None}
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["op"] == "layernorm"
    assert node_specs[0]["in"] == "x"
    assert node_specs[0]["dim"] == "D"


def test_type_shape_annotations_infer_rmsnorm_dim() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- rmsnorm@n x eps=1e-06
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["op"] == "rmsnorm"
    assert node_specs[0]["in"] == "x"
    assert node_specs[0]["dim"] == "D"


def test_infer_split_last_sizes_from_known_last_dim() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  qkv <- linear x out_features=3*D
  q, k, v <- split_last qkv
  return q
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[1]["op"] == "split_last"
    assert node_specs[1]["sizes"] == ["D", "D", "D"]


def test_infer_linear_out_features_from_return_shape() -> None:
    source = """
gpt2 :: TokenIds[B,T] -> Tensor[B,T,V]
gpt2 input_ids = do
  h <- embedding@wte input_ids num_embeddings=V embedding_dim=D
  logits <- linear h tie_weight=wte.weight
  return logits
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[1]["op"] == "linear"
    assert node_specs[1]["in"] == "h"
    assert node_specs[1]["out"] == "logits"
    assert node_specs[1]["out_features"] == "V"


def test_infer_embedding_dim_from_typed_output_shape() -> None:
    source = """
emb :: TokenIds[B,T] -> Tensor[B,T,D]
emb ids = do
  x <- embedding ids num_embeddings=V
  return x
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["op"] == "embedding"
    assert node_specs[0]["embedding_dim"] == "D"


def test_embedding_accepts_dim_alias_for_embedding_dim() -> None:
    source = """
emb :: TokenIds[B,T] -> Tensor[B,T,D]
emb ids = do
  x <- embedding ids dim=D
  return x
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["op"] == "embedding"
    assert node_specs[0]["embedding_dim"] == "D"


def test_linear_accepts_transpose_alias_for_io_layout() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- linear x out_features=D transpose=true bias=false
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["op"] == "linear"
    assert node_specs[0]["weight_layout"] == "io"


def test_infer_repeat_kv_heads_and_kv_heads_from_typed_shapes() -> None:
    source = """
rk :: Tensor[B,Kh,T,Hd] -> Tensor[B,H,T,Hd]
rk k = do
  k_ctx <- repeat_kv k
  return k_ctx
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["op"] == "repeat_kv"
    assert node_specs[0]["kv_heads"] == "Kh"
    assert node_specs[0]["heads"] == "H"


def test_parse_axon_ignores_haskell_style_comments() -> None:
    source = """
-- leading comment
module tiny(x, cache?) -> (y) do -- module comment
  -- statement comment
  y <- x |> linear@proj(out_features=4, bias=false) -- inline comment
  return y -- trailing comment
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert len(node_specs) == 1
    assert node_specs[0]["op"] == "linear"
    assert node_specs[0]["in"] == "x"
    assert node_specs[0]["out"] == "y"


def test_parse_and_lower_pipeline_with_trailing_operator_continuations() -> None:
    source = """
module tiny(x) -> (qkv) do
  qkv <- x |>
    layernorm@ln_1(x, dim=768, eps=1e-05) |>
    linear@attn.c_attn(out_features=2304, weight_layout=io, bias=true)
  return qkv
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert len(node_specs) == 2
    assert node_specs[0]["op"] == "layernorm"
    assert node_specs[0]["out"] == "pipe_1"
    assert "graph" in node_specs[1]
    c_attn = node_specs[1]["graph"][0]["c_attn"]
    assert c_attn["op"] == "linear"
    assert c_attn["in"] == "pipe_1"
    assert c_attn["out"] == "qkv"


def test_lower_pipeline_axon_to_synapse_spec() -> None:
    source = """
module tiny(x) -> (y) do
  y <- x |> linear@proj |> act::gelu_new
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)

    model = spec["model"]
    assert model["inputs"] == {"x": {"optional": False}}
    assert model["outputs"] == {"y": "y"}

    node_specs = _node_specs(model["graph"])
    assert node_specs[0] == {
        "op": "linear",
        "in": "x",
        "out": "pipe_1",
    }
    assert node_specs[1] == {
        "op": "activation",
        "in": "pipe_1",
        "out": "y",
        "kind": "gelu_new",
    }


def test_lower_return_pipeline_expression_to_named_output() -> None:
    source = """
module tiny(x, wte) -> (logits) do
  return layernorm@ln_f(x, dim=768, eps=1e-05) |> linear(out_features=50257, tie_weight=wte.weight)
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert len(node_specs) == 2
    assert node_specs[0] == {
        "op": "layernorm",
        "in": "x",
        "out": "pipe_1",
        "dim": 768,
        "eps": "1e-05",
    }
    assert node_specs[1] == {
        "op": "linear",
        "in": "pipe_1",
        "out": "logits",
        "out_features": 50257,
        "tie_weight": "wte.weight",
    }
    assert spec["model"]["outputs"] == {"logits": "logits"}


def test_lower_bind_operator_to_synapse_spec() -> None:
    source = """
module tiny(x) -> (y) do
  y <- linear@p1(x) >>= \\z -> act::gelu_new(z)
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0] == {
        "op": "linear",
        "in": "x",
        "out": "bind_1",
    }
    assert node_specs[1] == {
        "op": "activation",
        "in": "bind_1",
        "out": "y",
        "kind": "gelu_new",
    }


def test_lower_pipeline_multi_output_stage_into_next_call_args() -> None:
    source = """
module tiny(q, k, v, bias) -> (ctx_heads) do
  ctx_heads <- reshape_heads_triplet(q, k, v, heads=12, head_dim=64) |>
    attention(backend=sdpa, causal=true, causal_mask_buffer=bias)
  return ctx_heads
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert len(node_specs) == 2
    assert node_specs[0] == {
        "op": "reshape_heads_triplet",
        "in": ["q", "k", "v"],
        "out": ["pipe_1", "pipe_2", "pipe_3"],
        "heads": 12,
        "head_dim": 64,
    }
    assert node_specs[1] == {
        "op": "attention",
        "in": ["pipe_1", "pipe_2", "pipe_3"],
        "out": "ctx_heads",
        "backend": "sdpa",
        "causal": True,
        "causal_mask_buffer": "bias",
    }


def test_lower_ternary_to_when_guards() -> None:
    source = """
module tiny(x, use_cache?) -> (k, v) do
  k, v <- use_cache ? cache::update(past, k0, v0) : k0, v0
  return k, v
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert len(node_specs) == 3
    assert node_specs[0]["when"] == "use_cache"
    assert node_specs[1]["when"] == "not (use_cache)"
    assert node_specs[2]["when"] == "not (use_cache)"


def test_synapse_to_axon_roundtrip_equivalence_for_subset() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"optional": False}},
            "graph": [
                {
                    "n1": {
                        "op": "linear",
                        "in": "x",
                        "out": "h",
                        "params": {"weight": "proj.weight", "bias": "proj.bias"},
                    }
                },
                {
                    "n2": {
                        "op": "activation",
                        "in": "h",
                        "out": "y",
                        "kind": "gelu_new",
                    }
                },
            ],
            "outputs": {"y": "y"},
        },
    }

    axon = synapse_spec_to_axon_module_text(spec, module_name="tiny")
    reparsed = parse_axon_module(axon)
    spec2 = lower_axon_module_to_synapse_spec(reparsed)

    assert spec2["model"]["inputs"] == spec["model"]["inputs"]
    assert spec2["model"]["outputs"] == spec["model"]["outputs"]
    assert _node_specs(spec2["model"]["graph"])[0]["op"] == "linear"
    assert _node_specs(spec2["model"]["graph"])[0]["in"] == "x"
    assert _node_specs(spec2["model"]["graph"])[0]["out"] == "h"
    assert _node_specs(spec2["model"]["graph"])[1] == _node_specs(spec["model"]["graph"])[1]


def test_synapse_to_axon_roundtrip_with_meta_and_control_nodes() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "name": "Tiny",
            "symbols": {"L": 2},
            "inputs": {"x": {"optional": False}},
            "graph": [
                {
                    "n1": {
                        "op": "repeat",
                        "var": "i",
                        "range": "L",
                        "body": [{"a": {"op": "add", "in": ["x", "x"], "out": "x"}}],
                    }
                },
                {"n2": {"use": "block", "in": {"x": "x"}, "out": {"y": "x"}}},
                {
                    "n3": {
                        "op": "layernorm",
                        "in": "x",
                        "out": "y",
                        "dim": 4,
                        "eps": 1e-5,
                        "when": "true",
                    }
                },
            ],
            "outputs": {"logits": "y"},
            "blocks": {"block": {"inputs": {"x": {}}, "graph": [], "outputs": {"y": "x"}}},
        },
    }

    axon = synapse_spec_to_axon_module_text(spec, module_name="tiny")
    reparsed = parse_axon_program(axon)
    spec2 = lower_axon_program_to_synapse_spec(reparsed)
    assert spec2["synapse"] == 1
    assert "block" in spec2["model"]["blocks"]


def test_synapse_to_axon_readable_omits_meta_lines() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "name": "Tiny",
            "symbols": {"D": 4},
            "inputs": {"x": {"optional": False}},
            "graph": [{"n1": {"op": "activation", "in": "x", "out": "y", "kind": "gelu_new"}}],
            "outputs": {"y": "y"},
        },
    }
    axon = synapse_spec_to_axon_module_text(spec, module_name="tiny")
    assert "meta " not in axon
    assert "y <- act::gelu_new(x)" in axon


def test_synapse_to_axon_readable_blocks_lower_back_via_program() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {"L": 2},
            "blocks": {
                "blk": {
                    "inputs": {"x": {"optional": False}},
                    "graph": [
                        {"n": {"op": "activation", "in": "x", "out": "y", "kind": "gelu_new"}}
                    ],
                    "outputs": {"y": "y"},
                }
            },
            "inputs": {"x": {"optional": False}},
            "graph": [
                {
                    "loop": {
                        "op": "repeat",
                        "var": "i",
                        "range": 2,
                        "body": [{"u": {"use": "blk", "in": {"x": "x"}, "out": {"y": "x"}}}],
                    }
                }
            ],
            "outputs": {"y": "x"},
        },
    }
    axon = synapse_spec_to_axon_module_text(spec, module_name="main")
    modules = parse_axon_program(axon)
    spec2 = lower_axon_program_to_synapse_spec(modules)
    assert spec2["model"]["outputs"] == spec["model"]["outputs"]
    assert "blocks" in spec2["model"]
    assert "blk" in spec2["model"]["blocks"]
    assert "repeat loop: i in 2 do" in axon
