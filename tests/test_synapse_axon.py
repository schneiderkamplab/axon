from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from brainsurgery.synapse import (
    lower_axon_module_to_synapse_spec,
    lower_axon_program_to_synapse_spec,
    parse_axon_module,
    parse_axon_program,
    parse_axon_program_from_path,
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
tiny :: Tensor -> ?Tensor -> Tensor
tiny x cache = do
  y <- x |> linear@proj |> act::gelu_new
  return y
"""
    module = parse_axon_module(source)
    assert module.name == "tiny"
    assert [param.name for param in module.params] == ["x", "cache"]
    assert [param.optional for param in module.params] == [False, True]
    assert module.returns == ()
    assert len(module.statements) == 2


def test_parse_axon_module_expression_definition_without_do() -> None:
    source = """
inc :: I -> I
inc x = x + 1
"""
    module = parse_axon_module(source)
    assert module.name == "inc"
    assert [param.name for param in module.params] == ["x"]
    assert len(module.statements) == 1
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    ops = [node["op"] for node in node_specs]
    assert "add" in ops


def test_primitive_activation_alias_lowering() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  y <- _act_gelu_pytorch_tanh x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["op"] == "activation"
    assert node_specs[0]["kind"] == "gelu_pytorch_tanh"


def test_pointfree_definition_is_eta_expanded() -> None:
    source = """
silu :: Tensor[B,T,D] -> Tensor[B,T,D]
silu = _act_silu

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- silu x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    blocks = spec["model"]["blocks"]
    assert "silu" in blocks
    silu_nodes = _node_specs(blocks["silu"]["graph"])
    assert silu_nodes[0]["op"] == "activation"
    assert silu_nodes[0]["kind"] == "silu"


def test_lowering_reports_shape_mismatch_from_signature_on_block_call() -> None:
    source = """
blk :: Tensor[B,T,768] -> Tensor[B,T,768]
blk x = do
  return x

main :: Tensor[B,T,640] -> Tensor[B,T,768]
main x = do
  y <- blk x
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"shape mismatch in call 'blk'"):
        lower_axon_program_to_synapse_spec(modules, main_module="main")


def test_namespaced_module_call_with_import() -> None:
    source = """
import Lib

Lib.swiglu :: Tensor[B,T,D] -> Tensor[B,T,D]
Lib.swiglu x = do
  return act::silu x * x

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- Lib.swiglu x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["use"] == "Lib.swiglu"


def test_namespaced_module_call_requires_import() -> None:
    source = """
Lib.swiglu :: Tensor[B,T,D] -> Tensor[B,T,D]
Lib.swiglu x = do
  return act::silu x * x

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- Lib.swiglu x
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"requires `import Lib`"):
        lower_axon_program_to_synapse_spec(modules, main_module="main")


def test_builtin_act_import_resolves_from_builtin_file(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Activations

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- Activations.swiglu x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if "use" in first:
        assert first["use"] == "Activations.swiglu"
    else:
        assert first["op"] == "activation"
        assert first["kind"] == "swiglu"


def test_builtin_cache_import_resolves_from_builtin_file(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Cache

main :: ?Cache -> Tensor[B,H,T,D] -> Tensor[B,H,T,D] -> ?Bool -> ?Cache
main past k v use_cache = do
  k_all, v_all, present <- Cache.update past k v when=use_cache
  cache <- Cache.init
  cache <- use_cache ? Cache.append cache present : cache
  return cache
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["use"] == "Cache.update"
    second = node_specs[1]
    if "use" in second:
        assert second["use"] == "Cache.init"
    else:
        assert second["op"] == "init_list"
    third = node_specs[2]
    if "use" in third:
        assert third["use"] == "Cache.append"
    else:
        assert third["op"] == "append"


def test_import_resolution_prefers_local_file_over_builtins(tmp_path: Path) -> None:
    local_act = tmp_path / "Activations.axon"
    local_act.write_text(
        """
Activations.swiglu :: Tensor[B,T,D] -> Tensor[B,T,D]
Activations.swiglu x = do
  return act::relu x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Activations

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- Activations.swiglu x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    blocks = spec["model"]["blocks"]
    assert "Activations.swiglu" in blocks
    block_nodes = _node_specs(blocks["Activations.swiglu"]["graph"])
    assert block_nodes[0]["op"] == "activation"
    assert block_nodes[0]["kind"] == "relu"


def test_selective_import_parenthesized_brings_member_into_scope(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Activations (gelu_new)

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- gelu_new x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if "use" in first:
        assert first["use"] == "Activations.gelu_new"
    else:
        assert first["op"] == "activation"
        assert first["kind"] == "gelu_new"


def test_selective_import_shorthand_brings_member_into_scope(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Activations gelu_new

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- gelu_new x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if "use" in first:
        assert first["use"] == "Activations.gelu_new"
    else:
        assert first["op"] == "activation"
        assert first["kind"] == "gelu_new"


def test_local_module_name_shadows_selective_import() -> None:
    source = """
import Activations (gelu_new)

gelu_new :: Tensor[B,T,D] -> Tensor[B,T,D]
gelu_new x = do
  return x

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- gelu_new x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["use"] == "gelu_new"


def test_parse_program_from_path_loads_imported_axon_modules(tmp_path: Path) -> None:
    lib_path = tmp_path / "Lib.axon"
    lib_path.write_text(
        """
Lib.id :: Tensor[B,T,D] -> Tensor[B,T,D]
Lib.id x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Lib

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- Lib.id x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["use"] == "Lib.id"


def test_prelude_is_implicitly_available_from_file_parse(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- gelu_new x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if "use" in first:
        assert first["use"] == "Prelude.gelu_new"
    else:
        assert first["op"] == "activation"
        assert first["kind"] == "gelu_new"


def test_prelude_does_not_override_native_linear_op_semantics(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- linear@proj x dim=16 bias=true transpose=true
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["op"] == "linear"
    assert node_specs[0]["dim"] == 16
    assert node_specs[0]["bias"] is True
    assert node_specs[0]["transpose"] is True


def test_cache_builtin_import_resolves_from_builtin_file(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Cache

main :: ?Cache -> Tensor[B,H,T,D] -> Tensor[B,H,T,D] -> (?Tensor[B,H,T,D], ?Tensor[B,H,T,D], ?Cache)
main past k v = do
  k_all, v_all, present <- Cache.update past k v
  return k_all, v_all, present
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["use"] == "Cache.update"


def test_moe_builtin_import_resolves_from_builtin_file(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import MoE

main :: Tensor[B,T,D] -> Tensor[B,T,K] -> Tensor[B,T,K] -> I -> (Tensor[N,D], Tensor[N], Tensor[N], Tensor[N])
main hidden topk_scores topk_indices expert = do
  selected_hidden, token_idx, topk_pos, selected_scores <- MoE.select hidden topk_scores topk_indices expert
  return selected_hidden, token_idx, topk_pos, selected_scores
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if "use" in first:
        assert first["use"] == "MoE.select"
    else:
        assert first["op"] == "moe_select_tokens"


def test_multi_path_parameters_support_triple_at_call_syntax() -> None:
    source = """
expert_ffn@gate@up@down :: @Path -> @Path -> @Path -> Tensor[B,T,D] -> Tensor[B,T,D]
expert_ffn@gate@up@down x = do
  g <- linear@gate x
  u <- linear@up x
  y <- g |> mul u |> linear@down
  return y

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- expert_ffn@mlp.gate_proj@mlp.up_proj@mlp.down_proj x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["use"] == "expert_ffn"
    assert node_specs[0]["in"]["gate"] == "'mlp.gate_proj'"
    assert node_specs[0]["in"]["up"] == "'mlp.up_proj'"
    assert node_specs[0]["in"]["down"] == "'mlp.down_proj'"


def test_top_level_constant_stays_symbol_with_expression_module_definition() -> None:
    source = """
D = 7
inc :: I -> I
inc x = x + D
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    assert spec["model"]["symbols"]["D"] == 7


def test_parse_repeat_block_statements() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  for@loop i <- [0..3) do
    y <- add(x, x)
  return y
"""
    module = parse_axon_module(source)
    assert len(module.statements) == 2


def test_parse_rejects_legacy_node_statement() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  node n1 = {"op":"add","in":["x","x"],"out":"y"}
  return y
"""
    with pytest.raises(ValueError, match="unsupported Axon statement"):
        parse_axon_module(source)


def test_parse_rejects_legacy_meta_statement() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  meta symbols = {"D":768}
  return x
"""
    with pytest.raises(ValueError, match="unsupported Axon statement"):
        parse_axon_module(source)


def test_parse_and_lower_scope_bind_expression() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  y <- scope@attn do
    h <- linear@proj x dim=4
    return act::gelu_new h
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    graph = spec["model"]["graph"]
    assert isinstance(graph, list)

    def collect_ops(items: list[dict[str, Any]]) -> list[str]:
        ops: list[str] = []
        for item in items:
            _, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                continue
            op = node_spec.get("op")
            if isinstance(op, str):
                ops.append(op)
            nested = node_spec.get("graph")
            if isinstance(nested, list):
                ops.extend(collect_ops(nested))
        return ops

    ops = collect_ops(graph)
    assert "linear" in ops
    assert "activation" in ops
    assert "_ir_alias" in ops


def test_parenthesized_expression_argument_is_lowered() -> None:
    source = """
tiny :: Tensor[B,T,D] -> Tensor[B,T,D]
tiny x = do
  y <- add((zeros_like x), x)
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    ops = [node["op"] for node in node_specs]
    assert "zeros_like" in ops
    assert "add" in ops


def test_at_path_is_scoped_inside_scope_bind() -> None:
    source = """
tiny :: TokenIds[B,T] -> Tensor[B,T,V]
tiny input_ids = do
  y <- scope@model do
    x <- embedding@embed_tokens input_ids dim=D
    return linear@lm_head x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    graph = spec["model"]["graph"]

    def _collect_param_paths(
        items: list[dict[str, Any]], prefix: str = ""
    ) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for item in items:
            name, node_spec = next(iter(item.items()))
            path = f"{prefix}.{name}" if prefix else name
            if not isinstance(node_spec, dict):
                continue
            op = node_spec.get("op")
            if isinstance(op, str):
                out.append((path, op))
            nested = node_spec.get("graph")
            if isinstance(nested, list):
                out.extend(_collect_param_paths(nested, path))
        return out

    ops = _collect_param_paths(graph)
    assert ("model.embed_tokens", "embedding") in ops
    assert ("model.lm_head", "linear") in ops


def test_double_at_path_is_absolute_inside_scope_bind() -> None:
    source = """
tiny :: TokenIds[B,T] -> Tensor[B,T,V]
tiny input_ids = do
  y <- scope@model do
    x <- embedding@embed_tokens input_ids dim=D
    return linear@@lm_head x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    graph = spec["model"]["graph"]

    def _collect_param_paths(
        items: list[dict[str, Any]], prefix: str = ""
    ) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for item in items:
            name, node_spec = next(iter(item.items()))
            path = f"{prefix}.{name}" if prefix else name
            if not isinstance(node_spec, dict):
                continue
            op = node_spec.get("op")
            if isinstance(op, str):
                out.append((path, op))
            nested = node_spec.get("graph")
            if isinstance(nested, list):
                out.extend(_collect_param_paths(nested, path))
        return out

    ops = _collect_param_paths(graph)
    assert ("model.embed_tokens", "embedding") in ops
    assert ("lm_head", "linear") in ops
    assert ("model.lm_head", "linear") not in ops


def test_parse_rejects_scope_statement_form() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  scope@attn do
    y <- linear@proj x dim=4
    return y
  return x
"""
    with pytest.raises(ValueError, match="scope statement form is not supported"):
        parse_axon_module(source)


def test_parse_for_at_range_loop_sugar() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
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
tiny :: Tensor -> Tensor
tiny x = do
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
tiny :: Tensor -> Tensor
tiny x = do
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
tiny :: Tensor -> Tensor
tiny x = do
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


def test_infer_split_sizes_from_known_last_dim() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  qkv <- linear x dim=3*D
  q, k, v <- split qkv
  return q
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[1]["op"] == "split"
    assert node_specs[1]["sizes"] == ["D", "D", "D"]


def test_split_rejects_parts_and_sizes_together() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  q, k <- split x parts=2 sizes=[4,4]
  return q
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="split accepts either parts or sizes, not both"):
        lower_axon_program_to_synapse_spec(modules)


def test_split_rejects_output_arity_mismatch_for_parts() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  q, k <- split x parts=3
  return q
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"split parts=3 requires 3 outputs, got 2"):
        lower_axon_program_to_synapse_spec(modules)


def test_split_requires_tuple_binding_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  q <- split x parts=1
  return q
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="split requires tuple/list binding outputs"):
        lower_axon_program_to_synapse_spec(modules)


def test_topk_requires_k_kwarg() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  vals, idx <- topk x
  return vals
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"topk missing required kwargs: k"):
        lower_axon_program_to_synapse_spec(modules)


def test_topk_requires_two_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  vals <- topk x k=8
  return vals
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="topk requires exactly two outputs: values, indices"):
        lower_axon_program_to_synapse_spec(modules)


def test_moe_select_tokens_requires_expert_kwarg() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,EPT] -> Tensor[B,T,EPT] -> Tensor[B,T,D]
blk x scores idx = do
  x_sel, token_idx, topk_pos, sel_scores <- moe_select_tokens x scores idx
  return x_sel
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"moe_select_tokens missing required kwargs: expert"):
        lower_axon_program_to_synapse_spec(modules)


def test_moe_select_tokens_requires_four_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,EPT] -> Tensor[B,T,EPT] -> Tensor[B,T,D]
blk x scores idx = do
  x_sel <- moe_select_tokens x scores idx expert=0
  return x_sel
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "moe_select_tokens requires exactly four outputs: "
            "selected_hidden, token_idx, topk_pos, selected_scores"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_moe_scatter_add_requires_single_output() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[N] -> Tensor[N,D] -> Tensor[N] -> Tensor[B,T,D]
blk m idx upd scores = do
  a, b <- moe_scatter_add m idx upd scores
  return a
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="moe_scatter_add requires a single scalar output binding"):
        lower_axon_program_to_synapse_spec(modules)


def test_coalesce_requires_tuple_binding_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- cache::coalesce x x
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="coalesce requires tuple/list binding outputs"):
        lower_axon_program_to_synapse_spec(modules)


def test_coalesce_requires_divisible_input_output_counts() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  a, b <- cache::coalesce x x x
  return a
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="coalesce input count must be divisible by output count"):
        lower_axon_program_to_synapse_spec(modules)


def test_coalesce_requires_variable_name_inputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  a, b <- cache::coalesce x add(x, x)
  return a
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="coalesce inputs must be variable names"):
        lower_axon_program_to_synapse_spec(modules)


def test_topk_accepts_largest_and_sorted_kwargs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  vals, idx <- topk x k=8 dim=-1 largest=false sorted=false
  return vals
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["op"] == "topk"
    assert node_specs[0]["largest"] is False
    assert node_specs[0]["sorted"] is False


def test_softmax_rejects_tuple_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y, z <- softmax x dim=-1
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="softmax requires a single scalar output binding"):
        lower_axon_program_to_synapse_spec(modules)


def test_softmax_rejects_unsupported_dtype() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- softmax x dim=-1 dtype=float64
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"Unsupported softmax dtype: float64"):
        lower_axon_program_to_synapse_spec(modules)


def test_softmax_accepts_supported_dtype_and_default_dim() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- softmax x dtype=bfloat16
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["op"] == "softmax"
    assert node_specs[0]["dtype"] == "bfloat16"
    assert "dim" not in node_specs[0]


def test_zeros_like_rejects_tuple_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y, z <- zeros_like x
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="zeros_like requires a single scalar output binding"):
        lower_axon_program_to_synapse_spec(modules)


def test_add_rejects_tuple_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D] -> Tensor[B,T,D]
blk x y = do
  a, b <- add x y
  return a
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="add requires a single scalar output binding"):
        lower_axon_program_to_synapse_spec(modules)


def test_mul_rejects_tuple_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D] -> Tensor[B,T,D]
blk x y = do
  a, b <- mul x y
  return a
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="mul requires a single scalar output binding"):
        lower_axon_program_to_synapse_spec(modules)


def test_infer_linear_dim_from_return_shape() -> None:
    source = """
gpt2 :: TokenIds[B,T] -> Tensor[B,T,V]
gpt2 input_ids = do
  h <- embedding@wte input_ids dim=D
  logits <- linear@wte h
  return logits
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[1]["op"] == "linear"
    assert node_specs[1]["in"] == "h"
    assert node_specs[1]["out"] == "logits"
    assert node_specs[1]["dim"] == "V"


def test_infer_embedding_dim_from_typed_output_shape() -> None:
    source = """
emb :: TokenIds[B,T] -> Tensor[B,T,D]
emb ids = do
  x <- embedding ids
  return x
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["op"] == "embedding"
    assert node_specs[0]["embedding_dim"] == "D"


def test_add_unifies_symbolic_last_dim_for_following_ops() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor -> Tensor[B,T,D]
blk tok pos = do
  x <- tok + pos
  y <- layernorm pos
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[1]["op"] == "layernorm"
    assert node_specs[1]["dim"] == "D"


def test_embedding_accepts_dim_kwarg() -> None:
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


def test_embedding_rejects_embedding_dim_kwarg() -> None:
    source = """
emb :: TokenIds[B,T] -> Tensor[B,T,D]
emb ids = do
  x <- embedding ids embedding_dim=D
  return x
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="embedding does not support embedding_dim; use dim"):
        lower_axon_program_to_synapse_spec(modules)


def test_embedding_rejects_num_embeddings_kwarg() -> None:
    source = """
emb :: TokenIds[B,T] -> Tensor[B,T,D]
emb ids = do
  x <- embedding ids dim=D num_embeddings=V
  return x
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError, match="embedding unsupported kwargs: num_embeddings; allowed: dim, scale"
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_path_parameterized_block_call_binds_param_base() -> None:
    source = """
lin_bt :: @Path -> Tensor -> I -> Tensor
lin_bt@path x d = do
  return linear@path x dim=d bias=true transpose=true

blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- lin_bt@attn.c_proj x D
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    model_nodes = _node_specs(spec["model"]["graph"])
    assert model_nodes[0]["use"] == "lin_bt"
    assert model_nodes[0]["in"]["path"] == "'attn.c_proj'"
    block_nodes = _node_specs(spec["model"]["blocks"]["lin_bt"]["graph"])
    assert block_nodes[0]["op"] == "linear"
    assert block_nodes[0]["param_base"] == "path"


def test_path_parameter_annotation_rejects_non_path_type() -> None:
    source = """
lin_bt :: @ParamPath -> Tensor -> I -> Tensor
lin_bt@path x d = do
  return linear@path x dim=d bias=true transpose=true

blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- lin_bt@attn.c_proj x D
  return y
"""
    with pytest.raises(ValueError, match=r"path signature type must be Path"):
        parse_axon_program(source)


def test_linear_accepts_transpose_flag() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- linear x dim=D transpose=true bias=false
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["op"] == "linear"
    assert node_specs[0]["transpose"] is True


def test_block_signature_propagates_output_last_dim_from_tensor_shape() -> None:
    source = """
rms :: @Path -> Tensor[B,T,D] -> Tensor[B,T,D]
rms@path x = rmsnorm@path x

blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- rms@norm x
  z <- layernorm y
  return z
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[1]["op"] == "layernorm"
    assert node_specs[1]["dim"] == "D"


def test_block_signature_propagates_output_last_dim_from_scalar_param() -> None:
    source = """
lin :: @Path -> Tensor[B,T,Din] -> I -> Tensor[B,T,dim]
lin@path x dim = linear@path x dim=dim bias=true transpose=true

blk :: Tensor[B,T,D] -> Tensor[B,T,16]
blk x = do
  y <- lin@proj x 16
  z <- layernorm y
  return z
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[1]["op"] == "layernorm"
    assert node_specs[1]["dim"] == 16


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
tiny :: Tensor -> ?Tensor -> Tensor -- signature comment
tiny x cache = do -- def comment
  -- statement comment
  y <- x |> linear@proj(dim=4, bias=false) -- inline comment
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
tiny :: Tensor -> Tensor
tiny x = do
  qkv <- x |>
    layernorm@ln_1(x, dim=768, eps=1e-05) |>
    linear@attn.c_attn(dim=2304, transpose=true, bias=true)
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
tiny :: Tensor -> Tensor
tiny x = do
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
tiny :: Tensor -> Tensor -> Tensor
tiny x wte = do
  return layernorm@ln_f(x, dim=768, eps=1e-05) |> linear@wte(dim=50257)
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
        "out": "out_0",
        "dim": 50257,
    }
    assert spec["model"]["outputs"] == {"out_0": "out_0"}


def test_lower_bind_operator_to_synapse_spec() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
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
tiny :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor
tiny q k v bias = do
  ctx_heads <- reshape_heads_triplet(q, k, v, heads=12, head_dim=64) |>
    attention(backend=sdpa, causal=true, causal_mask_buffer=bias)
  return ctx_heads
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert len(node_specs) == 4
    assert node_specs[0] == {
        "op": "reshape_heads",
        "in": "q",
        "out": "pipe_1",
        "heads": 12,
        "head_dim": 64,
    }
    assert node_specs[1] == {
        "op": "reshape_heads",
        "in": "k",
        "out": "pipe_2",
        "heads": 12,
        "head_dim": 64,
    }
    assert node_specs[2] == {
        "op": "reshape_heads",
        "in": "v",
        "out": "pipe_3",
        "heads": 12,
        "head_dim": 64,
    }
    assert node_specs[3] == {
        "op": "attention",
        "in": ["pipe_1", "pipe_2", "pipe_3"],
        "out": "ctx_heads",
        "backend": "sdpa",
        "causal": True,
        "causal_mask_buffer": "bias",
    }


def test_lower_ternary_to_when_guards() -> None:
    source = """
tiny :: Tensor -> ?Tensor -> (Tensor, Tensor)
tiny x use_cache = do
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


def test_lower_if_then_else_matches_ternary_lowering() -> None:
    ternary_source = """
tiny :: Tensor -> ?Tensor -> (Tensor, Tensor)
tiny x use_cache = do
  k, v <- use_cache ? cache::update(past, k0, v0) : k0, v0
  return k, v
"""
    if_source = """
tiny :: Tensor -> ?Tensor -> (Tensor, Tensor)
tiny x use_cache = do
  k, v <- if use_cache then cache::update(past, k0, v0) else k0, v0
  return k, v
"""

    ternary_spec = lower_axon_module_to_synapse_spec(parse_axon_module(ternary_source))
    if_spec = lower_axon_module_to_synapse_spec(parse_axon_module(if_source))

    assert _node_specs(if_spec["model"]["graph"]) == _node_specs(ternary_spec["model"]["graph"])


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
    assert spec2["model"]["outputs"] == {"x": "x"}
    assert "blocks" in spec2["model"]
    assert "blk" in spec2["model"]["blocks"]
    assert "for@loop i <- [0..(0) + (2)) do" in axon
