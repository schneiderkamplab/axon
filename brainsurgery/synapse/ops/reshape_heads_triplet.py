from __future__ import annotations

from typing import Any

OP_NAME = "reshape_heads_triplet"


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    ins = node_spec.get("in")
    outs = node_spec.get("out")
    if not isinstance(ins, list) or not isinstance(outs, list) or len(ins) != 3 or len(outs) != 3:
        raise ValueError("reshape_heads_triplet requires 3 inputs and 3 outputs")
    first = env[ins[0]]
    hidden = int(first.shape[-1])
    raw_heads = node_spec.get("heads")
    raw_head_dim = node_spec.get("head_dim")
    if raw_heads is None and raw_head_dim is None:
        raise ValueError("reshape_heads_triplet requires heads or head_dim")
    heads = int(model._eval_expr(raw_heads, env, symbols)) if raw_heads is not None else None
    head_dim = (
        int(model._eval_expr(raw_head_dim, env, symbols)) if raw_head_dim is not None else None
    )
    if heads is None:
        assert head_dim is not None
        if hidden % head_dim != 0:
            raise ValueError("reshape_heads_triplet could not infer heads from head_dim")
        heads = hidden // head_dim
    if head_dim is None:
        assert heads is not None
        if hidden % heads != 0:
            raise ValueError("reshape_heads_triplet could not infer head_dim from heads")
        head_dim = hidden // heads
    if hidden != heads * head_dim:
        raise ValueError("reshape_heads_triplet heads*head_dim must equal input width")
    for src_name, dst_name in zip(ins, outs, strict=True):
        src = env[src_name]
        if int(src.shape[-1]) != heads * head_dim:
            raise ValueError("reshape_heads_triplet inputs must all have equal hidden width")
        bsz, seq_len, _ = src.shape
        reshaped = src.view(bsz, seq_len, heads, head_dim).transpose(1, 2)
        env[dst_name] = reshaped
    return


def compile(
    emitter: Any,
    node_spec: dict[str, Any],
    env: dict[str, str],
    *,
    node_path_var: str,
    scope_var: str,
    indent: str,
) -> list[str]:
    lines: list[str] = []

    def assign_out_var(out_name: str) -> str:
        return emitter._assign_out_var(env, out_name)

    def infer_param(param_name: str) -> str:
        return emitter._infer_param_expr(node_spec, node_path_var, param_name)

    def read(name: str) -> str:
        return emitter._read_env_var(env, name)

    ins = node_spec.get("in")
    outs = node_spec.get("out")
    if not isinstance(ins, list) or not isinstance(outs, list) or len(ins) != 3 or len(outs) != 3:
        raise ValueError("reshape_heads_triplet requires 3 inputs and outputs")
    src0 = read(str(ins[0]))
    raw_heads = node_spec.get("heads")
    raw_head_dim = node_spec.get("head_dim")
    if raw_heads is None and raw_head_dim is None:
        raise ValueError("reshape_heads_triplet requires heads or head_dim")
    heads_code = emitter._expr_code(raw_heads, env) if raw_heads is not None else None
    head_dim_code = emitter._expr_code(raw_head_dim, env) if raw_head_dim is not None else None
    heads_var = emitter._fresh("heads")
    head_dim_var = emitter._fresh("head_dim")
    expected_var = emitter._fresh("expected_hidden")
    if heads_code is None:
        lines.append(f"{indent}{heads_var} = None")
    else:
        lines.append(f"{indent}{heads_var} = int({heads_code})")
    if head_dim_code is None:
        lines.append(f"{indent}{head_dim_var} = None")
    else:
        lines.append(f"{indent}{head_dim_var} = int({head_dim_code})")
    lines.append(f"{indent}if {heads_var} is None and {head_dim_var} is None:")
    lines.append(
        f"{indent}    raise ValueError('reshape_heads_triplet requires heads or head_dim')"
    )
    lines.append(f"{indent}if {heads_var} is None:")
    lines.append(f"{indent}    if {src0}.shape[-1] % int({head_dim_var}) != 0:")
    lines.append(
        f"{indent}        raise ValueError('reshape_heads_triplet could not infer heads from head_dim')"
    )
    lines.append(f"{indent}    {heads_var} = {src0}.shape[-1] // int({head_dim_var})")
    lines.append(f"{indent}if {head_dim_var} is None:")
    lines.append(f"{indent}    if {src0}.shape[-1] % int({heads_var}) != 0:")
    lines.append(
        f"{indent}        raise ValueError('reshape_heads_triplet could not infer head_dim from heads')"
    )
    lines.append(f"{indent}    {head_dim_var} = {src0}.shape[-1] // int({heads_var})")
    lines.append(f"{indent}{expected_var} = int({heads_var}) * int({head_dim_var})")
    lines.append(f"{indent}if {src0}.shape[-1] != {expected_var}:")
    lines.append(
        f"{indent}    raise ValueError('reshape_heads_triplet heads*head_dim must equal input width')"
    )
    for src_name, out_name in zip(ins, outs, strict=True):
        src = read(str(src_name))
        out_var = assign_out_var(str(out_name))
        lines.append(f"{indent}if {src}.shape[-1] != {expected_var}:")
        lines.append(
            f"{indent}    raise ValueError('reshape_heads_triplet inputs must all have equal hidden width')"
        )
        lines.append(
            f"{indent}{out_var} = {src}.view({src}.shape[0], {src}.shape[1], int({heads_var}), int({head_dim_var})).transpose(1, 2)"
        )
    return lines


__all__ = ["OP_NAME", "interpret", "compile", "uses_node_path"]
