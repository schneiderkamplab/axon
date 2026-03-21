from __future__ import annotations

from typing import Any

OP_NAME = "repeat"
LOWERING_ARITY = (1, 2)
LOWERING_ALLOWED_KWARGS: set[str] = {"heads", "kv_heads", "repeats", "times", "dim"}
LOWERING_REQUIRED_KWARGS: set[str] = set()
LOWERING_KWARG_KINDS: dict[str, Any] = {
    "heads": "dim",
    "kv_heads": "dim",
    "repeats": "dim",
    "times": "dim",
    "dim": "int",
}


def uses_node_path(emitter: Any, node_spec: dict[str, Any]) -> bool:
    del emitter, node_spec
    return False


def lowering_normalize_kwargs(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> None:
    if "times" in kwargs and "repeats" not in kwargs:
        kwargs["repeats"] = kwargs.pop("times")
    if len(args) >= 2 and "repeats" not in kwargs:
        kwargs["repeats"] = args[1]
        del args[1:]
    dim = kwargs.pop("dim", 1)
    if dim != 1:
        raise ValueError("repeat currently supports only dim=1 (head axis)")
    if "repeats" in kwargs:
        return
    src_name = args[0].strip()
    if not src_name.isidentifier():
        return
    if "kv_heads" not in kwargs:
        inferred_kv_heads = ctx.tensor_heads.get(src_name)
        if inferred_kv_heads is not None:
            kwargs["kv_heads"] = inferred_kv_heads
    if "heads" not in kwargs and isinstance(out, str):
        inferred_heads = ctx.tensor_heads.get(out)
        if inferred_heads is not None:
            kwargs["heads"] = inferred_heads


def lowering_infer_metadata(
    *,
    args: list[str],
    out: str | list[str],
    kwargs: dict[str, Any],
    ctx: Any,
) -> bool:
    del args
    if not isinstance(out, str):
        return False
    heads = kwargs.get("heads")
    if heads is not None:
        ctx.tensor_heads[out] = heads
    return True


def interpret(
    model: Any,
    node_spec: dict[str, Any],
    env: dict[str, Any],
    *,
    node_path: str,
    scope: str,
    symbols: dict[str, int],
) -> None:
    src = model._read_tensor_input(node_spec.get("_args"), env)
    repeats = node_spec.get("repeats")
    if repeats is None:
        heads = int(model._eval_expr(node_spec.get("heads"), env, symbols))
        kv_heads = int(model._eval_expr(node_spec.get("kv_heads"), env, symbols))
        n_rep = heads // kv_heads
    else:
        n_rep = int(model._eval_expr(repeats, env, symbols))
    out = model._require_name(node_spec.get("_bind"), field="repeat._bind")
    if n_rep == 1:
        env[out] = src
    else:
        bsz, kvh, seq_len, hd = src.shape
        expanded = src[:, :, None, :, :].expand(bsz, kvh, n_rep, seq_len, hd)
        env[out] = expanded.reshape(bsz, kvh * n_rep, seq_len, hd)
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

    src = read(str(node_spec.get("_args")))
    out_name = str(node_spec.get("_bind"))
    out_var = assign_out_var(out_name)
    repeats = node_spec.get("repeats")
    if repeats is None:
        heads = emitter._expr_code(node_spec.get("heads"), env)
        kv_heads = emitter._expr_code(node_spec.get("kv_heads"), env)
        repeats_code = f"(int({heads}) // int({kv_heads}))"
    else:
        repeats_code = emitter._expr_code(repeats, env)
    n_rep = emitter._fresh("n_rep")
    lines.append(f"{indent}{n_rep} = int({repeats_code})")
    lines.append(f"{indent}if {n_rep} == 1:")
    lines.append(f"{indent}    {out_var} = {src}")
    lines.append(f"{indent}else:")
    lines.append(
        f"{indent}    {out_var} = {src}[:, :, None, :, :].expand({src}.shape[0], {src}.shape[1], {n_rep}, {src}.shape[2], {src}.shape[3]).reshape({src}.shape[0], {src}.shape[1] * {n_rep}, {src}.shape[2], {src}.shape[3])"
    )
    return lines


__all__ = [
    "OP_NAME",
    "LOWERING_ARITY",
    "LOWERING_ALLOWED_KWARGS",
    "LOWERING_REQUIRED_KWARGS",
    "LOWERING_KWARG_KINDS",
    "lowering_normalize_kwargs",
    "lowering_infer_metadata",
    "interpret",
    "compile",
    "uses_node_path",
]
