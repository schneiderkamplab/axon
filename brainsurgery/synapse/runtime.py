from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from .ops import get_op_module


class SynapseProgramModel(nn.Module):
    """Generic runtime for Synapse graph specs backed by checkpoint tensors."""

    SPEC: dict[str, Any] = {}
    OP_MAP: dict[str, Any] = {}

    def __init__(
        self,
        spec: dict[str, Any] | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.spec: dict[str, Any] = self._resolve_spec(spec)
        self._state: dict[str, torch.Tensor] = {}
        if state_dict is not None:
            self.load_state_dict_tensors(state_dict)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        spec: dict[str, Any] | None = None,
    ) -> "SynapseProgramModel":
        return cls(spec=spec, state_dict=state_dict)

    @classmethod
    def from_spec(
        cls,
        spec: dict[str, Any],
        *,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> "SynapseProgramModel":
        return cls(spec=spec, state_dict=state_dict)

    @classmethod
    def from_yaml(
        cls,
        spec_path: str | Path,
        *,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> "SynapseProgramModel":
        loaded = OmegaConf.load(Path(spec_path))
        data = OmegaConf.to_container(loaded, resolve=True)
        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML mapping at {spec_path}, got {type(data).__name__}")
        return cls(spec={str(key): value for key, value in data.items()}, state_dict=state_dict)

    def load_state_dict_tensors(self, state_dict: dict[str, torch.Tensor]) -> None:
        self._state = dict(state_dict)

    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:
        if input_ids is not None:
            inputs = {"input_ids": input_ids, **inputs}

        spec = self.spec
        model = spec.get("model", {})
        symbols_raw = model.get("symbols", {})
        symbols = {k: v for k, v in symbols_raw.items() if isinstance(v, (int, float, bool))}
        blocks = model.get("blocks", {})
        input_specs = model.get("inputs", {})
        if not isinstance(input_specs, dict):
            raise ValueError("model.inputs must be a mapping when present")

        env: dict[str, Any] = dict(inputs)
        for input_name, input_spec in input_specs.items():
            optional = isinstance(input_spec, dict) and bool(input_spec.get("optional", False))
            if input_name in env:
                continue
            if optional:
                env[input_name] = None
            else:
                raise ValueError(f"Missing required input: {input_name}")
        self._run_graph(model.get("graph", []), env, scope="", symbols=symbols, blocks=blocks)

        outputs = model.get("outputs", {})
        if not isinstance(outputs, dict):
            raise ValueError("model.outputs must be a mapping")
        resolved_outputs: dict[str, Any] = {}
        for key, ref in outputs.items():
            resolved_outputs[key] = self._resolve_output_ref(ref, env)

        if "logits" in resolved_outputs and len(resolved_outputs) == 1:
            return resolved_outputs["logits"]
        return resolved_outputs

    def _resolve_spec(self, spec: dict[str, Any] | None) -> dict[str, Any]:
        resolved = self.SPEC if spec is None else spec
        if not isinstance(resolved, dict):
            raise ValueError("Synapse spec must be a mapping")
        if resolved.get("synapse") != 1:
            raise ValueError("Only synapse: 1 specs are supported")
        model = resolved.get("model")
        if not isinstance(model, dict):
            raise ValueError("spec.model must be a mapping")
        graph = model.get("graph")
        if not isinstance(graph, list):
            raise ValueError("spec.model.graph must be a list")
        return resolved

    def _require_name(self, value: Any, *, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
        return value

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        eos_token_id: int,
        max_len: int,
        attention_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be rank-2 [batch, seq]")
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        if attention_mask is not None and attn_mask is not None:
            raise ValueError("pass at most one of attention_mask or attn_mask")
        mask = attention_mask if attention_mask is not None else attn_mask
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError("attention_mask must be rank-2 [batch, seq]")
            if mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have same shape as input_ids")
        if input_ids.size(1) >= max_len:
            return input_ids[:, :max_len]

        generated = input_ids
        generated_mask = mask
        past_key_values = None
        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                while generated.size(1) < max_len and not torch.all(finished):
                    step_input = generated if past_key_values is None else generated[:, -1:]
                    if generated_mask is None:
                        model_out = self.forward(
                            step_input, past_key_values=past_key_values, use_cache=True
                        )
                    else:
                        model_out = self.forward(
                            step_input,
                            attention_mask=generated_mask,
                            attn_mask=generated_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                    if isinstance(model_out, dict):
                        logits = model_out["logits"]
                        if "past_key_values" in model_out:
                            past_key_values = model_out["past_key_values"]
                        elif "present_key_values" in model_out:
                            past_key_values = model_out["present_key_values"]
                    else:
                        logits = model_out
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    next_token = torch.where(
                        finished,
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )
                    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                    finished = torch.logical_or(finished, next_token == eos_token_id)
                    if generated_mask is not None:
                        next_mask = torch.ones(
                            (generated_mask.shape[0], 1),
                            dtype=generated_mask.dtype,
                            device=generated_mask.device,
                        )
                        generated_mask = torch.cat([generated_mask, next_mask], dim=1)
        finally:
            if was_training:
                self.train()
        return generated

    def _run_graph(
        self,
        graph: list[Any],
        env: dict[str, Any],
        *,
        scope: str,
        symbols: dict[str, int | float | bool],
        blocks: dict[str, Any],
    ) -> None:
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"Invalid graph item: {item!r}")
            node_name, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                raise ValueError(f"Node spec for {node_name!r} must be mapping")

            when_expr = node_spec.get("when")
            if when_expr is not None:
                for produced_name in self._node_output_names(node_spec):
                    env.setdefault(produced_name, None)
                if not self._check_when(when_expr, env, symbols):
                    continue

            op = node_spec.get("op")
            if op == "repeat":
                range_value = self._eval_expr(node_spec.get("range"), env, symbols)
                if not isinstance(range_value, int):
                    raise ValueError(f"repeat range must resolve to int, got {range_value!r}")
                start_value = self._eval_expr(node_spec.get("start", 0), env, symbols)
                if not isinstance(start_value, int):
                    raise ValueError(f"repeat start must resolve to int, got {start_value!r}")
                var_name = node_spec.get("var")
                if not isinstance(var_name, str):
                    raise ValueError("repeat requires string 'var'")
                body = node_spec.get("body")
                if not isinstance(body, list):
                    raise ValueError("repeat requires list 'body'")
                for i in range(range_value):
                    iter_value = start_value + i
                    env[var_name] = iter_value
                    repeat_scope = self._join(scope, f"{node_name}.{iter_value}")
                    self._run_graph(body, env, scope=repeat_scope, symbols=symbols, blocks=blocks)
                env.pop(var_name, None)
                continue

            if "use" in node_spec:
                self._run_block_use(node_spec, env, scope=scope, symbols=symbols, blocks=blocks)
                continue

            if "graph" in node_spec and op is None:
                nested = node_spec.get("graph")
                if not isinstance(nested, list):
                    raise ValueError("graph node must contain list 'graph'")
                nested_scope = self._join(scope, node_name)
                self._run_graph(nested, env, scope=nested_scope, symbols=symbols, blocks=blocks)
                continue

            if not isinstance(op, str):
                raise ValueError(f"Node {node_name!r} missing string op")

            node_path = self._join(scope, node_name)
            self._execute_op(op, node_spec, env, node_path=node_path, scope=scope, symbols=symbols)

    def _run_block_use(
        self,
        node_spec: dict[str, Any],
        env: dict[str, Any],
        *,
        scope: str,
        symbols: dict[str, int | float | bool],
        blocks: dict[str, Any],
    ) -> None:
        block_name = node_spec.get("use")
        if not isinstance(block_name, str):
            raise ValueError("use must be a string block name")
        block_spec = blocks.get(block_name)
        if not isinstance(block_spec, dict):
            raise ValueError(f"Unknown block {block_name!r}")

        block_env = dict(env)
        in_bindings = node_spec.get("in", {})
        if not isinstance(in_bindings, dict):
            raise ValueError("block use 'in' must be mapping")
        for block_input_name, src_name in in_bindings.items():
            if isinstance(src_name, str) and src_name in env:
                block_env[block_input_name] = env[src_name]
            else:
                block_env[block_input_name] = self._eval_expr(src_name, env, symbols)

        block_graph = block_spec.get("graph")
        if not isinstance(block_graph, list):
            raise ValueError("block spec must include list graph")
        self._run_graph(block_graph, block_env, scope=scope, symbols=symbols, blocks=blocks)

        out_bindings = node_spec.get("out", {})
        if not isinstance(out_bindings, dict):
            raise ValueError("block use 'out' must be mapping")
        for block_out_name, dst_name in out_bindings.items():
            env[dst_name] = block_env.get(block_out_name)

    def _node_output_names(self, node_spec: dict[str, Any]) -> list[str]:
        if "use" in node_spec:
            out_bindings = node_spec.get("out")
            if isinstance(out_bindings, dict):
                return [str(v) for v in out_bindings.values()]
            return []
        out_value = node_spec.get("out")
        if isinstance(out_value, str):
            return [out_value]
        if isinstance(out_value, list):
            return [str(v) for v in out_value]
        return []

    def _execute_op(
        self,
        op: str,
        node_spec: dict[str, Any],
        env: dict[str, Any],
        *,
        node_path: str,
        scope: str,
        symbols: dict[str, int | float | bool],
    ) -> None:
        op_module = get_op_module(op)
        if op_module is None:
            raise NotImplementedError(f"Unsupported op: {op}")
        op_module.interpret(
            self,
            node_spec,
            env,
            node_path=node_path,
            scope=scope,
            symbols=symbols,
        )

    def _infer_param_path(
        self, node_spec: dict[str, Any], *, node_path: str, param_name: str
    ) -> str:
        explicit_params = node_spec.get("params")
        if isinstance(explicit_params, dict):
            explicit = explicit_params.get(param_name)
            if isinstance(explicit, str):
                return explicit
        if param_name in node_spec and isinstance(node_spec[param_name], str):
            candidate = node_spec[param_name]
            if "." in candidate:
                return candidate
        return f"{node_path}.{param_name}" if node_path else param_name

    def _resolve_output_ref(self, ref: Any, env: dict[str, Any]) -> Any:
        if isinstance(ref, str):
            return env[ref]
        if isinstance(ref, dict):
            from_ref = ref.get("from")
            if isinstance(from_ref, str):
                return env[from_ref]
        raise ValueError(f"Unsupported output ref: {ref!r}")

    def _read_tensor_input(self, ref: Any, env: dict[str, Any]) -> torch.Tensor:
        if not isinstance(ref, str):
            raise ValueError(f"Expected string tensor reference, got {ref!r}")
        value = env.get(ref)
        if not torch.is_tensor(value):
            raise ValueError(f"Input reference {ref!r} does not resolve to tensor")
        return value

    def _check_when(
        self, when_expr: Any, env: dict[str, Any], symbols: dict[str, int | float | bool]
    ) -> bool:
        if when_expr is None:
            return True
        value = self._eval_expr(when_expr, env, symbols)
        return bool(value)

    def _eval_expr(
        self, expr: Any, env: dict[str, Any], symbols: dict[str, int | float | bool]
    ) -> Any:
        if expr is None:
            return None
        if isinstance(expr, (int, float, bool)):
            return expr
        if isinstance(expr, str):
            token = expr.strip()
            if token in env:
                return env[token]
            if token in symbols:
                return symbols[token]
            if token.lower() == "true":
                return True
            if token.lower() == "false":
                return False
            if token.lower() == "null":
                return None
            return self._safe_eval_numeric(token, env, symbols)
        return expr

    def _safe_eval_numeric(
        self, text: str, env: dict[str, Any], symbols: dict[str, int | float | bool]
    ) -> Any:
        names: dict[str, Any] = {}
        for key, value in symbols.items():
            names[key] = value
        for key, value in env.items():
            if isinstance(value, (int, float, bool)) or value is None:
                names[key] = value

        parsed = ast.parse(text, mode="eval")
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.Pow,
            ast.USub,
            ast.UAdd,
            ast.Constant,
            ast.Name,
            ast.Load,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.Not,
        )
        for node in ast.walk(parsed):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Unsupported expression: {text!r}")
            if isinstance(node, ast.Name) and node.id not in names:
                raise ValueError(f"Unknown symbol in expression: {node.id}")

        code = compile(parsed, "<synapse-expr>", "eval")
        return eval(code, {"__builtins__": {}, "math": math}, names)

    def _join(self, left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        return f"{left}.{right}"
