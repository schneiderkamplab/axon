from __future__ import annotations

import ast
import importlib.resources
import re
from typing import Any

from omegaconf import OmegaConf

from .ops import OP_MODULES, get_op_module


def load_synapse_torch_op_map() -> dict[str, Any]:
    data_text = (
        importlib.resources.files("brainsurgery.synapse")
        .joinpath("torch_op_map.yaml")
        .read_text(encoding="utf-8")
    )
    loaded = OmegaConf.create(data_text)
    data = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("synapse torch op map must be a mapping")
    return {str(key): value for key, value in data.items()}


def emit_model_code_from_synapse_spec(
    spec: dict[str, Any],
    *,
    class_name: str = "GeneratedSynapseModel",
    op_map: dict[str, Any] | None = None,
) -> str:
    if not class_name.isidentifier():
        raise ValueError(f"Invalid class name: {class_name!r}")
    if spec.get("synapse") != 1:
        raise ValueError("Only synapse: 1 specs are supported")

    resolved_op_map = load_synapse_torch_op_map() if op_map is None else op_map
    _validate_spec_ops(spec, resolved_op_map)

    model = spec.get("model")
    if not isinstance(model, dict):
        raise ValueError("spec.model must be a mapping")

    symbols_raw = model.get("symbols", {})
    symbols = {k: v for k, v in symbols_raw.items() if isinstance(v, (int, float, bool))}

    emitter = _Emitter(class_name=class_name, spec=spec, symbols=symbols)
    return emitter.render()


class _Emitter:
    def __init__(
        self, *, class_name: str, spec: dict[str, Any], symbols: dict[str, int | float | bool]
    ) -> None:
        self.class_name = class_name
        self.spec = spec
        self.model = spec["model"]
        self.symbols = symbols
        self._counter = 0

    def render(self) -> str:
        lines: list[str] = []
        lines.extend(
            [
                "from __future__ import annotations",
                "",
                "from typing import Any",
                "",
                "import torch",
                "from torch import nn",
                "from torch.nn import functional as F",
                "",
                "",
                f"class {self.class_name}(nn.Module):",
                "    def __init__(self, state_dict: dict[str, torch.Tensor] | None = None) -> None:",
                "        super().__init__()",
                "        self._state: dict[str, torch.Tensor] = {}",
                f"        self._symbols: dict[str, int | float | bool] = {repr(self.symbols)}",
                "        if state_dict is not None:",
                "            self.load_state_dict_tensors(state_dict)",
                "",
                "    @classmethod",
                '    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "'
                + self.class_name
                + '":',
                "        return cls(state_dict=state_dict)",
                "",
                "    def load_state_dict_tensors(self, state_dict: dict[str, torch.Tensor]) -> None:",
                "        self._state = dict(state_dict)",
                "",
                "    def _param(self, path: str) -> torch.Tensor:",
                "        return self._state[path]",
                "",
                "    def _join_scope(self, left: str, right: str) -> str:",
                "        if not left:",
                "            return right",
                "        if not right:",
                "            return left",
                '        return f"{left}.{right}"',
                "",
                "    def _safe_get(self, env: dict[str, Any], name: str) -> Any:",
                "        if name not in env:",
                '            raise ValueError(f"Missing variable in graph env: {name}")',
                "        return env[name]",
                "",
            ]
        )

        blocks = self.model.get("blocks", {})
        if isinstance(blocks, dict):
            for block_name, block_spec in blocks.items():
                lines.extend(self._render_block_method(block_name, block_spec))

        lines.extend(self._render_forward())
        lines.extend(self._render_generate())
        return "\n".join(lines) + "\n"

    def _render_block_method(self, block_name: str, block_spec: Any) -> list[str]:
        if not isinstance(block_spec, dict):
            raise ValueError("block spec must be mapping")
        inputs = block_spec.get("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError("block inputs must be mapping")
        graph = block_spec.get("graph")
        if not isinstance(graph, list):
            raise ValueError("block graph must be list")
        outputs = block_spec.get("outputs", {})
        if not isinstance(outputs, dict):
            raise ValueError("block outputs must be mapping")

        arg_names = [self._py_name(name) for name in inputs]
        env: dict[str, str] = {name: py for name, py in zip(inputs, arg_names, strict=True)}

        sig = ", ".join(["self", *arg_names, "scope: str"])
        lines = [f"    def _block_{self._py_name(block_name)}({sig}) -> tuple[Any, ...]:"]
        lines.append("        emitter = self")
        lines.append("        env: dict[str, Any] = {}")
        for syn_name, py_name in env.items():
            lines.append(f"        env[{syn_name!r}] = {py_name}")

        body = self._compile_graph(graph=graph, env=env, scope_var="scope", indent="        ")
        lines.extend(body)

        return_values: list[str] = []
        for _, ref in outputs.items():
            if isinstance(ref, str):
                return_values.append(env[ref])
            else:
                raise ValueError("block outputs currently support string refs only")
        if len(return_values) == 1:
            lines.append(f"        return {return_values[0]}")
        else:
            tuple_expr = ", ".join(return_values)
            lines.append(f"        return ({tuple_expr})")
        lines.append("")
        return lines

    def _render_forward(self) -> list[str]:
        graph = self.model.get("graph")
        if not isinstance(graph, list):
            raise ValueError("model.graph must be list")
        inputs = self.model.get("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError("model.inputs must be mapping")
        outputs = self.model.get("outputs", {})
        if not isinstance(outputs, dict):
            raise ValueError("model.outputs must be mapping")

        lines = [
            "    def forward(self, input_ids: torch.Tensor | None = None, **inputs: Any) -> Any:",
            "        if input_ids is not None:",
            "            inputs = {'input_ids': input_ids, **inputs}",
            "        env: dict[str, Any] = dict(inputs)",
            "        scope = ''",
            "        emitter = self",
        ]

        env: dict[str, str] = {}
        for name, input_spec in inputs.items():
            is_optional = isinstance(input_spec, dict) and bool(input_spec.get("optional", False))
            py_name = self._py_name(name)
            if is_optional:
                lines.append(f"        {py_name} = env.get({name!r})")
            else:
                lines.append(f"        {py_name} = self._safe_get(env, {name!r})")
            env[name] = py_name

        lines.extend(
            self._compile_graph(graph=graph, env=env, scope_var="scope", indent="        ")
        )

        lines.append("        outputs: dict[str, Any] = {}")
        for out_name, ref in outputs.items():
            if isinstance(ref, str):
                lines.append(f"        outputs[{out_name!r}] = {env[ref]}")
            elif isinstance(ref, dict) and isinstance(ref.get("from"), str):
                lines.append(f"        outputs[{out_name!r}] = {env[ref['from']]}")
            else:
                raise ValueError(f"Unsupported output ref shape: {ref!r}")

        lines.append("        if 'logits' in outputs and len(outputs) == 1:")
        lines.append("            return outputs['logits']")
        lines.append("        return outputs")
        lines.append("")
        return lines

    def _render_generate(self) -> list[str]:
        return [
            "    def generate(self, input_ids: torch.Tensor, *, eos_token_id: int, max_len: int, attention_mask: torch.Tensor | None = None) -> torch.Tensor:",
            "        if input_ids.ndim != 2:",
            "            raise ValueError('input_ids must be rank-2 [batch, seq]')",
            "        if max_len <= 0:",
            "            raise ValueError('max_len must be > 0')",
            "        if attention_mask is not None:",
            "            if attention_mask.ndim != 2:",
            "                raise ValueError('attention_mask must be rank-2 [batch, seq]')",
            "            if attention_mask.shape != input_ids.shape:",
            "                raise ValueError('attention_mask must have same shape as input_ids')",
            "        if input_ids.size(1) >= max_len:",
            "            return input_ids[:, :max_len]",
            "",
            "        generated = input_ids",
            "        generated_mask = attention_mask",
            "        past_key_values = None",
            "        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)",
            "        was_training = self.training",
            "        self.eval()",
            "        try:",
            "            with torch.no_grad():",
            "                while generated.size(1) < max_len and not torch.all(finished):",
            "                    step_input = generated if past_key_values is None else generated[:, -1:]",
            "                    if generated_mask is None:",
            "                        model_out = self.forward(step_input, past_key_values=past_key_values, use_cache=True)",
            "                    else:",
            "                        model_out = self.forward(step_input, attention_mask=generated_mask, past_key_values=past_key_values, use_cache=True)",
            "                    if isinstance(model_out, dict):",
            "                        logits = model_out['logits']",
            "                        if 'past_key_values' in model_out:",
            "                            past_key_values = model_out['past_key_values']",
            "                    else:",
            "                        logits = model_out",
            "                    next_token = torch.argmax(logits[:, -1, :], dim=-1)",
            "                    next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)",
            "                    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)",
            "                    finished = torch.logical_or(finished, next_token == eos_token_id)",
            "                    if generated_mask is not None:",
            "                        next_mask = torch.ones((generated_mask.shape[0], 1), dtype=generated_mask.dtype, device=generated_mask.device)",
            "                        generated_mask = torch.cat([generated_mask, next_mask], dim=1)",
            "        finally:",
            "            if was_training:",
            "                self.train()",
            "        return generated",
            "",
        ]

    def _compile_graph(
        self, *, graph: list[Any], env: dict[str, str], scope_var: str, indent: str
    ) -> list[str]:
        lines: list[str] = []
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"Invalid graph item: {item!r}")
            node_name, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                raise ValueError(f"Invalid node spec: {node_spec!r}")

            when = node_spec.get("when")
            inner_indent = indent
            if when is not None:
                produced_names = self._node_output_names(node_spec)
                for produced_name in produced_names:
                    existing = env.get(produced_name)
                    if isinstance(existing, str):
                        # Preserve the previously bound value when the conditional does not execute.
                        continue
                    out_var = self._fresh(self._py_name(produced_name))
                    lines.append(f"{indent}{out_var} = None")
                    env[produced_name] = out_var
                cond = self._expr_code(when, env)
                lines.append(f"{indent}if {cond}:")
                inner_indent = indent + "    "

            op = node_spec.get("op")
            if op == "repeat":
                var_name = node_spec.get("var")
                if not isinstance(var_name, str):
                    raise ValueError("repeat requires string var")
                range_code = self._expr_code(node_spec.get("range"), env)
                start_code = self._expr_code(node_spec.get("start", 0), env)
                loop_var = self._py_name(var_name)
                lines.append(f"{inner_indent}for {loop_var} in range(int({range_code})):")
                saved = env.get(var_name)
                iter_value = self._fresh(self._py_name(var_name))
                lines.append(
                    f"{inner_indent}    {iter_value} = int({start_code}) + int({loop_var})"
                )
                env[var_name] = iter_value
                child_scope = self._fresh("scope")
                lines.append(
                    f"{inner_indent}    {child_scope} = self._join_scope({scope_var}, f'{node_name}.{{{iter_value}}}')"
                )
                body = node_spec.get("body")
                if not isinstance(body, list):
                    raise ValueError("repeat requires list body")
                lines.extend(
                    self._compile_graph(
                        graph=body, env=env, scope_var=child_scope, indent=inner_indent + "    "
                    )
                )
                if saved is None:
                    env.pop(var_name, None)
                else:
                    env[var_name] = saved
                continue

            if "use" in node_spec:
                lines.extend(
                    self._compile_block_call(
                        node_spec=node_spec, env=env, scope_var=scope_var, indent=inner_indent
                    )
                )
                continue

            if "graph" in node_spec and op is None:
                nested = node_spec.get("graph")
                if not isinstance(nested, list):
                    raise ValueError("node graph must be list")
                child_scope = self._fresh("scope")
                lines.append(
                    f"{inner_indent}{child_scope} = self._join_scope({scope_var}, {node_name!r})"
                )
                lines.extend(
                    self._compile_graph(
                        graph=nested, env=env, scope_var=child_scope, indent=inner_indent
                    )
                )
                continue

            if not isinstance(op, str):
                raise ValueError(f"node {node_name!r} missing op")

            node_path = scope_var
            if self._op_uses_node_path(op, node_spec):
                node_path = self._fresh("node_path")
                lines.append(
                    f"{inner_indent}{node_path} = self._join_scope({scope_var}, {node_name!r})"
                )
            lines.extend(
                self._compile_op(
                    op=op,
                    node_spec=node_spec,
                    env=env,
                    node_path_var=node_path,
                    scope_var=scope_var,
                    indent=inner_indent,
                )
            )
        return lines

    def _op_uses_node_path(self, op: str, node_spec: dict[str, Any]) -> bool:
        op_module = get_op_module(op)
        if op_module is None:
            raise NotImplementedError(f"Unsupported op in codegen compiler: {op}")
        return bool(op_module.uses_node_path(self, node_spec))

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

    def _compile_block_call(
        self, *, node_spec: dict[str, Any], env: dict[str, str], scope_var: str, indent: str
    ) -> list[str]:
        block_name = node_spec.get("use")
        if not isinstance(block_name, str):
            raise ValueError("use must be a string block name")

        in_bindings = node_spec.get("in", {})
        if not isinstance(in_bindings, dict):
            raise ValueError("block use in must be mapping")
        arg_codes: list[str] = []
        for block_input_name, src in in_bindings.items():
            if isinstance(src, str) and src in env:
                arg_codes.append(f"{block_input_name}={env[src]}")
            else:
                arg_codes.append(f"{block_input_name}={self._expr_code(src, env)}")

        out_bindings = node_spec.get("out", {})
        if not isinstance(out_bindings, dict):
            raise ValueError("block use out must be mapping")

        tmp_vars: list[str] = []
        for block_out_name in out_bindings:
            var = self._fresh(self._py_name(block_out_name))
            tmp_vars.append(var)

        if len(tmp_vars) == 1:
            call_line = (
                f"{indent}{tmp_vars[0]} = self._block_{self._py_name(block_name)}("
                + ", ".join(arg_codes)
                + f", scope={scope_var})"
            )
        else:
            call_line = (
                f"{indent}{', '.join(tmp_vars)} = self._block_{self._py_name(block_name)}("
                + ", ".join(arg_codes)
                + f", scope={scope_var})"
            )

        lines = [call_line]
        for (block_out_name, dst_name), tmp in zip(out_bindings.items(), tmp_vars, strict=True):
            existing = env.get(dst_name)
            dst_var = (
                existing if isinstance(existing, str) else self._fresh(self._py_name(dst_name))
            )
            lines.append(f"{indent}{dst_var} = {tmp}")
            env[dst_name] = dst_var
        return lines

    def _compile_op(
        self,
        *,
        op: str,
        node_spec: dict[str, Any],
        env: dict[str, str],
        node_path_var: str,
        scope_var: str,
        indent: str,
    ) -> list[str]:
        op_module = get_op_module(op)
        if op_module is None:
            raise NotImplementedError(f"Unsupported op in codegen compiler: {op}")
        return op_module.compile(
            self,
            node_spec,
            env,
            node_path_var=node_path_var,
            scope_var=scope_var,
            indent=indent,
        )

    def _assign_out_var(self, env: dict[str, str], out_name: str) -> str:
        existing = env.get(out_name)
        if isinstance(existing, str):
            return existing
        out_var = self._fresh(self._py_name(out_name))
        env[out_name] = out_var
        return out_var

    def _infer_param_expr(
        self, node_spec: dict[str, Any], node_path_var: str, param_name: str
    ) -> str:
        explicit_params = node_spec.get("params")
        if isinstance(explicit_params, dict) and isinstance(explicit_params.get(param_name), str):
            return repr(explicit_params[param_name])
        if isinstance(node_spec.get(param_name), str):
            candidate = node_spec[param_name]
            if "." in candidate:
                return repr(candidate)
        if param_name == "weight" and isinstance(node_spec.get("share"), str):
            return repr(node_spec["share"])
        return f"self._join_scope({node_path_var}, {param_name!r})"

    def _read_env_var(self, env: dict[str, str], name: str) -> str:
        if name not in env:
            raise ValueError(f"Unknown input var {name!r}")
        return env[name]

    def _expr_code(self, expr: Any, env: dict[str, str]) -> str:
        if expr is None:
            return "None"
        if isinstance(expr, (int, float, bool)):
            return repr(expr)
        if isinstance(expr, str):
            token = expr.strip()
            if token in env:
                return env[token]
            if token in self.symbols:
                return repr(self.symbols[token])
            if token.lower() in {"true", "false", "null"}:
                return {"true": "True", "false": "False", "null": "None"}[token.lower()]
            numeric = self._try_eval_numeric(token)
            if numeric is not None:
                return repr(numeric)
            return self._substitute_expr_names(token, env)
        return repr(expr)

    def _substitute_expr_names(self, text: str, env: dict[str, str]) -> str:
        rewritten = text
        for name, py_name in sorted(env.items(), key=lambda kv: len(kv[0]), reverse=True):
            rewritten = re.sub(rf"\b{re.escape(name)}\b", py_name, rewritten)
        for name, value in sorted(self.symbols.items(), key=lambda kv: len(kv[0]), reverse=True):
            rewritten = re.sub(rf"\b{re.escape(name)}\b", repr(value), rewritten)
        return rewritten

    def _try_eval_numeric(self, text: str) -> int | float | None:
        names = dict(self.symbols)
        try:
            parsed = ast.parse(text, mode="eval")
        except SyntaxError:
            return None

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
        )
        for node in ast.walk(parsed):
            if not isinstance(node, allowed_nodes):
                return None
            if isinstance(node, ast.Name) and node.id not in names:
                return None
        try:
            value = eval(compile(parsed, "<synapse-expr>", "eval"), {"__builtins__": {}}, names)
        except Exception:
            return None
        if isinstance(value, (int, float)):
            return value
        return None

    def _py_name(self, value: str) -> str:
        name = re.sub(r"[^0-9A-Za-z_]", "_", value)
        if not name:
            name = "v"
        if name[0].isdigit():
            name = f"v_{name}"
        return name

    def _fresh(self, base: str) -> str:
        self._counter += 1
        return f"{base}_{self._counter}"


def _validate_spec_ops(spec: dict[str, Any], op_map: dict[str, Any]) -> None:
    ops = op_map.get("ops")
    if not isinstance(ops, dict):
        raise ValueError("op map must contain mapping key 'ops'")

    known_control_ops = {
        "repeat",
    }
    known_runtime_builtin_ops = set(OP_MODULES.keys())

    def _walk_graph(graph: list[Any]) -> None:
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"Invalid graph item: {item!r}")
            _, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                raise ValueError(f"Invalid node spec: {node_spec!r}")

            op = node_spec.get("op")
            if isinstance(op, str):
                if (
                    op not in known_control_ops
                    and op not in known_runtime_builtin_ops
                    and op not in ops
                ):
                    raise ValueError(f"Unsupported op in spec: {op!r}")

            if "graph" in node_spec:
                nested = node_spec["graph"]
                if not isinstance(nested, list):
                    raise ValueError("node 'graph' must be a list")
                _walk_graph(nested)

            if op == "repeat":
                body = node_spec.get("body")
                if not isinstance(body, list):
                    raise ValueError("repeat node requires list 'body'")
                _walk_graph(body)

    model = spec.get("model")
    if not isinstance(model, dict):
        raise ValueError("spec.model must be a mapping")

    graph = model.get("graph")
    if not isinstance(graph, list):
        raise ValueError("model.graph must be a list")
    _walk_graph(graph)

    blocks = model.get("blocks", {})
    if not isinstance(blocks, dict):
        raise ValueError("model.blocks must be a mapping when present")
    for block in blocks.values():
        if not isinstance(block, dict):
            raise ValueError("block spec must be mapping")
        block_graph = block.get("graph")
        if not isinstance(block_graph, list):
            raise ValueError("block.graph must be list")
        _walk_graph(block_graph)
