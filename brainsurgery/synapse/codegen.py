from __future__ import annotations

import ast
import importlib.resources
import re
from typing import Any

from omegaconf import OmegaConf


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
    standalone: bool = True,
) -> str:
    del standalone  # Emitted models are standalone by default now.

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
    symbols = {k: v for k, v in symbols_raw.items() if isinstance(v, int)}

    emitter = _Emitter(class_name=class_name, spec=spec, symbols=symbols)
    return emitter.render()


class _Emitter:
    def __init__(self, *, class_name: str, spec: dict[str, Any], symbols: dict[str, int]) -> None:
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
                f"        self._symbols: dict[str, int] = {repr(self.symbols)}",
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
            "    def generate(self, input_ids: torch.Tensor, *, eos_token_id: int, max_len: int) -> torch.Tensor:",
            "        if input_ids.ndim != 2:",
            "            raise ValueError('input_ids must be rank-2 [batch, seq]')",
            "        if max_len <= 0:",
            "            raise ValueError('max_len must be > 0')",
            "        if input_ids.size(1) >= max_len:",
            "            return input_ids[:, :max_len]",
            "",
            "        generated = input_ids",
            "        past_key_values = None",
            "        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)",
            "        was_training = self.training",
            "        self.eval()",
            "        try:",
            "            with torch.no_grad():",
            "                while generated.size(1) < max_len and not torch.all(finished):",
            "                    step_input = generated if past_key_values is None else generated[:, -1:]",
            "                    model_out = self.forward(step_input, past_key_values=past_key_values, use_cache=True)",
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
                loop_var = self._py_name(var_name)
                lines.append(f"{inner_indent}for {loop_var} in range(int({range_code})):")
                saved = env.get(var_name)
                env[var_name] = loop_var
                child_scope = self._fresh("scope")
                lines.append(
                    f"{inner_indent}    {child_scope} = self._join_scope({scope_var}, f'{node_name}.{{{loop_var}}}')"
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
        if op == "linear":
            tie = node_spec.get("tie_weight")
            has_bias = (
                bool(node_spec["bias"]) if "bias" in node_spec else (not isinstance(tie, str))
            )
            explicit_weight = node_spec.get("weight")
            has_explicit_weight = isinstance(explicit_weight, str) and "." in explicit_weight
            if not has_bias and (isinstance(tie, str) or has_explicit_weight):
                return False
        return op in {
            "embedding",
            "linear",
            "conv1d",
            "layernorm",
            "rmsnorm",
            "moe_router_topk",
        }

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
        lines: list[str] = []

        def assign_out_var(out_name: str) -> str:
            existing = env.get(out_name)
            if isinstance(existing, str):
                return existing
            out_var = self._fresh(self._py_name(out_name))
            env[out_name] = out_var
            return out_var

        def infer_param(param_name: str) -> str:
            explicit_params = node_spec.get("params")
            if isinstance(explicit_params, dict) and isinstance(
                explicit_params.get(param_name), str
            ):
                return repr(explicit_params[param_name])
            if isinstance(node_spec.get(param_name), str):
                candidate = node_spec[param_name]
                if "." in candidate:
                    return repr(candidate)
            if param_name == "weight" and isinstance(node_spec.get("share"), str):
                return repr(node_spec["share"])
            return f"self._join_scope({node_path_var}, {param_name!r})"

        def read(name: str) -> str:
            if name not in env:
                raise ValueError(f"Unknown input var {name!r}")
            return env[name]

        if op == "embedding":
            src = read(str(node_spec.get("in")))
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            scale_expr = node_spec.get("scale")
            if scale_expr is None:
                lines.append(
                    f"{indent}{out_var} = F.embedding({src}, self._param({infer_param('weight')}))"
                )
            else:
                scale = self._expr_code(scale_expr, env)
                lines.append(
                    f"{indent}{out_var} = F.embedding({src}, self._param({infer_param('weight')}))"
                )
                lines.append(
                    f"{indent}{out_var} = {out_var} * torch.tensor(float({scale}), dtype={out_var}.dtype, device={out_var}.device)"
                )
            return lines

        if op == "linear":
            src = read(str(node_spec.get("in")))
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            tie = node_spec.get("tie_weight")
            weight_expr = repr(tie) if isinstance(tie, str) else infer_param("weight")
            has_bias = (
                bool(node_spec["bias"]) if "bias" in node_spec else (not isinstance(tie, str))
            )
            if has_bias:
                bias_expr = f"self._state.get({infer_param('bias')})"
            else:
                bias_expr = "None"
            lines.append(
                f"{indent}{out_var} = F.linear({src}, self._param({weight_expr}), {bias_expr})"
            )
            return lines

        if op == "conv1d":
            src = read(str(node_spec.get("in")))
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            w = f"self._param({infer_param('weight')})"
            b = f"self._param({infer_param('bias')})"
            lines.append(f"{indent}{out_var} = torch.matmul({src}, {w}) + {b}")
            return lines

        if op == "layernorm":
            src = read(str(node_spec.get("in")))
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            eps = self._expr_code(node_spec.get("eps", 1e-5), env)
            w = f"self._param({infer_param('weight')})"
            b = f"self._param({infer_param('bias')})"
            lines.append(
                f"{indent}{out_var} = F.layer_norm({src}, ({src}.shape[-1],), weight={w}, bias={b}, eps=float({eps}))"
            )
            return lines

        if op == "rmsnorm":
            src = read(str(node_spec.get("in")))
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            eps = self._expr_code(node_spec.get("eps", 1e-6), env)
            tmp = self._fresh("xnorm")
            cast_float = bool(node_spec.get("cast_float", False))
            unit_offset = bool(node_spec.get("unit_offset", False))
            x_norm_src = f"{src}.float()" if cast_float else src
            w_src = (
                f"self._param({infer_param('weight')}).float()"
                if cast_float
                else f"self._param({infer_param('weight')})"
            )
            lines.append(
                f"{indent}{tmp} = {x_norm_src} * torch.rsqrt(torch.mean({x_norm_src} * {x_norm_src}, dim=-1, keepdim=True) + float({eps}))"
            )
            if unit_offset:
                lines.append(f"{indent}{out_var} = {tmp} * (1.0 + {w_src})")
            else:
                lines.append(f"{indent}{out_var} = {tmp} * {w_src}")
            if cast_float:
                lines.append(f"{indent}{out_var} = {out_var}.type_as({src})")
            return lines

        if op == "activation":
            src = read(str(node_spec.get("in")))
            out_name = str(node_spec.get("out"))
            kind = node_spec.get("kind", "gelu")
            out_var = assign_out_var(out_name)
            if kind in {"gelu_new", "gelu_pytorch_tanh"}:
                lines.append(
                    f"{indent}{out_var} = 0.5 * {src} * (1.0 + torch.tanh(0.7978845608028654 * ({src} + 0.044715 * {src} * {src} * {src})))"
                )
            elif kind == "gelu":
                lines.append(f"{indent}{out_var} = F.gelu({src})")
            elif kind == "relu":
                lines.append(f"{indent}{out_var} = F.relu({src})")
            elif kind == "silu":
                lines.append(f"{indent}{out_var} = F.silu({src})")
            else:
                raise ValueError(f"Unsupported activation kind: {kind}")
            return lines

        if op == "moe_router_topk":
            src = read(str(node_spec.get("in")))
            outs = node_spec.get("out")
            if not isinstance(outs, list) or len(outs) != 3:
                raise ValueError(
                    "moe_router_topk expects out=[router_probs,topk_scores,topk_indices]"
                )
            num_experts = self._expr_code(node_spec.get("num_experts"), env)
            k = self._expr_code(node_spec.get("k"), env)
            renorm = bool(node_spec.get("renorm_topk", False))
            softmax_dtype = str(node_spec.get("softmax_dtype", "float32"))
            flat = self._fresh("hidden_flat")
            logits = self._fresh("router_logits")
            probs = assign_out_var(str(outs[0]))
            topv = assign_out_var(str(outs[1]))
            topi = assign_out_var(str(outs[2]))
            bias_expr = "None"
            if node_spec.get("bias", False):
                bias_expr = f"self._state.get({infer_param('bias')})"
            lines.append(f"{indent}{flat} = {src}.reshape(-1, {src}.shape[-1])")
            lines.append(
                f"{indent}{logits} = F.linear({flat}, self._param({infer_param('weight')}), {bias_expr})"
            )
            lines.append(f"{indent}if int({num_experts}) != {logits}.shape[-1]:")
            lines.append(f"{indent}    raise ValueError('moe_router_topk num_experts mismatch')")
            if softmax_dtype == "float32":
                lines.append(f"{indent}{probs} = F.softmax({logits}, dim=-1, dtype=torch.float32)")
            else:
                lines.append(f"{indent}{probs} = F.softmax({logits}, dim=-1)")
            lines.append(f"{indent}{topv}, {topi} = torch.topk({probs}, int({k}), dim=-1)")
            if renorm:
                lines.append(f"{indent}{topv} = {topv} / {topv}.sum(dim=-1, keepdim=True)")
            lines.append(f"{indent}{topv} = {topv}.to({probs}.dtype)")
            return lines

        if op == "moe_experts_ffn":
            ins = node_spec.get("in")
            if not isinstance(ins, list) or len(ins) != 3:
                raise ValueError("moe_experts_ffn expects in=[hidden,topk_scores,topk_indices]")
            src = read(str(ins[0]))
            topk_scores = read(str(ins[1]))
            topk_indices = read(str(ins[2]))
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            num_experts = self._expr_code(node_spec.get("num_experts"), env)
            experts_scope = node_spec.get("experts_scope", "experts")
            if not isinstance(experts_scope, str):
                raise ValueError("moe_experts_ffn experts_scope must be string")
            experts_base = self._fresh("experts_base")
            hidden_flat = self._fresh("hidden_flat")
            final_hidden = self._fresh("final_hidden")
            expert_idx = self._fresh("expert_idx")
            expert_pos = self._fresh("expert_pos")
            token_idx = self._fresh("token_idx")
            topk_pos = self._fresh("topk_pos")
            current = self._fresh("current_state")
            current_hidden = self._fresh("current_hidden")
            gate = self._fresh("gate")
            up = self._fresh("up")
            packed_gate_up = self._fresh("packed_gate_up")
            packed_down = self._fresh("packed_down")
            packed_gate_up_w = self._fresh("packed_gate_up_w")
            packed_down_w = self._fresh("packed_down_w")
            gate_w = self._fresh("gate_w")
            up_w = self._fresh("up_w")
            down_w = self._fresh("down_w")
            gate_name = str(node_spec.get("gate_proj_name", "gate_proj.weight"))
            up_name = str(node_spec.get("up_proj_name", "up_proj.weight"))
            down_name = str(node_spec.get("down_proj_name", "down_proj.weight"))
            activation = str(node_spec.get("activation", "silu"))

            lines.append(f"{indent}{hidden_flat} = {src}.reshape(-1, {src}.shape[-1])")
            lines.append(f"{indent}{final_hidden} = torch.zeros_like({hidden_flat})")
            lines.append(
                f"{indent}{experts_base} = self._join_scope({scope_var}, {experts_scope!r})"
            )
            lines.append(
                f"{indent}{packed_gate_up} = self._join_scope({experts_base}, 'gate_up_proj')"
            )
            lines.append(f"{indent}{packed_down} = self._join_scope({experts_base}, 'down_proj')")
            lines.append(f"{indent}for {expert_idx} in range(int({num_experts})):")
            lines.append(
                f"{indent}    {expert_pos} = ({topk_indices} == {expert_idx}).nonzero(as_tuple=False)"
            )
            lines.append(f"{indent}    if {expert_pos}.numel() == 0:")
            lines.append(f"{indent}        continue")
            lines.append(f"{indent}    {token_idx} = {expert_pos}[:, 0]")
            lines.append(f"{indent}    {topk_pos} = {expert_pos}[:, 1]")
            lines.append(f"{indent}    {current} = {hidden_flat}[{token_idx}]")
            lines.append(
                f"{indent}    if {packed_gate_up} in self._state and {packed_down} in self._state:"
            )
            lines.append(f"{indent}        {packed_gate_up_w} = self._param({packed_gate_up})")
            lines.append(f"{indent}        if {packed_gate_up_w}.ndim == 3:")
            lines.append(
                f"{indent}            {packed_gate_up_w} = {packed_gate_up_w}[{expert_idx}]"
            )
            lines.append(
                f"{indent}        {gate}, {up} = F.linear({current}, {packed_gate_up_w}, None).chunk(2, dim=-1)"
            )
            lines.append(f"{indent}        {packed_down_w} = self._param({packed_down})")
            lines.append(f"{indent}        if {packed_down_w}.ndim == 3:")
            lines.append(f"{indent}            {packed_down_w} = {packed_down_w}[{expert_idx}]")
            lines.append(f"{indent}        {down_w} = {packed_down_w}")
            lines.append(f"{indent}        {current_hidden} = None")
            lines.append(f"{indent}    else:")
            lines.append(
                f"{indent}        {gate_w} = self._param(self._join_scope({experts_base}, f'{{{expert_idx}}}.{gate_name}'))"
            )
            lines.append(
                f"{indent}        {up_w} = self._param(self._join_scope({experts_base}, f'{{{expert_idx}}}.{up_name}'))"
            )
            lines.append(
                f"{indent}        {down_w} = self._param(self._join_scope({experts_base}, f'{{{expert_idx}}}.{down_name}'))"
            )
            lines.append(f"{indent}        {gate} = F.linear({current}, {gate_w}, None)")
            lines.append(f"{indent}        {up} = F.linear({current}, {up_w}, None)")
            if activation in {"gelu_new", "gelu_pytorch_tanh"}:
                lines.append(
                    f"{indent}    {current_hidden} = 0.5 * {gate} * (1.0 + torch.tanh(0.7978845608028654 * ({gate} + 0.044715 * {gate} * {gate} * {gate}))) * {up}"
                )
            elif activation == "gelu":
                lines.append(f"{indent}    {current_hidden} = F.gelu({gate}) * {up}")
            elif activation == "relu":
                lines.append(f"{indent}    {current_hidden} = F.relu({gate}) * {up}")
            elif activation == "silu":
                lines.append(f"{indent}    {current_hidden} = F.silu({gate}) * {up}")
            else:
                raise ValueError(f"Unsupported moe_experts_ffn activation kind: {activation}")
            lines.append(
                f"{indent}    {current_hidden} = F.linear({current_hidden}, {down_w}, None)"
            )
            lines.append(
                f"{indent}    {current_hidden} = {current_hidden} * {topk_scores}[{token_idx}, {topk_pos}].unsqueeze(-1).to({current_hidden}.dtype)"
            )
            lines.append(
                f"{indent}    {final_hidden}.index_add_(0, {token_idx}, {current_hidden}.to({final_hidden}.dtype))"
            )
            lines.append(f"{indent}{out_var} = {final_hidden}.reshape_as({src})")
            return lines

        if op in {"add", "mul"}:
            inputs = node_spec.get("in")
            if not isinstance(inputs, list) or len(inputs) != 2:
                raise ValueError(f"{op} expects two inputs")
            a = read(str(inputs[0]))
            b = read(str(inputs[1]))
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            sym = "+" if op == "add" else "*"
            lines.append(f"{indent}{out_var} = {a} {sym} {b}")
            return lines

        if op == "arange_positions":
            src = read(str(node_spec.get("in")))
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            past_var = env.get("past_key_values")
            if isinstance(past_var, str):
                offset = self._fresh("pos_offset")
                lines.append(
                    f"{indent}{offset} = 0 if {past_var} is None else int({past_var}[0][0].shape[-2])"
                )
                lines.append(
                    f"{indent}{out_var} = torch.arange({offset}, {offset} + {src}.shape[1], device={src}.device, dtype=torch.long).unsqueeze(0)"
                )
            else:
                lines.append(
                    f"{indent}{out_var} = torch.arange({src}.shape[1], device={src}.device, dtype=torch.long).unsqueeze(0)"
                )
            return lines

        if op == "split_last_dim":
            src = read(str(node_spec.get("in")))
            outs = node_spec.get("out")
            sizes = node_spec.get("sizes")
            if not isinstance(outs, list) or not isinstance(sizes, list):
                raise ValueError("split_last_dim requires list out and sizes")
            size_code = ", ".join([self._expr_code(s, env) for s in sizes])
            tmp = self._fresh("split")
            lines.append(f"{indent}{tmp} = torch.split({src}, [{size_code}], dim=-1)")
            for idx, out_name in enumerate(outs):
                out_var = assign_out_var(str(out_name))
                lines.append(f"{indent}{out_var} = {tmp}[{idx}]")
            return lines

        if op == "reshape_heads_triplet":
            ins = node_spec.get("in")
            outs = node_spec.get("out")
            if (
                not isinstance(ins, list)
                or not isinstance(outs, list)
                or len(ins) != 3
                or len(outs) != 3
            ):
                raise ValueError("reshape_heads_triplet requires 3 inputs and outputs")
            heads = self._expr_code(node_spec.get("heads"), env)
            head_dim = self._expr_code(node_spec.get("head_dim"), env)
            for src_name, out_name in zip(ins, outs, strict=True):
                src = read(str(src_name))
                out_var = assign_out_var(str(out_name))
                lines.append(
                    f"{indent}{out_var} = {src}.view({src}.shape[0], {src}.shape[1], int({heads}), int({head_dim})).transpose(1, 2)"
                )
            return lines

        if op == "reshape_heads":
            src = read(str(node_spec.get("in")))
            out_name = str(node_spec.get("out"))
            heads = self._expr_code(node_spec.get("heads"), env)
            head_dim = self._expr_code(node_spec.get("head_dim"), env)
            out_var = assign_out_var(out_name)
            lines.append(
                f"{indent}{out_var} = {src}.view({src}.shape[0], {src}.shape[1], int({heads}), int({head_dim})).transpose(1, 2)"
            )
            return lines

        if op == "apply_rope_pair":
            ins = node_spec.get("in")
            outs = node_spec.get("out")
            if (
                not isinstance(ins, list)
                or len(ins) != 2
                or not isinstance(outs, list)
                or len(outs) != 2
            ):
                raise ValueError("apply_rope_pair expects in=[q,k], out=[q_rot,k_rot]")
            q = read(str(ins[0]))
            k = read(str(ins[1]))
            q_out = assign_out_var(str(outs[0]))
            k_out = assign_out_var(str(outs[1]))
            theta = self._expr_code(node_spec.get("theta", 10000.0), env)
            half = self._fresh("half")
            inv_freq = self._fresh("inv_freq")
            pos = self._fresh("pos")
            ang = self._fresh("ang")
            cos = self._fresh("cos")
            sin = self._fresh("sin")
            q1 = self._fresh("q1")
            q2 = self._fresh("q2")
            k1 = self._fresh("k1")
            k2 = self._fresh("k2")
            lines.append(f"{indent}{half} = {q}.shape[-1] // 2")
            lines.append(
                f"{indent}{inv_freq} = 1.0 / (float({theta}) ** (torch.arange(0, {half}, device={q}.device, dtype={q}.dtype) / float({half})))"
            )
            lines.append(
                f"{indent}{pos} = torch.arange({q}.shape[-2], device={q}.device, dtype={q}.dtype)"
            )
            lines.append(f"{indent}{ang} = torch.einsum('t,d->td', {pos}, {inv_freq})")
            lines.append(f"{indent}{cos} = torch.cos({ang})[None, None, :, :]")
            lines.append(f"{indent}{sin} = torch.sin({ang})[None, None, :, :]")
            lines.append(f"{indent}{q1} = {q}[..., :{half}]")
            lines.append(f"{indent}{q2} = {q}[..., {half}: 2 * {half}]")
            lines.append(f"{indent}{k1} = {k}[..., :{half}]")
            lines.append(f"{indent}{k2} = {k}[..., {half}: 2 * {half}]")
            lines.append(
                f"{indent}{q_out} = torch.cat([{q1} * {cos} - {q2} * {sin}, {q1} * {sin} + {q2} * {cos}], dim=-1)"
            )
            lines.append(
                f"{indent}{k_out} = torch.cat([{k1} * {cos} - {k2} * {sin}, {k1} * {sin} + {k2} * {cos}], dim=-1)"
            )
            return lines

        if op == "repeat_kv":
            src = read(str(node_spec.get("in")))
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            repeats = node_spec.get("repeats")
            if repeats is None:
                heads = self._expr_code(node_spec.get("heads"), env)
                kv_heads = self._expr_code(node_spec.get("kv_heads"), env)
                repeats_code = f"(int({heads}) // int({kv_heads}))"
            else:
                repeats_code = self._expr_code(repeats, env)
            n_rep = self._fresh("n_rep")
            lines.append(f"{indent}{n_rep} = int({repeats_code})")
            lines.append(f"{indent}if {n_rep} == 1:")
            lines.append(f"{indent}    {out_var} = {src}")
            lines.append(f"{indent}else:")
            lines.append(
                f"{indent}    {out_var} = {src}[:, :, None, :, :].expand({src}.shape[0], {src}.shape[1], {n_rep}, {src}.shape[2], {src}.shape[3]).reshape({src}.shape[0], {src}.shape[1] * {n_rep}, {src}.shape[2], {src}.shape[3])"
            )
            return lines

        if op == "merge_heads":
            src = read(str(node_spec.get("in")))
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            lines.append(
                f"{indent}{out_var} = {src}.transpose(1, 2).contiguous().view({src}.shape[0], {src}.shape[2], {src}.shape[1] * {src}.shape[3])"
            )
            return lines

        if op == "attention":
            ins = node_spec.get("in")
            if not isinstance(ins, list) or len(ins) != 3:
                raise ValueError("attention expects 3 inputs")
            q = read(str(ins[0]))
            k = read(str(ins[1]))
            v = read(str(ins[2]))
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            mask_name = node_spec.get("mask")
            mask_expr = "None"
            if isinstance(mask_name, str) and mask_name in env:
                mask_expr = env[mask_name]
            if bool(node_spec.get("causal", False)):
                is_causal = f"({q}.shape[-2] > 1 and {mask_expr} is None)"
            else:
                is_causal = "False"
            scale_value = node_spec.get("scale")
            scale_expr = "None" if scale_value is None else self._expr_code(scale_value, env)
            lines.append(
                f"{indent}{out_var} = F.scaled_dot_product_attention({q}, {k}, {v}, attn_mask={mask_expr}, dropout_p=0.0, is_causal={is_causal}, scale={scale_expr})"
            )
            return lines

        if op == "causal_mask":
            q = read(str(node_spec.get("in")))
            k_name = node_spec.get("key")
            k = read(str(k_name)) if isinstance(k_name, str) else q
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            q_len = self._fresh("q_len")
            k_len = self._fresh("k_len")
            i_idx = self._fresh("i_idx")
            j_idx = self._fresh("j_idx")
            keep = self._fresh("keep")
            mask_val = self._fresh("mask_val")
            window_expr = node_spec.get("window")
            if window_expr is None:
                lines.append(f"{indent}{out_var} = None")
                return lines
            lines.append(f"{indent}{q_len} = {q}.shape[-2]")
            lines.append(f"{indent}{k_len} = {k}.shape[-2]")
            lines.append(f"{indent}{i_idx} = torch.arange({q_len}, device={q}.device).unsqueeze(1)")
            lines.append(f"{indent}{j_idx} = torch.arange({k_len}, device={q}.device).unsqueeze(0)")
            lines.append(f"{indent}{keep} = ({j_idx} <= {i_idx})")
            win = self._fresh("window")
            window_code = self._expr_code(window_expr, env)
            lines.append(f"{indent}{win} = int({window_code})")
            lines.append(f"{indent}if {win} >= {k_len} and {q_len} == {k_len}:")
            lines.append(f"{indent}    {out_var} = None")
            lines.append(f"{indent}else:")
            lines.append(f"{indent}    {keep} = {keep} & ({j_idx} >= ({i_idx} - {win} + 1))")
            lines.append(f"{indent}    {mask_val} = torch.finfo({q}.dtype).min")
            lines.append(
                f"{indent}    {out_var} = torch.where({keep}, torch.zeros((), dtype={q}.dtype, device={q}.device), torch.full((), {mask_val}, dtype={q}.dtype, device={q}.device)).view(1, 1, {q_len}, {k_len})"
            )
            return lines

        if op == "index":
            ins = node_spec.get("in")
            if not isinstance(ins, list) or len(ins) != 2:
                raise ValueError("index expects [collection,index]")
            coll = read(str(ins[0]))
            idx_expr = self._expr_code(ins[1], env)
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            lines.append(f"{indent}{out_var} = None if {coll} is None else {coll}[int({idx_expr})]")
            return lines

        if op == "init_list":
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            lines.append(f"{indent}{out_var} = []")
            return lines

        if op == "append":
            ins = node_spec.get("in")
            if not isinstance(ins, list) or len(ins) != 2:
                raise ValueError("append expects [list,item]")
            base = read(str(ins[0]))
            item = read(str(ins[1]))
            out_name = str(node_spec.get("out"))
            out_var = assign_out_var(out_name)
            lines.append(f"{indent}{out_var} = list({base})")
            lines.append(f"{indent}{out_var}.append({item})")
            return lines

        if op == "kv_cache_update":
            ins = node_spec.get("in")
            outs = node_spec.get("out")
            if (
                not isinstance(ins, list)
                or len(ins) != 3
                or not isinstance(outs, list)
                or len(outs) != 3
            ):
                raise ValueError("kv_cache_update expects in=[past,k,v], out=[k_all,v_all,present]")
            past = read(str(ins[0])) if str(ins[0]) in env else "None"
            k_new = read(str(ins[1]))
            v_new = read(str(ins[2]))
            k_all = assign_out_var(str(outs[0]))
            v_all = assign_out_var(str(outs[1]))
            present = assign_out_var(str(outs[2]))
            lines.append(f"{indent}if {past} is None:")
            lines.append(f"{indent}    {k_all} = {k_new}")
            lines.append(f"{indent}    {v_all} = {v_new}")
            lines.append(f"{indent}else:")
            lines.append(f"{indent}    {k_all} = torch.cat([{past}[0], {k_new}], dim=-2)")
            lines.append(f"{indent}    {v_all} = torch.cat([{past}[1], {v_new}], dim=-2)")
            lines.append(f"{indent}{present} = ({k_all}, {v_all})")
            return lines

        if op == "coalesce_triplet":
            ins = node_spec.get("in")
            outs = node_spec.get("out")
            if (
                not isinstance(ins, list)
                or len(ins) != 4
                or not isinstance(outs, list)
                or len(outs) != 2
            ):
                raise ValueError("coalesce_triplet expects in=[k_all,v_all,k,v], out=[k_ctx,v_ctx]")
            k_all = read(str(ins[0])) if str(ins[0]) in env else "None"
            v_all = read(str(ins[1])) if str(ins[1]) in env else "None"
            k = read(str(ins[2]))
            v = read(str(ins[3]))
            k_ctx = assign_out_var(str(outs[0]))
            v_ctx = assign_out_var(str(outs[1]))
            if k_all != "None":
                lines.append(
                    f"{indent}{k_ctx} = ({k_all} if ('{k_all}' in locals() and {k_all} is not None) else {k})"
                )
            else:
                lines.append(f"{indent}{k_ctx} = {k}")
            if v_all != "None":
                lines.append(
                    f"{indent}{v_ctx} = ({v_all} if ('{v_all}' in locals() and {v_all} is not None) else {v})"
                )
            else:
                lines.append(f"{indent}{v_ctx} = {v}")
            return lines

        raise NotImplementedError(f"Unsupported op in codegen compiler: {op}")

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
    known_runtime_builtin_ops = {
        "embedding",
        "linear",
        "conv1d",
        "layernorm",
        "rmsnorm",
        "activation",
        "add",
        "mul",
        "arange_positions",
        "split_last_dim",
        "reshape_heads_triplet",
        "reshape_heads",
        "apply_rope_pair",
        "repeat_kv",
        "merge_heads",
        "attention",
        "causal_mask",
        "moe_router_topk",
        "moe_experts_ffn",
        "index",
        "init_list",
        "append",
        "kv_cache_update",
        "coalesce_triplet",
    }

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
