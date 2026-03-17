from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F


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
        symbols = {k: v for k, v in symbols_raw.items() if isinstance(v, int)}
        blocks = model.get("blocks", {})

        env: dict[str, Any] = dict(inputs)
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

    def generate(self, input_ids: torch.Tensor, *, eos_token_id: int, max_len: int) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be rank-2 [batch, seq]")
        if max_len <= 0:
            raise ValueError("max_len must be > 0")
        if input_ids.size(1) >= max_len:
            return input_ids[:, :max_len]

        generated = input_ids
        finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                while generated.size(1) < max_len and not torch.all(finished):
                    logits = self.forward(generated)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    next_token = torch.where(
                        finished,
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )
                    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                    finished = torch.logical_or(finished, next_token == eos_token_id)
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
        symbols: dict[str, int],
        blocks: dict[str, Any],
    ) -> None:
        for item in graph:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"Invalid graph item: {item!r}")
            node_name, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                raise ValueError(f"Node spec for {node_name!r} must be mapping")

            if not self._check_when(node_spec.get("when"), env, symbols):
                continue

            op = node_spec.get("op")
            if op == "repeat":
                range_value = self._eval_expr(node_spec.get("range"), env, symbols)
                if not isinstance(range_value, int):
                    raise ValueError(f"repeat range must resolve to int, got {range_value!r}")
                var_name = node_spec.get("var")
                if not isinstance(var_name, str):
                    raise ValueError("repeat requires string 'var'")
                body = node_spec.get("body")
                if not isinstance(body, list):
                    raise ValueError("repeat requires list 'body'")
                for i in range(range_value):
                    env[var_name] = i
                    repeat_scope = self._join(scope, f"{node_name}.{i}")
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
        symbols: dict[str, int],
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
            if isinstance(src_name, str):
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
            env[dst_name] = block_env[block_out_name]

    def _execute_op(
        self,
        op: str,
        node_spec: dict[str, Any],
        env: dict[str, Any],
        *,
        node_path: str,
        scope: str,
        symbols: dict[str, int],
    ) -> None:
        if op == "embedding":
            x = self._read_tensor_input(node_spec.get("in"), env)
            weight_path = self._infer_param_path(
                node_spec, node_path=node_path, param_name="weight"
            )
            weight = self._state[weight_path]
            out = self._require_name(node_spec.get("out"), field="embedding.out")
            y = F.embedding(x, weight)
            if node_spec.get("scale") is not None:
                scale = float(self._eval_expr(node_spec.get("scale"), env, symbols))
                y = y * y.new_tensor(scale)
            env[out] = y
            return

        if op == "linear":
            x = self._read_tensor_input(node_spec.get("in"), env)
            linear_weight_path: str | None = node_spec.get("tie_weight")
            if not isinstance(linear_weight_path, str):
                linear_weight_path = self._infer_param_path(
                    node_spec, node_path=node_path, param_name="weight"
                )
            weight = self._state[linear_weight_path]
            bias = None
            if node_spec.get("bias", True):
                bias_path = self._infer_param_path(
                    node_spec, node_path=node_path, param_name="bias"
                )
                bias = self._state.get(bias_path)
            weight_layout = str(node_spec.get("weight_layout", "oi"))
            out = self._require_name(node_spec.get("out"), field="linear.out")
            if weight_layout == "oi":
                env[out] = F.linear(x, weight, bias)
            elif weight_layout == "io":
                y = torch.matmul(x, weight)
                env[out] = y + bias if bias is not None else y
            else:
                raise ValueError(f"Unsupported linear weight_layout: {weight_layout}")
            return

        if op == "layernorm":
            x = self._read_tensor_input(node_spec.get("in"), env)
            weight = self._state[
                self._infer_param_path(node_spec, node_path=node_path, param_name="weight")
            ]
            bias = self._state[
                self._infer_param_path(node_spec, node_path=node_path, param_name="bias")
            ]
            eps_value = self._eval_expr(node_spec.get("eps", 1e-5), env, symbols)
            out = self._require_name(node_spec.get("out"), field="layernorm.out")
            env[out] = F.layer_norm(
                x, (x.shape[-1],), weight=weight, bias=bias, eps=float(eps_value)
            )
            return

        if op == "rmsnorm":
            x = self._read_tensor_input(node_spec.get("in"), env)
            weight = self._state[
                self._infer_param_path(node_spec, node_path=node_path, param_name="weight")
            ]
            eps_value = float(self._eval_expr(node_spec.get("eps", 1e-6), env, symbols))
            cast_float = bool(node_spec.get("cast_float", False))
            unit_offset = bool(node_spec.get("unit_offset", False))
            x_norm_src = x.float() if cast_float else x
            w_src = weight.float() if cast_float else weight
            x_norm = x_norm_src * torch.rsqrt(
                torch.mean(x_norm_src * x_norm_src, dim=-1, keepdim=True) + eps_value
            )
            y = x_norm * ((1.0 + w_src) if unit_offset else w_src)
            out = self._require_name(node_spec.get("out"), field="rmsnorm.out")
            env[out] = y.type_as(x) if cast_float else y
            return

        if op == "activation":
            x = self._read_tensor_input(node_spec.get("in"), env)
            kind = node_spec.get("kind", "gelu")
            out = self._require_name(node_spec.get("out"), field="activation.out")
            if kind == "gelu_new" or kind == "gelu_pytorch_tanh":
                env[out] = (
                    0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
                )
            elif kind == "gelu":
                env[out] = F.gelu(x)
            elif kind == "relu":
                env[out] = F.relu(x)
            elif kind == "silu":
                env[out] = F.silu(x)
            else:
                raise ValueError(f"Unsupported activation kind: {kind}")
            return

        if op == "softmax":
            x = self._read_tensor_input(node_spec.get("in"), env)
            out = self._require_name(node_spec.get("out"), field="softmax.out")
            dim = int(self._eval_expr(node_spec.get("dim", -1), env, symbols))
            dtype_name = node_spec.get("dtype")
            if dtype_name is None:
                env[out] = F.softmax(x, dim=dim)
            else:
                if not isinstance(dtype_name, str):
                    raise ValueError("softmax dtype must be a string when provided")
                dtype_map: dict[str, torch.dtype] = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }
                if dtype_name not in dtype_map:
                    raise ValueError(f"Unsupported softmax dtype: {dtype_name}")
                env[out] = F.softmax(x, dim=dim, dtype=dtype_map[dtype_name])
            return

        if op == "topk":
            x = self._read_tensor_input(node_spec.get("in"), env)
            outs = node_spec.get("out")
            if not isinstance(outs, list) or len(outs) != 2:
                raise ValueError("topk expects out=[values,indices]")
            k = int(self._eval_expr(node_spec.get("k"), env, symbols))
            dim = int(self._eval_expr(node_spec.get("dim", -1), env, symbols))
            largest = bool(node_spec.get("largest", True))
            sorted_flag = bool(node_spec.get("sorted", True))
            values, indices = torch.topk(x, k, dim=dim, largest=largest, sorted=sorted_flag)
            env[outs[0]] = values
            env[outs[1]] = indices
            return

        if op == "zeros_like":
            x = self._read_tensor_input(node_spec.get("in"), env)
            out = self._require_name(node_spec.get("out"), field="zeros_like.out")
            env[out] = torch.zeros_like(x)
            return

        if op == "moe_select_tokens":
            ins = node_spec.get("in")
            outs = node_spec.get("out")
            if (
                not isinstance(ins, list)
                or len(ins) != 3
                or not isinstance(outs, list)
                or len(outs) != 4
            ):
                raise ValueError(
                    "moe_select_tokens expects in=[hidden,topk_scores,topk_indices], "
                    "out=[selected_hidden,token_idx,topk_pos,selected_scores]"
                )
            hidden = self._read_tensor_input(ins[0], env)
            topk_scores = self._read_tensor_input(ins[1], env)
            topk_indices = self._read_tensor_input(ins[2], env)
            expert = int(self._eval_expr(node_spec.get("expert"), env, symbols))
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            topk_scores_flat = topk_scores.reshape(-1, topk_scores.shape[-1])
            topk_indices_flat = topk_indices.reshape(-1, topk_indices.shape[-1])
            expert_pos = (topk_indices_flat == expert).nonzero(as_tuple=False)
            token_idx = expert_pos[:, 0]
            topk_pos = expert_pos[:, 1]
            selected_hidden = hidden_flat[token_idx]
            selected_scores = topk_scores_flat[token_idx, topk_pos].to(selected_hidden.dtype)
            env[outs[0]] = selected_hidden
            env[outs[1]] = token_idx
            env[outs[2]] = topk_pos
            env[outs[3]] = selected_scores
            return

        if op == "moe_expert_ffn":
            x = self._read_tensor_input(node_spec.get("in"), env)
            out = self._require_name(node_spec.get("out"), field="moe_expert_ffn.out")
            expert = int(self._eval_expr(node_spec.get("expert"), env, symbols))
            experts_scope = node_spec.get("experts_scope", "experts")
            if not isinstance(experts_scope, str):
                raise ValueError("moe_expert_ffn experts_scope must be string")
            experts_base = self._join(scope, experts_scope)
            experts_parent = experts_base.rsplit(".", 1)[0] if "." in experts_base else ""
            gate_name = str(node_spec.get("gate_proj_name", "gate_proj.weight"))
            up_name = str(node_spec.get("up_proj_name", "up_proj.weight"))
            down_name = str(node_spec.get("down_proj_name", "down_proj.weight"))
            activation = str(node_spec.get("activation", "silu"))
            packed_gate_up = self._join(experts_base, "gate_up_proj")
            packed_down = self._join(experts_base, "down_proj")
            packed_gate_up_parent = (
                self._join(experts_parent, "gate_up_proj") if experts_parent else ""
            )
            packed_down_parent = self._join(experts_parent, "down_proj") if experts_parent else ""

            def resolve_expert_weight(name: str) -> torch.Tensor:
                candidates = [
                    self._join(experts_base, f"{expert}.{name}"),
                    self._join(experts_base, name),
                ]
                if experts_parent:
                    candidates.append(self._join(experts_parent, f"{expert}.{name}"))
                    candidates.append(self._join(experts_parent, name))
                for path in candidates:
                    found = self._state.get(path)
                    if found is not None:
                        return found
                raise KeyError(candidates[0])

            if x.numel() == 0:
                packed_down_key = (
                    packed_down
                    if packed_down in self._state
                    else (packed_down_parent if packed_down_parent in self._state else None)
                )
                if packed_down_key is not None:
                    down_w_shape = self._state[packed_down_key].shape
                    out_dim = int(down_w_shape[-2])
                    env[out] = x.new_zeros((0, out_dim))
                else:
                    down_w = resolve_expert_weight(down_name)
                    env[out] = x.new_zeros((0, int(down_w.shape[-2])))
                return

            packed_gate_up_key = (
                packed_gate_up
                if packed_gate_up in self._state
                else (packed_gate_up_parent if packed_gate_up_parent in self._state else None)
            )
            packed_down_key = (
                packed_down
                if packed_down in self._state
                else (packed_down_parent if packed_down_parent in self._state else None)
            )
            if packed_gate_up_key is not None and packed_down_key is not None:
                packed_gate_up_w = self._state[packed_gate_up_key]
                if packed_gate_up_w.ndim == 3:
                    packed_gate_up_w = packed_gate_up_w[expert]
                gate, up = F.linear(x, packed_gate_up_w, None).chunk(2, dim=-1)
                down_w = self._state[packed_down_key]
                if down_w.ndim == 3:
                    down_w = down_w[expert]
            else:
                gate_w = resolve_expert_weight(gate_name)
                up_w = resolve_expert_weight(up_name)
                down_w = resolve_expert_weight(down_name)
                gate = F.linear(x, gate_w, None)
                up = F.linear(x, up_w, None)

            if activation in {"gelu_new", "gelu_pytorch_tanh"}:
                hidden = (
                    0.5
                    * gate
                    * (
                        1.0
                        + torch.tanh(0.7978845608028654 * (gate + 0.044715 * gate * gate * gate))
                    )
                    * up
                )
            elif activation == "gelu":
                hidden = F.gelu(gate) * up
            elif activation == "relu":
                hidden = F.relu(gate) * up
            elif activation == "silu":
                hidden = F.silu(gate) * up
            else:
                raise ValueError(f"Unsupported moe_expert_ffn activation kind: {activation}")

            env[out] = F.linear(hidden, down_w, None)
            return

        if op == "moe_scatter_add":
            ins = node_spec.get("in")
            if not isinstance(ins, list) or len(ins) != 4:
                raise ValueError("moe_scatter_add expects in=[accum,token_idx,updates,scores]")
            accum = self._read_tensor_input(ins[0], env)
            token_idx = self._read_tensor_input(ins[1], env)
            updates = self._read_tensor_input(ins[2], env)
            scores = self._read_tensor_input(ins[3], env)
            out = self._require_name(node_spec.get("out"), field="moe_scatter_add.out")
            if token_idx.numel() == 0:
                env[out] = accum
                return
            accum_flat = accum.reshape(-1, accum.shape[-1])
            weighted = updates * scores.unsqueeze(-1).to(updates.dtype)
            accum_flat.index_add_(0, token_idx, weighted.to(accum_flat.dtype))
            env[out] = accum
            return

        if op == "add":
            inputs = node_spec.get("in")
            if not isinstance(inputs, list) or len(inputs) != 2:
                raise ValueError("add expects two inputs")
            out = self._require_name(node_spec.get("out"), field="add.out")
            env[out] = env[inputs[0]] + env[inputs[1]]
            return

        if op == "mul":
            inputs = node_spec.get("in")
            if not isinstance(inputs, list) or len(inputs) != 2:
                raise ValueError("mul expects two inputs")
            out = self._require_name(node_spec.get("out"), field="mul.out")
            env[out] = env[inputs[0]] * env[inputs[1]]
            return

        if op == "arange_positions":
            x = self._read_tensor_input(node_spec.get("in"), env)
            seq_len = x.shape[1]
            out = self._require_name(node_spec.get("out"), field="arange_positions.out")
            env[out] = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
            return

        if op == "split_last_dim":
            x = self._read_tensor_input(node_spec.get("in"), env)
            sizes = node_spec.get("sizes")
            outs = node_spec.get("out")
            if not isinstance(sizes, list) or not isinstance(outs, list):
                raise ValueError("split_last_dim requires list sizes and out")
            split_sizes = [int(self._eval_expr(size, env, symbols)) for size in sizes]
            tensors = x.split(split_sizes, dim=-1)
            for name, tensor in zip(outs, tensors, strict=True):
                env[name] = tensor
            return

        if op == "reshape_heads_triplet":
            ins = node_spec.get("in")
            outs = node_spec.get("out")
            heads = int(self._eval_expr(node_spec.get("heads"), env, symbols))
            head_dim = int(self._eval_expr(node_spec.get("head_dim"), env, symbols))
            if (
                not isinstance(ins, list)
                or not isinstance(outs, list)
                or len(ins) != 3
                or len(outs) != 3
            ):
                raise ValueError("reshape_heads_triplet requires 3 inputs and 3 outputs")
            for src_name, dst_name in zip(ins, outs, strict=True):
                src = env[src_name]
                bsz, seq_len, _ = src.shape
                reshaped = src.view(bsz, seq_len, heads, head_dim).transpose(1, 2)
                env[dst_name] = reshaped
            return

        if op == "reshape_heads":
            src = self._read_tensor_input(node_spec.get("in"), env)
            heads = int(self._eval_expr(node_spec.get("heads"), env, symbols))
            head_dim = int(self._eval_expr(node_spec.get("head_dim"), env, symbols))
            bsz, seq_len, _ = src.shape
            out = self._require_name(node_spec.get("out"), field="reshape_heads.out")
            env[out] = src.view(bsz, seq_len, heads, head_dim).transpose(1, 2)
            return

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
            q = env[ins[0]]
            k = env[ins[1]]
            theta = float(self._eval_expr(node_spec.get("theta", 10000.0), env, symbols))
            half = q.shape[-1] // 2
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, half, device=q.device, dtype=q.dtype) / float(half))
            )
            offset = int(self._eval_expr(node_spec.get("offset", 0), env, symbols))
            pos = torch.arange(offset, offset + q.shape[-2], device=q.device, dtype=q.dtype)
            ang = torch.einsum("t,d->td", pos, inv_freq)
            cos = torch.cos(ang)[None, None, :, :]
            sin = torch.sin(ang)[None, None, :, :]
            q1, q2 = q[..., :half], q[..., half : 2 * half]
            k1, k2 = k[..., :half], k[..., half : 2 * half]
            env[outs[0]] = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
            env[outs[1]] = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
            return

        if op == "kv_seq_len":
            ref = node_spec.get("in")
            out_name = self._require_name(node_spec.get("out"), field="kv_seq_len.out")
            if not isinstance(ref, str):
                raise ValueError("kv_seq_len.in must be a string")
            value = env.get(ref)
            if value is None:
                env[out_name] = 0
                return
            if not isinstance(value, tuple) or len(value) < 1 or not torch.is_tensor(value[0]):
                raise ValueError("kv_seq_len expects kv tuple (k, v)")
            env[out_name] = int(value[0].shape[-2])
            return

        if op == "repeat_kv":
            src = self._read_tensor_input(node_spec.get("in"), env)
            repeats = node_spec.get("repeats")
            if repeats is None:
                heads = int(self._eval_expr(node_spec.get("heads"), env, symbols))
                kv_heads = int(self._eval_expr(node_spec.get("kv_heads"), env, symbols))
                n_rep = heads // kv_heads
            else:
                n_rep = int(self._eval_expr(repeats, env, symbols))
            out = self._require_name(node_spec.get("out"), field="repeat_kv.out")
            if n_rep == 1:
                env[out] = src
            else:
                bsz, kvh, seq_len, hd = src.shape
                expanded = src[:, :, None, :, :].expand(bsz, kvh, n_rep, seq_len, hd)
                env[out] = expanded.reshape(bsz, kvh * n_rep, seq_len, hd)
            return

        if op == "merge_heads":
            x = self._read_tensor_input(node_spec.get("in"), env)
            bsz, heads, seq_len, head_dim = x.shape
            merged = x.transpose(1, 2).contiguous().view(bsz, seq_len, heads * head_dim)
            out = self._require_name(node_spec.get("out"), field="merge_heads.out")
            env[out] = merged
            return

        if op == "attention":
            ins = node_spec.get("in")
            if not isinstance(ins, list) or len(ins) != 3:
                raise ValueError("attention expects [q, k, v]")
            q = env[ins[0]]
            k_tensor = env[ins[1]]
            v = env[ins[2]]
            mask = None
            mask_name = node_spec.get("mask")
            if isinstance(mask_name, str):
                mask = env.get(mask_name)
            scale_expr = node_spec.get("scale")
            scale_value = (
                None if scale_expr is None else float(self._eval_expr(scale_expr, env, symbols))
            )
            is_causal_flag = (
                bool(node_spec.get("causal", False)) and q.shape[2] > 1 and mask is None
            )
            attn_out = F.scaled_dot_product_attention(
                q,
                k_tensor,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=is_causal_flag,
                scale=scale_value,
            )
            out_name = self._require_name(node_spec.get("out"), field="attention.out")
            env[out_name] = attn_out
            return

        if op == "causal_mask":
            q = self._read_tensor_input(node_spec.get("in"), env)
            key_ref = node_spec.get("key")
            key_tensor = self._read_tensor_input(key_ref, env) if isinstance(key_ref, str) else q
            out_name = self._require_name(node_spec.get("out"), field="causal_mask.out")
            if node_spec.get("window") is None:
                env[out_name] = None
                return
            q_len = q.shape[-2]
            k_len = key_tensor.shape[-2]
            i_idx = torch.arange(q_len, device=q.device).unsqueeze(1)
            j_idx = torch.arange(k_len, device=q.device).unsqueeze(0)
            keep = j_idx <= i_idx
            win = int(self._eval_expr(node_spec.get("window"), env, symbols))
            if win >= k_len and q_len == k_len:
                env[out_name] = None
                return
            keep = keep & (j_idx >= (i_idx - win + 1))
            mask_value = torch.finfo(q.dtype).min
            mask = torch.where(
                keep,
                torch.zeros((), dtype=q.dtype, device=q.device),
                torch.full((), mask_value, dtype=q.dtype, device=q.device),
            )
            env[out_name] = mask.view(1, 1, q_len, k_len)
            return

        if op == "index":
            ins = node_spec.get("in")
            if not isinstance(ins, list) or len(ins) != 2:
                raise ValueError("index expects [collection, index]")
            collection = env[ins[0]]
            out_name = self._require_name(node_spec.get("out"), field="index.out")
            if collection is None:
                env[out_name] = None
                return
            idx = (
                int(self._eval_expr(ins[1], env, symbols))
                if not isinstance(ins[1], str)
                else int(env[ins[1]])
            )
            env[out_name] = collection[idx]
            return

        if op == "init_list":
            out_name = self._require_name(node_spec.get("out"), field="init_list.out")
            env[out_name] = []
            return

        if op == "append":
            ins = node_spec.get("in")
            if not isinstance(ins, list) or len(ins) != 2:
                raise ValueError("append expects [list_name, item_name]")
            base_list = list(env[ins[0]])
            base_list.append(env[ins[1]])
            out_name = self._require_name(node_spec.get("out"), field="append.out")
            env[out_name] = base_list
            return

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
            past = env.get(ins[0])
            k_new = env[ins[1]]
            v_new = env[ins[2]]
            if past is None:
                k_all = k_new
                v_all = v_new
            else:
                k_all = torch.cat([past[0], k_new], dim=-2)
                v_all = torch.cat([past[1], v_new], dim=-2)
            present = (k_all, v_all)
            env[outs[0]] = k_all
            env[outs[1]] = v_all
            env[outs[2]] = present
            return

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
            k_ctx = env[ins[0]] if ins[0] in env and env[ins[0]] is not None else env[ins[2]]
            v_ctx = env[ins[1]] if ins[1] in env and env[ins[1]] is not None else env[ins[3]]
            env[outs[0]] = k_ctx
            env[outs[1]] = v_ctx
            return

        raise NotImplementedError(f"Unsupported op: {op}")

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
        if "share" in node_spec and isinstance(node_spec["share"], str) and param_name == "weight":
            return node_spec["share"]
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

    def _check_when(self, when_expr: Any, env: dict[str, Any], symbols: dict[str, int]) -> bool:
        if when_expr is None:
            return True
        value = self._eval_expr(when_expr, env, symbols)
        return bool(value)

    def _eval_expr(self, expr: Any, env: dict[str, Any], symbols: dict[str, int]) -> Any:
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

    def _safe_eval_numeric(self, text: str, env: dict[str, Any], symbols: dict[str, int]) -> Any:
        names: dict[str, Any] = {}
        for key, value in symbols.items():
            names[key] = value
        for key, value in env.items():
            if isinstance(value, (int, float, bool)):
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
