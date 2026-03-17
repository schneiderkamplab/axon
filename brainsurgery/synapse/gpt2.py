from __future__ import annotations

import importlib.resources
import textwrap
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SynapseGPT2Config:
    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    n_inner: int
    layer_norm_epsilon: float
    activation_function: str

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        return self.n_embd // self.n_head


class Conv1D(nn.Module):
    """HF-style Conv1D with parameter layout [in_dim, out_dim]."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) + self.bias


class GPT2SelfAttention(nn.Module):
    def __init__(self, config: SynapseGPT2Config) -> None:
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.c_attn = Conv1D(config.n_embd, 3 * config.n_embd)
        self.c_proj = Conv1D(config.n_embd, config.n_embd)

        mask = torch.tril(torch.ones((config.n_positions, config.n_positions), dtype=torch.bool))
        self.register_buffer(
            "bias", mask.view(1, 1, config.n_positions, config.n_positions), persistent=True
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.n_head, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, self.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        qkv = self.c_attn(x)
        q_lin, k_lin, v_lin = qkv.split(self.n_embd, dim=-1)

        q = self._split_heads(q_lin)
        k_new = self._split_heads(k_lin)
        v_new = self._split_heads(v_lin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k_new], dim=-2)
            v = torch.cat([past_v, v_new], dim=-2)
        else:
            k = k_new
            v = v_new

        present_kv = (k, v)
        is_causal = past_kv is None
        ctx = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
        )

        merged = self._merge_heads(ctx)
        return self.c_proj(merged), present_kv


class GPT2MLP(nn.Module):
    def __init__(self, config: SynapseGPT2Config) -> None:
        super().__init__()
        self.c_fc = Conv1D(config.n_embd, config.n_inner)
        self.c_proj = Conv1D(config.n_inner, config.n_embd)
        if config.activation_function == "gelu_new":
            self.act = (
                lambda x: 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
            )
        elif config.activation_function == "gelu":
            self.act = F.gelu
        else:
            raise ValueError(f"Unsupported activation function: {config.activation_function}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class GPT2Block(nn.Module):
    def __init__(self, config: SynapseGPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class SynapseGPT2LMHeadModel(nn.Module):
    """Decoder-only GPT-2 forward model generated from concise Synapse specs."""

    def __init__(self, config: SynapseGPT2Config) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        spec: dict[str, Any],
        *,
        strict: bool = True,
        map_location: str | torch.device | None = None,
    ) -> "SynapseGPT2LMHeadModel":
        config = _config_from_synapse_spec(spec)
        model = cls(config)
        load_sd = {
            k: (v.to(map_location) if map_location is not None else v)
            for k, v in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(load_sd, strict=strict)
        if strict and (missing or unexpected):
            raise RuntimeError(f"State dict mismatch. missing={missing} unexpected={unexpected}")
        return model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be rank-2 [batch, seq]")

        _, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)

        x = self.wte(input_ids) + self.wpe(pos)
        for block in self.h:
            x, _ = block(x)
        x = self.ln_f(x)
        logits = F.linear(x, self.wte.weight)
        return logits

    def forward_with_past(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be rank-2 [batch, seq]")
        if past_key_values is not None and len(past_key_values) != len(self.h):
            raise ValueError("past_key_values must have one (k,v) pair per transformer layer")

        _, seq_len = input_ids.shape
        past_len = 0
        if past_key_values is not None:
            past_len = past_key_values[0][0].size(-2)
        pos = torch.arange(
            past_len,
            past_len + seq_len,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)

        x = self.wte(input_ids) + self.wpe(pos)
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, block in enumerate(self.h):
            layer_past = None if past_key_values is None else past_key_values[idx]
            x, present = block(x, past_kv=layer_past)
            present_key_values.append(present)
        x = self.ln_f(x)
        logits = F.linear(x, self.wte.weight)
        return logits, present_key_values

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
                logits, past_key_values = self.forward_with_past(generated)
                while generated.size(1) < max_len and not torch.all(finished):
                    next_token = torch.argmax(logits[:, -1, :], dim=-1)
                    next_token = torch.where(
                        finished,
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )
                    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                    finished = torch.logical_or(finished, next_token == eos_token_id)
                    if generated.size(1) >= max_len or torch.all(finished):
                        break
                    logits, past_key_values = self.forward_with_past(
                        next_token.unsqueeze(1),
                        past_key_values=past_key_values,
                    )
        finally:
            if was_training:
                self.train()
        return generated


def _read_symbol(model: dict[str, Any], name: str) -> int:
    value = model.get("symbols", {}).get(name)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer symbol '{name}', got {value!r}")
    return value


def _config_from_synapse_spec(spec: dict[str, Any]) -> SynapseGPT2Config:
    if spec.get("synapse") != 1:
        raise ValueError("Only synapse: 1 specs are supported")

    model = spec.get("model")
    if not isinstance(model, dict):
        raise ValueError("spec.model must be a mapping")

    symbols = model.get("symbols", {})
    if not isinstance(symbols, dict):
        raise ValueError("model.symbols must be a mapping")

    params = model.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("model.params must be a mapping")

    n_embd = _read_symbol(model, "D")
    n_head = _read_symbol(model, "H")
    n_layer = _read_symbol(model, "L")
    vocab_size = _read_symbol(model, "V")
    n_positions = _read_symbol(model, "C")
    n_inner_value = symbols.get("M", 4 * n_embd)
    if not isinstance(n_inner_value, int):
        raise ValueError("Expected integer symbol 'M' when provided")

    activation = params.get("activation", "gelu")
    if not isinstance(activation, str):
        raise ValueError("params.activation must be a string")

    ln_eps = params.get("layer_norm_epsilon", 1e-5)
    if not isinstance(ln_eps, (int, float)):
        raise ValueError("params.layer_norm_epsilon must be numeric")

    return SynapseGPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=n_inner_value,
        layer_norm_epsilon=float(ln_eps),
        activation_function=activation,
    )


def build_gpt2_from_synapse_spec(
    spec: dict[str, Any],
    state_dict: dict[str, torch.Tensor] | None = None,
    *,
    strict: bool = True,
) -> SynapseGPT2LMHeadModel:
    """Build a decoder-only GPT-2 LM head model from a Synapse spec.

    If `state_dict` is provided, tensor names are expected in GPT-2 root format:
    `wte.weight`, `wpe.weight`, `h.<i>.attn.c_attn.weight`, ..., `ln_f.bias`.
    """

    config = _config_from_synapse_spec(spec)
    model = SynapseGPT2LMHeadModel(config)
    if state_dict is None:
        return model

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"State dict mismatch. missing={missing} unexpected={unexpected}")
    return model


def emit_gpt2_model_code_from_synapse_spec(
    spec: dict[str, Any],
    *,
    class_name: str = "GeneratedGPT2LMHeadModel",
    op_map: dict[str, Any] | None = None,
) -> str:
    """Generate standalone PyTorch source code for a GPT-2 decoder LM class."""

    if not class_name.isidentifier():
        raise ValueError(f"Invalid class name: {class_name!r}")

    resolved_op_map = _load_synapse_torch_op_map() if op_map is None else op_map
    _validate_codegen_op_map(resolved_op_map)
    config = _config_from_synapse_spec(spec)
    activation_repr = repr(config.activation_function)
    return textwrap.dedent(
        f"""\
        from __future__ import annotations

        import torch
        from torch import nn
        from torch.nn import functional as F


        class Conv1D(nn.Module):
            def __init__(self, in_dim: int, out_dim: int) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
                self.bias = nn.Parameter(torch.zeros(out_dim))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(x, self.weight) + self.bias


        class GPT2SelfAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.n_embd = {config.n_embd}
                self.n_head = {config.n_head}
                self.head_dim = {config.head_dim}
                self.c_attn = Conv1D(self.n_embd, 3 * self.n_embd)
                self.c_proj = Conv1D(self.n_embd, self.n_embd)

                mask = torch.tril(torch.ones(({config.n_positions}, {config.n_positions}), dtype=torch.bool))
                self.register_buffer("bias", mask.view(1, 1, {config.n_positions}, {config.n_positions}), persistent=True)

            def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
                bsz, seq_len, _ = x.shape
                x = x.view(bsz, seq_len, self.n_head, self.head_dim)
                return x.transpose(1, 2)

            def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
                bsz, _, seq_len, _ = x.shape
                x = x.transpose(1, 2).contiguous()
                return x.view(bsz, seq_len, self.n_embd)

            def forward(
                self,
                x: torch.Tensor,
                *,
                past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
            ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                qkv = self.c_attn(x)
                q_lin, k_lin, v_lin = qkv.split(self.n_embd, dim=-1)
                q = self._split_heads(q_lin)
                k_new = self._split_heads(k_lin)
                v_new = self._split_heads(v_lin)

                if past_kv is not None:
                    past_k, past_v = past_kv
                    k = torch.cat([past_k, k_new], dim=-2)
                    v = torch.cat([past_v, v_new], dim=-2)
                else:
                    k = k_new
                    v = v_new

                present_kv = (k, v)
                is_causal = past_kv is None
                ctx = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
                )

                return self.c_proj(self._merge_heads(ctx)), present_kv


        class GPT2MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c_fc = Conv1D({config.n_embd}, {config.n_inner})
                self.c_proj = Conv1D({config.n_inner}, {config.n_embd})
                self.activation = {activation_repr}

            def _apply_act(self, x: torch.Tensor) -> torch.Tensor:
                if self.activation == "gelu_new":
                    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
                if self.activation == "gelu":
                    return F.gelu(x)
                raise ValueError(f"Unsupported activation function: {{self.activation}}")

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.c_proj(self._apply_act(self.c_fc(x)))


        class GPT2Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ln_1 = nn.LayerNorm({config.n_embd}, eps={config.layer_norm_epsilon!r})
                self.attn = GPT2SelfAttention()
                self.ln_2 = nn.LayerNorm({config.n_embd}, eps={config.layer_norm_epsilon!r})
                self.mlp = GPT2MLP()

            def forward(
                self,
                x: torch.Tensor,
                *,
                past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
            ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv)
                x = x + attn_out
                x = x + self.mlp(self.ln_2(x))
                return x, present_kv


        class {class_name}(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.wte = nn.Embedding({config.vocab_size}, {config.n_embd})
                self.wpe = nn.Embedding({config.n_positions}, {config.n_embd})
                self.h = nn.ModuleList([GPT2Block() for _ in range({config.n_layer})])
                self.ln_f = nn.LayerNorm({config.n_embd}, eps={config.layer_norm_epsilon!r})

            @classmethod
            def from_state_dict(cls, state_dict: dict[str, torch.Tensor], *, strict: bool = True) -> "{class_name}":
                model = cls()
                missing, unexpected = model.load_state_dict(state_dict, strict=strict)
                if strict and (missing or unexpected):
                    raise RuntimeError(f"State dict mismatch. missing={{missing}} unexpected={{unexpected}}")
                return model

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                if input_ids.ndim != 2:
                    raise ValueError("input_ids must be rank-2 [batch, seq]")
                _, seq_len = input_ids.shape
                pos = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
                x = self.wte(input_ids) + self.wpe(pos)
                for block in self.h:
                    x, _ = block(x)
                x = self.ln_f(x)
                return F.linear(x, self.wte.weight)

            def forward_with_past(
                self,
                input_ids: torch.Tensor,
                *,
                past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
            ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
                if input_ids.ndim != 2:
                    raise ValueError("input_ids must be rank-2 [batch, seq]")
                if past_key_values is not None and len(past_key_values) != len(self.h):
                    raise ValueError("past_key_values must have one (k,v) pair per transformer layer")

                _, seq_len = input_ids.shape
                past_len = 0
                if past_key_values is not None:
                    past_len = past_key_values[0][0].size(-2)
                pos = torch.arange(
                    past_len,
                    past_len + seq_len,
                    device=input_ids.device,
                    dtype=torch.long,
                ).unsqueeze(0)

                x = self.wte(input_ids) + self.wpe(pos)
                present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
                for idx, block in enumerate(self.h):
                    layer_past = None if past_key_values is None else past_key_values[idx]
                    x, present = block(x, past_kv=layer_past)
                    present_key_values.append(present)
                x = self.ln_f(x)
                logits = F.linear(x, self.wte.weight)
                return logits, present_key_values

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
                        logits, past_key_values = self.forward_with_past(generated)
                        while generated.size(1) < max_len and not torch.all(finished):
                            next_token = torch.argmax(logits[:, -1, :], dim=-1)
                            next_token = torch.where(
                                finished,
                                torch.full_like(next_token, eos_token_id),
                                next_token,
                            )
                            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                            finished = torch.logical_or(finished, next_token == eos_token_id)
                            if generated.size(1) >= max_len or torch.all(finished):
                                break
                            logits, past_key_values = self.forward_with_past(
                                next_token.unsqueeze(1),
                                past_key_values=past_key_values,
                            )
                finally:
                    if was_training:
                        self.train()
                return generated
        """
    )


def _load_synapse_torch_op_map() -> dict[str, Any]:
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


def _validate_codegen_op_map(op_map: dict[str, Any]) -> None:
    ops = op_map.get("ops")
    if not isinstance(ops, dict):
        raise ValueError("op map must contain mapping key 'ops'")

    required_targets = {
        "embedding": "torch.nn.Embedding",
        "linear": "torch.nn.Linear",
        "layernorm": "torch.nn.LayerNorm",
        "attention": "torch.nn.functional.scaled_dot_product_attention",
    }
    for op_name, expected_target in required_targets.items():
        op_spec = ops.get(op_name)
        if not isinstance(op_spec, dict):
            raise ValueError(f"op map is missing required op '{op_name}'")
        target = op_spec.get("target")
        if target != expected_target:
            raise ValueError(
                f"op '{op_name}' must map to '{expected_target}' for this generator, got {target!r}"
            )
