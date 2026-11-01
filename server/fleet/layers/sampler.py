# Vendored from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm)
# Copyright (c) 2025 Xingkai Yu — MIT License.
# Modified: ``@torch.compile`` dropped for v1 (compile cost on CPU tests and
# host-portable paths; re-enable behind a config flag when CUDA-graph capture
# returns). The Gumbel-style exponential-noise argmax is unchanged.
import torch
from torch import nn


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
