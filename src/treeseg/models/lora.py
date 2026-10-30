"""
lora.py - low-rank adapters on Point-SAM's attention projections.

Adapted from Kevin Hu's pointsam_training/lora.py. Adapters start as a no-op. Saving folds them into
plain Linear layers, so a checkpoint has the same keys as stock Point-SAM and any tool can load it.
"""
import contextlib
import math

import torch
from torch import nn


class LinearWithLoRA(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scale

    def merged(self) -> nn.Linear:
        linear = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                           device=self.base.weight.device, dtype=self.base.weight.dtype)
        with torch.no_grad():
            linear.weight.copy_(self.base.weight + (self.lora_b @ self.lora_a) * self.scale)
            if self.base.bias is not None:
                linear.bias.copy_(self.base.bias)
        return linear


def add_lora(module: nn.Module, targets=("q_proj", "v_proj"), rank: int = 8, alpha: float = 16.0) -> int:
    """Wrap every Linear whose name is in targets. Returns how many were wrapped."""
    wrapped = 0
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LinearWithLoRA(child, rank, alpha))
                wrapped += 1
    return wrapped


@contextlib.contextmanager
def merged_lora(module: nn.Module):
    """Inside this block the model holds plain Linear layers, ready to save. Adapters come back after."""
    swaps = [(parent, name, child) for parent in module.modules() for name, child in parent.named_children()
             if isinstance(child, LinearWithLoRA)]
    for parent, name, child in swaps:
        setattr(parent, name, child.merged())
    try:
        yield
    finally:
        for parent, name, child in swaps:
            setattr(parent, name, child)
