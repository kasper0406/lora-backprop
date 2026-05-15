from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


MaskMode = Literal["full", "batch_topk", "ema_topk"]


class _CompressedBackwardLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, active_idx):
        ctx.save_for_backward(x, weight, active_idx)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, active_idx = ctx.saved_tensors
        if active_idx.numel() == weight.shape[0]:
            active_grad = grad_output
            active_weight = weight
        else:
            active_grad = grad_output.index_select(-1, active_idx)
            active_weight = weight.index_select(0, active_idx)
        grad_x = active_grad @ active_weight
        flat_grad = active_grad.reshape(-1, active_grad.shape[-1])
        flat_x = x.reshape(-1, x.shape[-1])
        active_grad_w = flat_grad.T @ flat_x
        grad_w = torch.zeros_like(weight)
        grad_w.index_copy_(0, active_idx, active_grad_w)
        if ctx.needs_input_grad[2]:
            grad_b = torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)
            grad_b.index_copy_(0, active_idx, flat_grad.sum(dim=0))
        else:
            grad_b = None
        return grad_x, grad_w, grad_b, None


class CompressedBackwardLinear(nn.Module):
    """Linear layer whose backward error signal is restricted to selected output channels."""

    def __init__(self, in_features: int, out_features: int, rank: int, mode: MaskMode = "full", ema_decay: float = 0.95, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, out_features)
        self.mode = mode
        self.ema_decay = ema_decay
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.register_buffer("ema_energy", torch.zeros(out_features))
        self.reset_parameters()
        self.last_active_fraction = 1.0
        self.last_dense_backward_flops = 0
        self.last_indexed_backward_flops = 0

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = 1 / (self.in_features ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def _active_idx(self, y: torch.Tensor) -> torch.Tensor:
        if self.mode == "full" or self.rank >= self.out_features:
            self.last_active_fraction = 1.0
            return torch.arange(self.out_features, device=y.device)
        reduce_dims = tuple(range(y.ndim - 1))
        energy = y.detach().pow(2).mean(dim=reduce_dims)
        if self.mode == "ema_topk":
            self.ema_energy.mul_(self.ema_decay).add_(energy, alpha=1 - self.ema_decay)
            score = self.ema_energy
        else:
            score = energy
        idx = score.topk(self.rank).indices.sort().values
        self.last_active_fraction = float(self.rank / self.out_features)
        return idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        active_idx = self._active_idx(y)
        token_count = x.numel() // x.shape[-1]
        # dx and dW each cost roughly 2 * tokens * out * in multiply-add FLOPs.
        self.last_dense_backward_flops = 4 * token_count * self.out_features * self.in_features
        self.last_indexed_backward_flops = 4 * token_count * active_idx.numel() * self.in_features
        return _CompressedBackwardLinearFn.apply(x, self.weight, self.bias, active_idx)


@dataclass
class CompressedMLPConfig:
    vocab_size: int
    d_model: int = 64
    d_hidden: int = 128
    n_layers: int = 8
    rank: int = 16
    mode: MaskMode = "full"


class CompressedBackwardMLP(nn.Module):
    """Simple deep sequence classifier for testing compressed backward channels."""

    def __init__(self, config: CompressedMLPConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(nn.ModuleDict({
                "up": CompressedBackwardLinear(config.d_model, config.d_hidden, config.rank, config.mode),
                "down": CompressedBackwardLinear(config.d_hidden, config.d_model, min(config.rank, config.d_model), config.mode),
            }))
        self.head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, retrieved_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(torch.cat([input_ids, retrieved_ids], dim=1)).mean(dim=1)
        for layer in self.layers:
            h = F.silu(layer["up"](x))
            x = x + layer["down"](h)
        return self.head(x)

    def active_fraction(self) -> float:
        vals = []
        for layer in self.layers:
            vals.extend([layer["up"].last_active_fraction, layer["down"].last_active_fraction])
        return sum(vals) / len(vals)

    def backward_flop_ratio(self) -> float:
        dense = 0
        indexed = 0
        for layer in self.layers:
            dense += layer["up"].last_dense_backward_flops + layer["down"].last_dense_backward_flops
            indexed += layer["up"].last_indexed_backward_flops + layer["down"].last_indexed_backward_flops
        return indexed / dense if dense else 1.0
