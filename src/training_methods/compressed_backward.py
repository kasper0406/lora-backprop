from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


MaskMode = Literal[
    "full",
    "batch_topk",
    "ema_topk",
    "random_lowrank",
    "activation_pca_lowrank",
    "activation_sketch_lowrank",
]


class _CompressedBackwardLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, active_idx, module):
        ctx.save_for_backward(x, weight, active_idx)
        ctx.module = module
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
        if ctx.module.track_diagnostics:
            ctx.module.row_grad_mass.index_add_(0, active_idx, active_grad_w.abs().sum(dim=1))
        return grad_x, grad_w, grad_b, None, None


class _LowRankBackwardLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, basis):
        ctx.save_for_backward(x, weight, basis)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, basis = ctx.saved_tensors
        # B spans a dense rank-r output-channel subspace:
        #   z  = dy B
        #   dx = z (B^T W)
        #   dW = B (z^T x)
        lowrank_grad = grad_output @ basis
        projected_weight = basis.T @ weight
        grad_x = lowrank_grad @ projected_weight
        flat_grad = lowrank_grad.reshape(-1, lowrank_grad.shape[-1])
        flat_x = x.reshape(-1, x.shape[-1])
        grad_w = basis @ (flat_grad.T @ flat_x)
        if ctx.needs_input_grad[2]:
            grad_b = basis @ flat_grad.sum(dim=0)
        else:
            grad_b = None
        return grad_x, grad_w, grad_b, None


class CompressedBackwardLinear(nn.Module):
    """Linear layer whose backward error signal is restricted to selected output channels."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        mode: MaskMode = "full",
        ema_decay: float = 0.95,
        basis_refresh_every: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, out_features)
        self.mode = mode
        self.ema_decay = ema_decay
        self.basis_refresh_every = basis_refresh_every
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.register_buffer("ema_energy", torch.zeros(out_features))
        self.register_buffer("random_basis", self._make_random_basis(out_features, self.rank))
        self.register_buffer("cached_basis", torch.empty(0))
        self.reset_parameters()
        self.last_active_fraction = 1.0
        self.last_dense_backward_flops = 0
        self.last_indexed_backward_flops = 0
        self.register_buffer("selection_counts", torch.zeros(out_features))
        self.register_buffer("row_grad_mass", torch.zeros(out_features))
        # Hot-path bookkeeping kept as Python ints to avoid GPU↔CPU sync per forward.
        self._selection_steps = 0
        self._basis_age = 0
        self._turnover_acc = 0.0
        self._turnover_n = 0
        self._last_selection_idx: torch.Tensor | None = None
        self.last_turnover = 0.0
        # Stochastic top-k temperature (Gumbel-top-k). When None the selection is
        # the deterministic argmax over ema_energy (original behaviour). When set,
        # the score is eta * log(ema_energy) + Gumbel(0,1) — small eta = diffuse
        # exploration, large eta -> deterministic. The lightning analogy: this is
        # the dielectric-breakdown concentration exponent η. Driven externally by
        # the training loop so it can be annealed across steps.
        self.eta: float | None = None
        # Per-step diagnostics (selection_counts, row_grad_mass) cost a real GPU
        # index_add_ per layer per forward/backward. Off by default so the
        # speed-critical path stays clean; flip to True via the convenience
        # helper below before runs where you want stats.
        self.track_diagnostics = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            bound = 1 / (self.in_features ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def _make_random_basis(out_features: int, rank: int) -> torch.Tensor:
        q, _ = torch.linalg.qr(torch.randn(out_features, rank), mode="reduced")
        return q

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
        # topk indices need no .sort(); index_select/index_copy_ don't require ordering.
        idx = score.topk(self.rank).indices
        self.last_active_fraction = float(self.rank / self.out_features)
        # Always tick the step counter — a cheap Python increment with no GPU work.
        # The fast ema_topk forward path keys off this counter to know the EMA
        # buffer is warm and y is unnecessary for scoring.
        self._selection_steps += 1
        if self.track_diagnostics:
            # Update selection histogram on-device without materializing a bool mask.
            self.selection_counts.index_add_(
                0, idx, torch.ones_like(idx, dtype=self.selection_counts.dtype)
            )
            self._last_selection_idx = idx
        return idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_count = x.numel() // x.shape[-1]
        self.last_dense_backward_flops = 4 * token_count * self.out_features * self.in_features

        if self.mode == "full" or self.rank >= self.out_features:
            self.last_active_fraction = 1.0
            self.last_indexed_backward_flops = self.last_dense_backward_flops
            return F.linear(x, self.weight, self.bias)

        is_lowrank = self.mode in {
            "random_lowrank", "activation_pca_lowrank", "activation_sketch_lowrank"
        }

        if is_lowrank:
            self.last_active_fraction = float(self.rank / self.out_features)
            self.last_indexed_backward_flops = (
                2 * token_count * self.out_features * self.rank
                + 4 * token_count * self.rank * self.in_features
                + 4 * self.out_features * self.rank * self.in_features
            )
            basis = self._cached_or_refresh_basis(x)
            return _LowRankBackwardLinearFn.apply(x, self.weight, self.bias, basis)

        # Top-k path. ema_topk uses the cached EMA energy from prior steps as the
        # score, so this step's y is not needed for selection — we update the EMA
        # after the autograd Function has computed y. batch_topk needs this step's
        # y for scoring, so we pay the double-matmul there (rare in practice).
        if self.mode == "ema_topk" and self._selection_steps > 0:
            if self.eta is not None:
                # Gumbel-top-k sampling over the EMA energy. Equivalent to drawing
                # k channels without replacement from p_i ∝ ema_energy_i^eta.
                log_energy = (self.ema_energy + 1e-12).log()
                u = torch.rand_like(log_energy).clamp_min(1e-12)
                gumbel = -(-u.log()).clamp_min(1e-12).log()
                scores = self.eta * log_energy + gumbel
                active_idx = scores.topk(self.rank).indices
            else:
                active_idx = self.ema_energy.topk(self.rank).indices
            self.last_active_fraction = float(self.rank / self.out_features)
            self.last_indexed_backward_flops = 4 * token_count * active_idx.numel() * self.in_features
            out = _CompressedBackwardLinearFn.apply(x, self.weight, self.bias, active_idx, self)
            # Refresh EMA from y (=out), no extra matmul.
            with torch.no_grad():
                reduce_dims = tuple(range(out.ndim - 1))
                energy = out.detach().pow(2).mean(dim=reduce_dims)
                self.ema_energy.mul_(self.ema_decay).add_(energy, alpha=1 - self.ema_decay)
                self._selection_steps += 1
                if self.track_diagnostics:
                    self.selection_counts.index_add_(
                        0, active_idx, torch.ones_like(active_idx, dtype=self.selection_counts.dtype)
                    )
                    self._last_selection_idx = active_idx
            return out

        # batch_topk and the first ema_topk step still need y to compute the score.
        y = F.linear(x, self.weight, self.bias)
        active_idx = self._active_idx(y)
        self.last_indexed_backward_flops = 4 * token_count * active_idx.numel() * self.in_features
        return _CompressedBackwardLinearFn.apply(x, self.weight, self.bias, active_idx, self)

    def _cached_or_refresh_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Return the cached basis if still warm; else compute y and refresh."""
        if self.mode == "random_lowrank":
            return self.random_basis.to(device=x.device, dtype=x.dtype)
        if (
            self.cached_basis.numel()
            and self._basis_age < self.basis_refresh_every
            and self.cached_basis.device == x.device
            and self.cached_basis.dtype == x.dtype
        ):
            self._basis_age += 1
            return self.cached_basis
        # Need a fresh basis — pay the extra F.linear once per refresh interval.
        y = F.linear(x, self.weight, self.bias)
        return self._lowrank_basis(y)

    def _lowrank_basis(self, y: torch.Tensor) -> torch.Tensor:
        """Compute a fresh basis from y and cache it. Caller is responsible for
        checking the cache before calling — see _cached_or_refresh_basis."""
        if self.mode == "random_lowrank":
            return self.random_basis.to(device=y.device, dtype=y.dtype)
        flat_y = y.detach().reshape(-1, y.shape[-1])
        centered = flat_y - flat_y.mean(dim=0, keepdim=True)
        if self.mode == "activation_sketch_lowrank":
            sketch_rank = min(self.out_features, self.rank + 4)
            omega = torch.randn(self.out_features, sketch_rank, device=y.device, dtype=y.dtype)
            sample = centered @ omega
            q, _ = torch.linalg.qr(sample, mode="reduced")
            basis, _ = torch.linalg.qr(centered.T @ q, mode="reduced")
            result = basis[:, : self.rank]
        else:
            cov = centered.T @ centered / max(1, centered.shape[0] - 1)
            if cov.device.type == "mps":
                _, evecs_cpu = torch.linalg.eigh(cov.cpu())
                evecs = evecs_cpu.to(cov.device)
            else:
                _, evecs = torch.linalg.eigh(cov)
            result = evecs[:, -self.rank :]
        self.cached_basis = result
        self._basis_age = 0
        return result

    def theoretical_optimizer_state_ratio(self) -> float:
        # Dense projected gradients still require dense optimizer state with AdamW.
        return 1.0

    def selection_stats(self) -> tuple[float, float, float, float]:
        if self._selection_steps == 0:
            return 0.0, 0.0, 0.0, 0.0
        coverage = self.selection_counts.gt(0).float().mean().item()
        probs = self.selection_counts / self.selection_counts.sum().clamp_min(1)
        concentration = probs.square().sum().item()
        grad_probs = self.row_grad_mass / self.row_grad_mass.sum().clamp_min(1)
        grad_concentration = grad_probs.square().sum().item()
        return coverage, concentration, self.last_turnover, grad_concentration


@dataclass
class CompressedMLPConfig:
    vocab_size: int
    d_model: int = 64
    d_hidden: int = 128
    n_layers: int = 8
    rank: int = 16
    mode: MaskMode = "full"


@dataclass(frozen=True)
class BackwardCostEstimate:
    dense_backward_flops: int
    method_backward_flops: int
    dense_adam_state_elements: int
    method_adam_state_elements: int

    @property
    def flop_ratio(self) -> float:
        return self.method_backward_flops / self.dense_backward_flops

    @property
    def adam_state_ratio(self) -> float:
        return self.method_adam_state_elements / self.dense_adam_state_elements


def estimate_linear_backward_cost(
    *,
    token_count: int,
    in_features: int,
    out_features: int,
    rank: int,
    mode: MaskMode,
) -> BackwardCostEstimate:
    rank = min(rank, out_features)
    dense_flops = 4 * token_count * out_features * in_features
    dense_adam_state = 2 * out_features * in_features
    if mode in {"batch_topk", "ema_topk"} and rank < out_features:
        method_flops = 4 * token_count * rank * in_features
        method_adam_state = 2 * rank * in_features
    elif mode in {"random_lowrank", "activation_pca_lowrank", "activation_sketch_lowrank"} and rank < out_features:
        method_flops = (
            2 * token_count * out_features * rank
            + 4 * token_count * rank * in_features
            + 4 * out_features * rank * in_features
        )
        # This experiment lifts the update back to dense dW, so ordinary AdamW remains dense.
        method_adam_state = dense_adam_state
    else:
        method_flops = dense_flops
        method_adam_state = dense_adam_state
    return BackwardCostEstimate(dense_flops, method_flops, dense_adam_state, method_adam_state)


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
