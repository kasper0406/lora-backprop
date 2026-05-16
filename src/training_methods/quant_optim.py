from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


def _quantize_blockwise(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    flat = x.reshape(-1)
    original_numel = flat.numel()
    n_blocks = math.ceil(original_numel / block_size)
    padded_numel = n_blocks * block_size
    if padded_numel != original_numel:
        flat = torch.nn.functional.pad(flat, (0, padded_numel - original_numel))
    blocks = flat.reshape(n_blocks, block_size)
    scale = blocks.abs().amax(dim=1).clamp_min(1e-12) / 127
    q = torch.round(blocks / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale, original_numel


def _dequantize_blockwise(q: torch.Tensor, scale: torch.Tensor, original_numel: int, shape: torch.Size) -> torch.Tensor:
    flat = (q.to(scale.dtype) * scale[:, None]).reshape(-1)[:original_numel]
    return flat.reshape(shape)


def _quantize_positive_log_blockwise(
    x: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    flat = x.reshape(-1)
    original_numel = flat.numel()
    n_blocks = math.ceil(original_numel / block_size)
    padded_numel = n_blocks * block_size
    if padded_numel != original_numel:
        flat = torch.nn.functional.pad(flat, (0, padded_numel - original_numel))
    blocks = flat.reshape(n_blocks, block_size).clamp_min(0)
    log_blocks = torch.log1p(blocks)
    lo = log_blocks.amin(dim=1)
    hi = log_blocks.amax(dim=1)
    scale = (hi - lo).clamp_min(1e-12) / 255
    q = torch.round((log_blocks - lo[:, None]) / scale[:, None]).clamp(0, 255).to(torch.uint8)
    return q, lo, scale, original_numel


def _dequantize_positive_log_blockwise(
    q: torch.Tensor,
    lo: torch.Tensor,
    scale: torch.Tensor,
    original_numel: int,
    shape: torch.Size,
) -> torch.Tensor:
    log_flat = (q.to(scale.dtype) * scale[:, None] + lo[:, None]).reshape(-1)[:original_numel]
    return torch.expm1(log_flat).reshape(shape).clamp_min(0)


class Int8AdamW(Optimizer):
    """AdamW with blockwise symmetric int8 storage for first and second moments.

    This is a clarity-first prototype: states are stored compressed between steps,
    then dequantized to compute each update and requantized afterward.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        block_size: int = 256,
    ):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, block_size=block_size)
        super().__init__(params, defaults)
        self.last_diagnostics: dict[str, float] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            block_size = group["block_size"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                state["step"] = state.get("step", 0) + 1
                if "exp_avg_q" not in state:
                    zeros = torch.zeros_like(p)
                    state["exp_avg_q"], state["exp_avg_scale"], state["numel"] = _quantize_blockwise(zeros, block_size)
                    (
                        state["exp_avg_sq_q"],
                        state["exp_avg_sq_lo"],
                        state["exp_avg_sq_scale"],
                        _,
                    ) = _quantize_positive_log_blockwise(zeros, block_size)

                m = _dequantize_blockwise(
                    state["exp_avg_q"], state["exp_avg_scale"], state["numel"], p.shape
                )
                v = _dequantize_positive_log_blockwise(
                    state["exp_avg_sq_q"],
                    state["exp_avg_sq_lo"],
                    state["exp_avg_sq_scale"],
                    state["numel"],
                    p.shape,
                )
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                step_size = lr / bc1
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                if wd:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m, denom, value=-step_size)

                m_q, m_scale, _ = _quantize_blockwise(m, block_size)
                v_q, v_lo, v_scale, _ = _quantize_positive_log_blockwise(v, block_size)
                m_hat = _dequantize_blockwise(m_q, m_scale, state["numel"], p.shape)
                v_hat = _dequantize_positive_log_blockwise(v_q, v_lo, v_scale, state["numel"], p.shape)
                self.last_diagnostics = {
                    "m_rel_err": float((m_hat - m).norm().div(m.norm().clamp_min(1e-12)).item()),
                    "v_rel_err": float((v_hat - v).norm().div(v.norm().clamp_min(1e-12)).item()),
                    "v_zero_frac": float(v_hat.eq(0).float().mean().item()),
                }
                state["exp_avg_q"], state["exp_avg_scale"] = m_q, m_scale
                state["exp_avg_sq_q"], state["exp_avg_sq_lo"], state["exp_avg_sq_scale"] = v_q, v_lo, v_scale
        return loss

    def state_numel(self) -> int:
        total = 0
        for state in self.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel()
        return total

    def state_nbytes(self) -> int:
        total = 0
        for state in self.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total


class BF16AdamW(Optimizer):
    """AdamW that stores moment tensors in bfloat16 between updates."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                state["step"] = state.get("step", 0) + 1
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.bfloat16)
                m = state["exp_avg"].to(dtype=p.dtype)
                v = state["exp_avg_sq"].to(dtype=p.dtype)
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** state["step"]
                bc2 = 1 - beta2 ** state["step"]
                step_size = lr / bc1
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                if wd:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m, denom, value=-step_size)
                state["exp_avg"].copy_(m.to(torch.bfloat16))
                state["exp_avg_sq"].copy_(v.to(torch.bfloat16))
        return loss

    def state_nbytes(self) -> int:
        total = 0
        for state in self.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total
