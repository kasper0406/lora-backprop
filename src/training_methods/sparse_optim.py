from __future__ import annotations

from collections import defaultdict

import torch
from torch.optim import Optimizer

from .compressed_backward import CompressedBackwardLinear


class RowSparseAdamW(Optimizer):
    """AdamW that stores row moments sparsely for CompressedBackwardLinear weights/biases."""

    def __init__(self, model: torch.nn.Module, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        params = list(model.parameters())
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.sparse_weight_params: dict[torch.nn.Parameter, CompressedBackwardLinear] = {}
        self.sparse_bias_params: dict[torch.nn.Parameter, CompressedBackwardLinear] = {}
        for module in model.modules():
            if isinstance(module, CompressedBackwardLinear):
                self.sparse_weight_params[module.weight] = module
                if module.bias is not None:
                    self.sparse_bias_params[module.bias] = module

    def _active_idx(self, module: CompressedBackwardLinear, param: torch.Tensor) -> torch.Tensor:
        # Nonzero gradient rows are exactly the selected backward channels.
        grad = param.grad
        if grad is None:
            return torch.empty(0, dtype=torch.long, device=param.device)
        if grad.ndim == 2:
            return grad.abs().sum(dim=1).ne(0).nonzero(as_tuple=False).flatten()
        return grad.ne(0).nonzero(as_tuple=False).flatten()

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = group['lr'], group['betas'], group['eps'], group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if p in self.sparse_weight_params or p in self.sparse_bias_params:
                    module = self.sparse_weight_params.get(p) or self.sparse_bias_params[p]
                    idx = self._active_idx(module, p)
                    if idx.numel() == 0:
                        continue
                    state = self.state[p]
                    state['step'] = state.get('step', 0) + 1
                    exp_avg = state.setdefault('exp_avg', {})
                    exp_avg_sq = state.setdefault('exp_avg_sq', {})
                    for row in idx.tolist():
                        g = grad[row]
                        m = exp_avg.get(row)
                        v = exp_avg_sq.get(row)
                        if m is None:
                            m = exp_avg[row] = torch.zeros_like(g)
                            v = exp_avg_sq[row] = torch.zeros_like(g)
                        m.mul_(beta1).add_(g, alpha=1 - beta1)
                        v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                        bc1 = 1 - beta1 ** state['step']
                        bc2 = 1 - beta2 ** state['step']
                        step_size = lr / bc1
                        denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
                        if wd:
                            p[row].mul_(1 - lr * wd)
                        p[row].addcdiv_(m, denom, value=-step_size)
                else:
                    state = self.state[p]
                    state['step'] = state.get('step', 0) + 1
                    m = state.setdefault('exp_avg', torch.zeros_like(p))
                    v = state.setdefault('exp_avg_sq', torch.zeros_like(p))
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    step_size = lr / bc1
                    denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(m, denom, value=-step_size)
        return loss

    def state_numel(self) -> int:
        total = 0
        for state in self.state.values():
            for value in state.values():
                if isinstance(value, dict):
                    total += sum(t.numel() for t in value.values())
                elif torch.is_tensor(value):
                    total += value.numel()
        return total
