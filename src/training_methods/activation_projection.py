from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


ProjectionKind = Literal["none", "random", "activation_pca", "topk_features"]


@dataclass
class ProjectionStats:
    projected_layers: int = 0
    mean_rank: float = 0.0
    mean_activation_energy: float = 0.0
    mean_grad_cosine: float = 0.0


class ActivationGradientProjector:
    """Project Linear-layer weight gradients using bases chosen from forward activations."""

    def __init__(
        self,
        model: nn.Module,
        rank: int,
        kind: ProjectionKind = "activation_pca",
        min_input_dim: int = 2,
    ):
        self.rank = rank
        self.kind = kind
        self.min_input_dim = min_input_dim
        self.inputs: dict[nn.Linear, torch.Tensor] = {}
        self.random_bases: dict[nn.Linear, torch.Tensor] = {}
        self.handles = []
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.in_features >= min_input_dim:
                self.handles.append(module.register_forward_hook(self._capture_input))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _capture_input(self, module: nn.Linear, args, output) -> None:
        if not args:
            return
        x = args[0].detach()
        self.inputs[module] = x.reshape(-1, x.shape[-1])

    def _basis(self, module: nn.Linear, x: torch.Tensor, rank: int) -> tuple[torch.Tensor, float]:
        if self.kind == "random":
            basis = self.random_bases.get(module)
            if basis is None or basis.device != x.device or basis.dtype != x.dtype or basis.shape != (x.shape[-1], rank):
                q, _ = torch.linalg.qr(torch.randn(x.shape[-1], rank, device=x.device, dtype=x.dtype), mode="reduced")
                basis = q
                self.random_bases[module] = basis
            return basis, float(rank / x.shape[-1])

        if self.kind == "topk_features":
            energy = x.pow(2).mean(dim=0)
            idx = energy.topk(rank).indices
            basis = torch.zeros(x.shape[-1], rank, device=x.device, dtype=x.dtype)
            basis[idx, torch.arange(rank, device=x.device)] = 1
            captured = energy[idx].sum() / energy.sum().clamp_min(1e-12)
            return basis, float(captured.item())

        # activation_pca
        centered = x - x.mean(dim=0, keepdim=True)
        cov = centered.T @ centered / max(1, centered.shape[0] - 1)
        eig_device = cov.device
        if eig_device.type == "mps":
            evals_cpu, evecs_cpu = torch.linalg.eigh(cov.cpu())
            evals = evals_cpu.to(eig_device)
            evecs = evecs_cpu.to(eig_device)
        else:
            evals, evecs = torch.linalg.eigh(cov)
        basis = evecs[:, -rank:]
        captured = evals[-rank:].clamp_min(0).sum() / evals.clamp_min(0).sum().clamp_min(1e-12)
        return basis, float(captured.item())

    @torch.no_grad()
    def project(self) -> ProjectionStats:
        if self.kind == "none":
            return ProjectionStats()
        projected = 0
        ranks: list[int] = []
        energies: list[float] = []
        cosines: list[float] = []
        for module, x in self.inputs.items():
            grad = module.weight.grad
            if grad is None:
                continue
            rank = min(self.rank, x.shape[-1])
            if rank >= x.shape[-1]:
                continue
            original = grad.detach().clone()
            if self.kind == "topk_features":
                energy = x.pow(2).mean(dim=0)
                idx = energy.topk(rank).indices
                projected_grad = torch.zeros_like(grad)
                projected_grad[:, idx] = grad[:, idx]
                captured = float((energy[idx].sum() / energy.sum().clamp_min(1e-12)).item())
            else:
                basis, captured = self._basis(module, x, rank)
                projected_grad = grad @ basis @ basis.T
            module.weight.grad.copy_(projected_grad)
            cosine = torch.nn.functional.cosine_similarity(
                original.flatten(), projected_grad.flatten(), dim=0
            ).item()
            projected += 1
            ranks.append(rank)
            energies.append(captured)
            cosines.append(cosine)
        return ProjectionStats(
            projected_layers=projected,
            mean_rank=(sum(ranks) / len(ranks)) if ranks else 0.0,
            mean_activation_energy=(sum(energies) / len(energies)) if energies else 0.0,
            mean_grad_cosine=(sum(cosines) / len(cosines)) if cosines else 0.0,
        )
