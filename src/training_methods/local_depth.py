from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .model import GatedDeltaLayer, RMSNorm


@dataclass
class LocalDepthConfig:
    vocab_size: int
    d_model: int = 64
    d_state: int = 32
    n_layers: int = 8
    use_log_skip: bool = False


class SkipMixer(nn.Module):
    """Mix the previous layer with sparse ancestors at offsets 2, 4, 8, ..."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: torch.Tensor, ancestors: list[torch.Tensor]) -> torch.Tensor:
        if not ancestors:
            return h
        pooled = torch.stack(ancestors, dim=0).mean(dim=0)
        return h + self.alpha * self.proj(self.norm(pooled))


class LocalDepthSequenceModel(nn.Module):
    """Sequence model built to compare global vs strictly local depth training."""

    def __init__(self, config: LocalDepthConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [GatedDeltaLayer(config.d_model, config.d_state) for _ in range(config.n_layers)]
        )
        self.skip_mixers = nn.ModuleList([SkipMixer(config.d_model) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.heads = nn.ModuleList(
            [nn.Linear(config.d_model, config.vocab_size, bias=False) for _ in range(config.n_layers)]
        )

    def _ancestors(self, states: list[torch.Tensor], idx: int) -> list[torch.Tensor]:
        if not self.config.use_log_skip or idx < 2:
            return []
        ancestors: list[torch.Tensor] = []
        offset = 2
        while offset <= idx:
            ancestors.append(states[idx - offset])
            offset *= 2
        return ancestors

    def _prepare_sequence(self, input_ids: torch.Tensor, retrieved_ids: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([input_ids, retrieved_ids], dim=1)
        return self.embedding(joined)

    def forward_global(
        self,
        input_ids: torch.Tensor,
        retrieved_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = self._prepare_sequence(input_ids, retrieved_ids)
        states: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        for idx, (layer, head) in enumerate(zip(self.layers, self.heads)):
            h = self.skip_mixers[idx](h, self._ancestors(states, idx))
            h, _ = layer(h)
            states.append(h)
            logits.append(head(self.final_norm(h[:, -1])))
        return logits[-1], logits


    @torch.no_grad()
    def _forward_cache(
        self,
        input_ids: torch.Tensor,
        retrieved_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        h = self._prepare_sequence(input_ids, retrieved_ids)
        layer_inputs: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            mixed = self.skip_mixers[idx](h, self._ancestors(states, idx))
            layer_inputs.append(mixed.detach())
            h, _ = layer(mixed)
            states.append(h.detach())
        return self._prepare_sequence(input_ids, retrieved_ids).detach(), layer_inputs, states

    def edge_local_step(
        self,
        input_ids: torch.Tensor,
        retrieved_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Train each head through only its current layer and direct depth parents."""
        initial, layer_inputs, _ = self._forward_cache(input_ids, retrieved_ids)
        logits: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        for idx, (layer, head) in enumerate(zip(self.layers, self.heads)):
            parent_ids: list[int] = []
            if idx > 0:
                parent_ids.append(idx - 1)
            offset = 2
            while self.config.use_log_skip and offset <= idx:
                parent_ids.append(idx - offset)
                offset *= 2

            recomputed: dict[int, torch.Tensor] = {}
            for parent_idx in parent_ids:
                parent_out, _ = self.layers[parent_idx](layer_inputs[parent_idx].detach())
                recomputed[parent_idx] = parent_out

            base = recomputed[idx - 1] if idx > 0 else initial
            sparse_ids = [parent_idx for parent_idx in parent_ids if parent_idx != idx - 1]
            sparse_parents = [recomputed[parent_idx] for parent_idx in sparse_ids]
            mixed = self.skip_mixers[idx](base, sparse_parents)
            h, _ = layer(mixed)
            layer_logits = head(self.final_norm(h[:, -1]))
            loss = F.cross_entropy(layer_logits, target_ids)
            logits.append(layer_logits)
            losses.append(loss.detach())
            loss.backward()
        return torch.stack(losses).mean(), logits

    def local_step(
        self,
        input_ids: torch.Tensor,
        retrieved_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Backprop one local loss per layer, with every incoming activation detached."""
        h = self._prepare_sequence(input_ids, retrieved_ids)
        states: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        for idx, (layer, head) in enumerate(zip(self.layers, self.heads)):
            ancestors = [state.detach() for state in self._ancestors(states, idx)]
            h = self.skip_mixers[idx](h.detach(), ancestors)
            h, _ = layer(h)
            layer_logits = head(self.final_norm(h[:, -1]))
            logits.append(layer_logits)
            losses.append(F.cross_entropy(layer_logits, target_ids))
            states.append(h.detach())
        mean_loss = torch.stack([loss.detach() for loss in losses]).mean()
        for loss in losses:
            loss.backward()
        return mean_loss, logits
