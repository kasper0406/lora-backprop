from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .compressed_backward import CompressedBackwardLinear


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    d_state: int = 32
    n_layers: int = 4
    film_source_layer: int = 3
    film_target_layer: int = 0
    max_retrieved: int = 4
    think_token_id: int = 1
    use_log_skip: bool = False
    compressed_backward_rank: int | None = None
    compressed_backward_mode: str = "full"
    compressed_backward_basis_refresh_every: int = 1


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt() * self.weight


class GatedDeltaLayer(nn.Module):
    """Small recurrent layer inspired by gated DeltaNet updates.

    This is intentionally legible rather than kernel-optimized. Each token
    proposes a candidate state and a learned gate decides how much of the
    delta should be written into the persistent state.
    """

    def __init__(self, d_model: int, d_state: int, linear_factory=None):
        super().__init__()
        make_linear = linear_factory or (lambda in_f, out_f, bias=True: nn.Linear(in_f, out_f, bias=bias))
        self.norm = RMSNorm(d_model)
        self.to_candidate = make_linear(d_model, d_state)
        self.to_gate = make_linear(d_model, d_state)
        self.to_output = make_linear(d_state, d_model)
        self.residual_gate = make_linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        if state is None:
            state = x.new_zeros(batch, self.to_candidate.out_features)

        # Batch all input-side projections across timesteps to fold ~3 linear ops
        # × T kernel launches into 3 batched matmuls. The state update remains
        # sequential, and to_output must wait for state, so they stay in the loop.
        h_all = self.norm(x)
        candidates = torch.tanh(self.to_candidate(h_all))           # [B, T, d_state]
        write_gates = torch.sigmoid(self.to_gate(h_all))            # [B, T, d_state]
        residual_scales = torch.sigmoid(self.residual_gate(x))      # [B, T, d_model]

        states: list[torch.Tensor] = []
        for t in range(seq_len):
            state = state + write_gates[:, t] * (candidates[:, t] - state)
            states.append(state)
        states_seq = torch.stack(states, dim=1)                      # [B, T, d_state]
        outputs = x + residual_scales * self.to_output(states_seq)
        return outputs, state


class FiLMBroadcast(nn.Module):
    """Late-layer state broadcasts into an early layer with one-token lag."""

    def __init__(self, d_model: int, linear_factory=None):
        super().__init__()
        make_linear = linear_factory or (lambda in_f, out_f, bias=True: nn.Linear(in_f, out_f, bias=bias))
        self.to_scale = make_linear(d_model, d_model, bias=False)
        self.to_shift = make_linear(d_model, d_model, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        lagged = torch.cat([torch.zeros_like(source[:, :1]), source[:, :-1]], dim=1)
        scale = self.to_scale(lagged)
        shift = self.to_shift(lagged)
        return x * (1 + self.alpha * scale) + self.alpha * shift


class RetrievedFactEncoder(nn.Module):
    """Compress fixed-width retrieved facts into one vector per fact."""

    def __init__(self, d_model: int, fact_width: int = 4, linear_factory=None):
        super().__init__()
        make_linear = linear_factory or (lambda in_f, out_f, bias=True: nn.Linear(in_f, out_f, bias=bias))
        self.fact_width = fact_width
        self.proj = nn.Sequential(
            make_linear(d_model * fact_width, d_model),
            nn.SiLU(),
            make_linear(d_model, d_model),
        )

    def forward(self, retrieved: torch.Tensor) -> torch.Tensor:
        batch, token_count, dim = retrieved.shape
        if token_count % self.fact_width != 0:
            raise ValueError(
                f"retrieved token count {token_count} must be divisible by fact_width={self.fact_width}"
            )
        facts = retrieved.view(batch, token_count // self.fact_width, self.fact_width * dim)
        return self.proj(facts)


class RetrievalInjector(nn.Module):
    """Attention-pools retrieved fact vectors and injects them during thought passes."""

    def __init__(self, d_model: int, linear_factory=None):
        super().__init__()
        make_linear = linear_factory or (lambda in_f, out_f, bias=True: nn.Linear(in_f, out_f, bias=bias))
        self.fact_encoder = RetrievedFactEncoder(d_model, linear_factory=linear_factory)
        self.query = make_linear(d_model, d_model, bias=False)
        self.key = make_linear(d_model, d_model, bias=False)
        self.value = make_linear(d_model, d_model, bias=False)
        self.out = make_linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], retrieved: [B, R, D]
        facts = self.fact_encoder(retrieved)
        q = self.query(x)
        k = self.key(facts)
        v = self.value(facts)
        scores = torch.einsum("btd,brd->btr", q, k) / (x.shape[-1] ** 0.5)
        weights = scores.softmax(dim=-1)
        pooled = torch.einsum("btr,brd->btd", weights, v)
        return x + self.out(pooled)


class PowerOfTwoSkipMixer(nn.Module):
    """Inject sparse depth skips from layers at offsets 2, 4, 8, ..."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: torch.Tensor, skip_states: list[torch.Tensor]) -> torch.Tensor:
        if not skip_states:
            return h
        pooled = torch.stack(skip_states, dim=0).mean(dim=0)
        return h + self.alpha * self.proj(self.norm(pooled))


class GatedDeltaFiLMRAGModel(nn.Module):
    """Toy language model with Delta-style recurrence, FiLM feedback, and thinking.

    Forward path:
      1. Vanilla recurrent pass to expose late-layer states.
      2. Second pass with one sparse FiLM broadcast from a late layer into an
         early layer, matching the architectural idea in parallel-ss-dep.
      3. Optional thought passes. When enabled, retrieved vectors are injected
         and the recurrent stack refines the hidden sequence in place.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if not (0 <= config.film_target_layer < config.n_layers):
            raise ValueError("film_target_layer out of range")
        if not (0 <= config.film_source_layer < config.n_layers):
            raise ValueError("film_source_layer out of range")
        self.config = config
        if config.compressed_backward_rank is None:
            linear_factory = lambda in_f, out_f, bias=True: nn.Linear(in_f, out_f, bias=bias)
        else:
            linear_factory = lambda in_f, out_f, bias=True: CompressedBackwardLinear(
                in_f,
                out_f,
                rank=min(config.compressed_backward_rank, out_f),
                mode=config.compressed_backward_mode,
                basis_refresh_every=config.compressed_backward_basis_refresh_every,
                bias=bias,
            )
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [GatedDeltaLayer(config.d_model, config.d_state, linear_factory=linear_factory) for _ in range(config.n_layers)]
        )
        self.skip_mixers = nn.ModuleList(
            [PowerOfTwoSkipMixer(config.d_model) for _ in range(config.n_layers)]
        )
        self.film = FiLMBroadcast(config.d_model, linear_factory=linear_factory)
        self.retrieval = RetrievalInjector(config.d_model, linear_factory=linear_factory)
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = linear_factory(config.d_model, config.vocab_size, bias=False)
        self.local_heads = nn.ModuleList(
            [linear_factory(config.d_model, config.vocab_size, bias=False) for _ in range(config.n_layers)]
        )
        self.token_gate = linear_factory(config.d_model, 1)

    def _run_stack(
        self,
        x: torch.Tensor,
        film_source: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        states: list[torch.Tensor] = []
        h = x
        for idx, layer in enumerate(self.layers):
            if self.config.use_log_skip and idx >= 2:
                offsets: list[int] = []
                offset = 2
                while offset <= idx:
                    offsets.append(offset)
                    offset *= 2
                skip_states = [states[idx - offset] for offset in offsets]
                h = self.skip_mixers[idx](h, skip_states)
            if film_source is not None and idx == self.config.film_target_layer:
                h = self.film(h, film_source)
            h, _ = layer(h)
            states.append(h)
        return h, states

    def forward(
        self,
        input_ids: torch.Tensor,
        retrieved_ids: torch.Tensor | None = None,
        think_steps: int = 0,
        return_gate: bool = False,
        return_local_logits: bool = False,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]] | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]] | torch.Tensor:
        x = self.embedding(input_ids)

        _, first_pass_states = self._run_stack(x)
        h, states = self._run_stack(x, film_source=first_pass_states[self.config.film_source_layer])

        if retrieved_ids is not None and think_steps > 0:
            retrieved = self.embedding(retrieved_ids)
            for _ in range(think_steps):
                h = self.retrieval(h, retrieved)
                h, states = self._run_stack(h)

        h = self.final_norm(h)
        logits = self.lm_head(h)
        local_logits = [head(self.final_norm(state)) for head, state in zip(self.local_heads, states)]
        emit_gate = torch.sigmoid(self.token_gate(h)).squeeze(-1)
        if return_gate and return_local_logits:
            return logits, emit_gate, local_logits
        if return_hidden_states and return_local_logits:
            return logits, local_logits, states
        if return_gate:
            return logits, emit_gate
        if return_local_logits:
            return logits, local_logits
        if return_hidden_states:
            return logits, states
        return logits

    def compressed_backward_stats(self) -> tuple[float, float]:
        active = []
        dense = 0
        indexed = 0
        for module in self.modules():
            if isinstance(module, CompressedBackwardLinear):
                active.append(module.last_active_fraction)
                dense += module.last_dense_backward_flops
                indexed += module.last_indexed_backward_flops
        return ((sum(active) / len(active)) if active else 1.0, (indexed / dense) if dense else 1.0)

    def selection_stats(self) -> tuple[float, float, float, float]:
        vals = [module.selection_stats() for module in self.modules() if isinstance(module, CompressedBackwardLinear)]
        vals = [v for v in vals if v != (0.0, 0.0, 0.0, 0.0)]
        if not vals:
            return 0.0, 0.0, 0.0, 0.0
        return tuple(sum(items) / len(items) for items in zip(*vals))

    @torch.no_grad()
    def decide_think_mask(self, input_ids: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        _, gate = self(input_ids, return_gate=True)
        return gate < threshold
