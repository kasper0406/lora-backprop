from __future__ import annotations

from dataclasses import dataclass
import random

import torch


PAD = 0
THINK = 1
QUERY = 2
SEP = 3
ARROW = 4


@dataclass
class RelationalReasoningBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    retrieved_ids: torch.Tensor
    hops: torch.Tensor


class RelationalReasoningTask:
    """Synthetic multi-hop RAG task with controllable reasoning depth.

    A sample contains a query entity and a requested hop count. Retrieved facts
    encode a directed chain plus distractor edges. The target is the entity at
    the end of the relevant chain. Example, schematically:

      query:  Q e3 hop2
      retrieved: e3 -> e8 ; e8 -> e1 ; e7 -> e2 ; e4 -> e9
      target: e1

    This is still synthetic, but it asks the model to combine retrieval, state,
    and iterative computation rather than merely copy one looked-up value.
    """

    def __init__(
        self,
        n_entities: int = 32,
        max_hops: int = 3,
        n_distractors: int = 5,
    ):
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        self.n_entities = n_entities
        self.max_hops = max_hops
        self.n_distractors = n_distractors
        self.entity_offset = 5
        self.hop_offset = self.entity_offset + n_entities
        self.vocab_size = self.hop_offset + max_hops

    def _entity(self, idx: int) -> int:
        return self.entity_offset + idx

    def _hop_token(self, hops: int) -> int:
        return self.hop_offset + hops - 1

    def _sample_chain(self, hops: int) -> list[int]:
        chain = random.sample(range(self.n_entities), hops + 1)
        return chain

    def _sample_example(self, hops: int | None = None) -> tuple[list[int], int, list[int], int]:
        hops = hops or random.randint(1, self.max_hops)
        chain = self._sample_chain(hops)

        facts: list[tuple[int, int]] = list(zip(chain[:-1], chain[1:]))
        used_edges = set(facts)
        while len(facts) < hops + self.n_distractors:
            src, dst = random.sample(range(self.n_entities), 2)
            edge = (src, dst)
            if edge not in used_edges:
                used_edges.add(edge)
                facts.append(edge)
        random.shuffle(facts)

        prompt = [QUERY, self._entity(chain[0]), self._hop_token(hops), SEP]
        retrieved: list[int] = []
        for src, dst in facts:
            retrieved.extend([self._entity(src), ARROW, self._entity(dst), SEP])

        return prompt, self._entity(chain[-1]), retrieved, hops

    def sample_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
        hops: int | None = None,
    ) -> RelationalReasoningBatch:
        samples = [self._sample_example(hops=hops) for _ in range(batch_size)]
        prompts, targets, retrieved, hop_counts = zip(*samples)
        max_retrieved_len = max(len(row) for row in retrieved)
        padded_retrieved = [
            row + [PAD] * (max_retrieved_len - len(row))
            for row in retrieved
        ]
        return RelationalReasoningBatch(
            input_ids=torch.tensor(prompts, dtype=torch.long, device=device),
            target_ids=torch.tensor(targets, dtype=torch.long, device=device),
            retrieved_ids=torch.tensor(padded_retrieved, dtype=torch.long, device=device),
            hops=torch.tensor(hop_counts, dtype=torch.long, device=device),
        )
