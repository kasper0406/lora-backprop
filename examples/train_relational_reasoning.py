from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from training_methods import (
    GatedDeltaFiLMRAGModel,
    ModelConfig,
    RelationalReasoningTask,
    pick_device,
)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--think-steps", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=3)
    args = parser.parse_args()

    device = pick_device(require_accelerator=True)
    task = RelationalReasoningTask(max_hops=args.max_hops)
    model = GatedDeltaFiLMRAGModel(ModelConfig(vocab_size=task.vocab_size)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = task.sample_batch(args.batch_size, device=device)
        logits = model(batch.input_ids, batch.retrieved_ids, think_steps=args.think_steps)
        loss = F.cross_entropy(logits[:, -1], batch.target_ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 25 == 0 or step == 1:
            pred = logits[:, -1].argmax(dim=-1)
            acc = (pred == batch.target_ids).float().mean().item()
            elapsed = time.perf_counter() - start
            per_hop = []
            for hop in range(1, args.max_hops + 1):
                mask = batch.hops == hop
                if mask.any():
                    per_hop.append(f"h{hop}={(pred[mask] == batch.target_ids[mask]).float().mean().item():.2f}")
            print(
                f"step={step:04d} loss={loss.item():.4f} acc={acc:.3f} "
                f"{' '.join(per_hop)} elapsed={elapsed:.1f}s"
            )


if __name__ == "__main__":
    main()
