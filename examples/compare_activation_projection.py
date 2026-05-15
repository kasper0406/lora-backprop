from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from training_methods import (
    ActivationGradientProjector,
    LocalDepthConfig,
    LocalDepthSequenceModel,
    RelationalReasoningTask,
    pick_device,
)


@dataclass
class EvalResult:
    loss: float
    acc: float


def evaluate(model, task, batch_size, device) -> EvalResult:
    model.eval()
    with torch.no_grad():
        batch = task.sample_batch(batch_size, device=device)
        logits, _ = model.forward_global(batch.input_ids, batch.retrieved_ids)
        loss = F.cross_entropy(logits, batch.target_ids).item()
        acc = (logits.argmax(-1) == batch.target_ids).float().mean().item()
    model.train()
    return EvalResult(loss, acc)


def run(kind: str, args, device) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    task = RelationalReasoningTask(args.n_entities, args.max_hops, args.n_distractors)
    model = LocalDepthSequenceModel(
        LocalDepthConfig(
            vocab_size=task.vocab_size,
            d_model=args.d_model,
            d_state=args.d_state,
            n_layers=args.n_layers,
        )
    ).to(device)
    projector = ActivationGradientProjector(model, rank=args.rank, kind=kind)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.perf_counter()

    print(f"\n[{kind}] params={sum(p.numel() for p in model.parameters()):,}")
    for step in range(1, args.steps + 1):
        batch = task.sample_batch(args.batch_size, device=device)
        logits, _ = model.forward_global(batch.input_ids, batch.retrieved_ids)
        loss = F.cross_entropy(logits, batch.target_ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        stats = projector.project()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            result = evaluate(model, task, args.eval_batch_size, device)
            print(
                f"step={step:04d} train={loss.item():.4f} val={result.loss:.4f} acc={result.acc:.3f} "
                f"layers={stats.projected_layers} act_energy={stats.mean_activation_energy:.3f} "
                f"grad_cos={stats.mean_grad_cosine:.3f} elapsed={time.perf_counter() - start:.1f}s"
            )
    projector.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="none,random,activation_pca,topk_features")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--max-hops", type=int, default=3)
    p.add_argument("--n-entities", type=int, default=16)
    p.add_argument("--n-distractors", type=int, default=3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-state", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = pick_device(require_accelerator=True)
    print(f"device={device}")
    for method in args.methods.split(","):
        run(method.strip(), args, device)


if __name__ == "__main__":
    main()
