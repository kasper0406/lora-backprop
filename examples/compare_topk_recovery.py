from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from training_methods import BF16AdamW, CompressedBackwardLinear, GatedDeltaFiLMRAGModel, ModelConfig, RelationalReasoningTask, pick_device


def set_mode(model, mode: str) -> None:
    for module in model.modules():
        if isinstance(module, CompressedBackwardLinear):
            module.mode = mode


def evaluate(model, task, batch_size, think_steps, device):
    model.eval()
    with torch.no_grad():
        batch = task.sample_batch(batch_size, device=device)
        logits = model(batch.input_ids, batch.retrieved_ids, think_steps=think_steps)
        pred = logits[:, -1].argmax(-1)
        acc = (pred == batch.target_ids).float().mean().item()
    model.train()
    return acc


def run(method: str, args, device) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    task = RelationalReasoningTask(args.n_entities, args.max_hops, args.n_distractors)
    initial_mode = {
        "full": "full",
        "ema_topk": "ema_topk",
        "ema_then_full": "ema_topk",
        "activation_pca_lowrank": "activation_pca_lowrank",
        "activation_sketch_lowrank": "activation_sketch_lowrank",
    }[method]
    rank = None if method == "full" else args.rank
    compressed_mode = "full" if method == "full" else initial_mode
    model = GatedDeltaFiLMRAGModel(
        ModelConfig(
            vocab_size=task.vocab_size,
            d_model=args.d_model,
            d_state=args.d_state,
            n_layers=args.n_layers,
            film_source_layer=args.n_layers - 1,
            film_target_layer=0,
            compressed_backward_rank=rank,
            compressed_backward_mode=compressed_mode,
            compressed_backward_basis_refresh_every=args.basis_refresh_every,
        )
    ).to(device)
    opt = BF16AdamW(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    print(f"\n[{method}]")
    for step in range(1, args.steps + 1):
        if method == "ema_then_full" and step == args.switch_step:
            set_mode(model, "full")
        batch = task.sample_batch(args.batch_size, device=device)
        logits = model(batch.input_ids, batch.retrieved_ids, think_steps=args.think_steps)
        loss = F.cross_entropy(logits[:, -1], batch.target_ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            acc = evaluate(model, task, args.eval_batch_size, args.think_steps, device)
            coverage, concentration, turnover, grad_concentration = model.selection_stats()
            print(
                f"step={step:04d} acc={acc:.3f} coverage={coverage:.3f} "
                f"concentration={concentration:.3f} turnover={turnover:.3f} "
                f"grad_concentration={grad_concentration:.3f} "
                f"elapsed={time.perf_counter() - start:.1f}s"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="full,ema_topk,ema_then_full,activation_pca_lowrank,activation_sketch_lowrank")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--switch-step", type=int, default=151)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--think-steps", type=int, default=3)
    p.add_argument("--max-hops", type=int, default=3)
    p.add_argument("--n-entities", type=int, default=16)
    p.add_argument("--n-distractors", type=int, default=3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-state", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--basis-refresh-every", type=int, default=10)
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
