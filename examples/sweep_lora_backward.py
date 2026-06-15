"""Multi-seed sweep of the LoRA-shaped backward transport rule at scale.

The question this script tests: does dense low-rank backward transport
(z = dy B; dx = z B^T W; dW = B z^T x) become competitive with, or beat,
EMA top-k sparse backward and the dense baseline once we move past the
toy 4-layer / d=64 / 300-step setting?

It compares full, ema_topk, random_lowrank, activation_pca_lowrank, and
activation_sketch_lowrank across multiple ranks and seeds on a larger
recurrent FiLM/RAG model and a harder relational-reasoning task. Results
are appended to a JSON file after every cell so partial progress
survives a spot preemption.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from training_methods import (
    BF16AdamW,
    CompressedBackwardLinear,
    GatedDeltaFiLMRAGModel,
    ModelConfig,
    RelationalReasoningTask,
    estimate_linear_backward_cost,
    pick_device,
)


def set_eta(model, eta):
    """Push the stochastic top-k temperature into every CompressedBackwardLinear."""
    for m in model.modules():
        if isinstance(m, CompressedBackwardLinear):
            m.eta = eta


def eta_at_step(step: int, total_steps: int, eta_start: float | None, eta_end: float | None):
    """Linear schedule from eta_start to eta_end over [1, total_steps].
    Returns None when eta_start is None (deterministic mode = original behaviour)."""
    if eta_start is None:
        return None
    if eta_end is None or total_steps <= 1:
        return float(eta_start)
    frac = (step - 1) / (total_steps - 1)
    return float(eta_start + (eta_end - eta_start) * frac)


@dataclass
class CellResult:
    method: str
    rank: int
    seed: int
    final_acc: float
    final_loss: float
    per_hop_acc: dict
    backward_flop_ratio: float
    theory_flop_ratio: float
    theory_adam_state_ratio: float
    selection_coverage: float
    selection_concentration: float
    grad_concentration: float
    opt_state_bytes: int
    wall_clock_s: float
    history: list
    eta_start: float | None = None
    eta_end: float | None = None


def evaluate(model, task, batch_size, think_steps, device, max_hops):
    model.eval()
    with torch.no_grad():
        batch = task.sample_batch(batch_size, device=device)
        logits = model(batch.input_ids, batch.retrieved_ids, think_steps=think_steps)
        loss = F.cross_entropy(logits[:, -1], batch.target_ids).item()
        pred = logits[:, -1].argmax(-1)
        acc = (pred == batch.target_ids).float().mean().item()
        per_hop = {}
        for hop in range(1, max_hops + 1):
            mask = batch.hops == hop
            if mask.any():
                per_hop[hop] = (pred[mask] == batch.target_ids[mask]).float().mean().item()
    model.train()
    return loss, acc, per_hop


def run_cell(method: str, rank: int, seed: int, args, device) -> CellResult:
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    task = RelationalReasoningTask(args.n_entities, args.max_hops, args.n_distractors)
    use_rank = None if method == "full" else rank
    compressed_mode = "full" if method == "full" else method
    cfg = ModelConfig(
        vocab_size=task.vocab_size,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        film_source_layer=args.n_layers - 1,
        film_target_layer=0,
        compressed_backward_rank=use_rank,
        compressed_backward_mode=compressed_mode,
        compressed_backward_basis_refresh_every=args.basis_refresh_every,
    )
    model = GatedDeltaFiLMRAGModel(cfg).to(device)
    opt = BF16AdamW(model.parameters(), lr=args.lr)

    # Theoretical accounting.
    estimates = [
        estimate_linear_backward_cost(
            token_count=args.batch_size * 4,
            in_features=m.in_features,
            out_features=m.out_features,
            rank=m.rank,
            mode=m.mode,
        )
        for m in model.modules()
        if isinstance(m, CompressedBackwardLinear)
    ]
    if estimates:
        dense_flops = sum(e.dense_backward_flops for e in estimates)
        method_flops = sum(e.method_backward_flops for e in estimates)
        dense_state = sum(e.dense_adam_state_elements for e in estimates)
        method_state = sum(e.method_adam_state_elements for e in estimates)
        theory_flop_ratio = method_flops / dense_flops
        theory_state_ratio = method_state / dense_state
    else:
        theory_flop_ratio = 1.0
        theory_state_ratio = 1.0

    start = time.perf_counter()
    history = []
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [{method:30s} rank={rank:3d} seed={seed}] params={n_params:,} "
          f"theory_flop={theory_flop_ratio:.3f}")

    for step in range(1, args.steps + 1):
        eta = eta_at_step(step, args.steps, args.eta_start, args.eta_end)
        if eta is not None:
            set_eta(model, eta)
        batch = task.sample_batch(args.batch_size, device=device)
        logits = model(batch.input_ids, batch.retrieved_ids, think_steps=args.think_steps)
        loss = F.cross_entropy(logits[:, -1], batch.target_ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            val, acc, per_hop = evaluate(model, task, args.eval_batch_size, args.think_steps, device, args.max_hops)
            history.append({
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": val,
                "acc": acc,
                "per_hop": per_hop,
                "elapsed": time.perf_counter() - start,
            })
            hops_str = " ".join(f"h{k}={v:.2f}" for k, v in per_hop.items())
            print(f"    step={step:04d} train={loss.item():.4f} val={val:.4f} acc={acc:.3f} {hops_str} "
                  f"elapsed={time.perf_counter()-start:.1f}s")

    # Final stats.
    val_loss, final_acc, per_hop = evaluate(model, task, args.eval_batch_size, args.think_steps, device, args.max_hops)
    active, ratio = model.compressed_backward_stats()
    coverage, concentration, _turnover, grad_concentration = model.selection_stats()
    state_bytes = (
        opt.state_nbytes()
        if hasattr(opt, "state_nbytes")
        else sum(v.numel() * v.element_size() for st in opt.state.values() for v in st.values() if torch.is_tensor(v))
    )
    wall = time.perf_counter() - start
    return CellResult(
        method=method,
        rank=rank,
        seed=seed,
        final_acc=final_acc,
        final_loss=val_loss,
        per_hop_acc=per_hop,
        backward_flop_ratio=ratio,
        theory_flop_ratio=theory_flop_ratio,
        theory_adam_state_ratio=theory_state_ratio,
        selection_coverage=coverage,
        selection_concentration=concentration,
        grad_concentration=grad_concentration,
        opt_state_bytes=state_bytes,
        wall_clock_s=wall,
        history=history,
        eta_start=args.eta_start,
        eta_end=args.eta_end,
    )


def cells_for(methods, ranks, seeds):
    for method in methods:
        if method == "full":
            # rank is irrelevant for full; emit one rank value (the largest) per seed.
            for seed in seeds:
                yield method, ranks[-1], seed
        else:
            for rank in ranks:
                for seed in seeds:
                    yield method, rank, seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="full,ema_topk,random_lowrank,activation_pca_lowrank,activation_sketch_lowrank")
    p.add_argument("--ranks", default="8,16,32")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--think-steps", type=int, default=3)
    p.add_argument("--max-hops", type=int, default=4)
    p.add_argument("--n-entities", type=int, default=32)
    p.add_argument("--n-distractors", type=int, default=8)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--d-state", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--basis-refresh-every", type=int, default=10)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eta-start", type=float, default=None,
                   help="Stochastic top-k Gumbel temperature at step 1. None = deterministic argmax (original behaviour)")
    p.add_argument("--eta-end", type=float, default=None,
                   help="Stochastic top-k Gumbel temperature at last step. Linear schedule between eta_start and eta_end. None = constant eta_start")
    p.add_argument("--output", default="results/lora_backward_sweep.json")
    p.add_argument("--sanity-only", action="store_true",
                   help="Run only a short dense baseline check to confirm the task is learnable")
    p.add_argument("--sanity-steps", type=int, default=400)
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ranks = sorted({int(r) for r in args.ranks.split(",")})
    seeds = [int(s) for s in args.seeds.split(",")]

    device = pick_device(require_accelerator=True)
    print(f"device={device} torch={torch.__version__}")
    if device.type == "cuda":
        print(f"cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sanity_only:
        sanity_args = argparse.Namespace(**{**vars(args), "steps": args.sanity_steps})
        cell = run_cell("full", ranks[-1], 0, sanity_args, device)
        sanity_path = output_path.with_suffix(".sanity.json")
        sanity_path.write_text(json.dumps({"sanity": asdict(cell)}, indent=2))
        print(f"\nSANITY: final_acc={cell.final_acc:.3f} per_hop={cell.per_hop_acc}")
        print(f"sanity result written to {sanity_path}")
        return

    # Load any prior partial results so we can resume after a spot preemption.
    results = []
    seen = set()
    if output_path.exists():
        try:
            prior = json.loads(output_path.read_text())
            results = prior.get("cells", [])
            seen = {(c["method"], c["rank"], c["seed"]) for c in results}
            print(f"resuming from {output_path}: {len(seen)} cells already complete")
        except Exception as exc:
            print(f"could not parse prior results ({exc}); starting fresh")

    meta = {
        "config": vars(args),
        "device": str(device),
        "torch": torch.__version__,
    }

    todo = [c for c in cells_for(methods, ranks, seeds) if c not in seen]
    print(f"\nTotal cells: {len(todo)} pending, {len(seen)} done")
    overall_start = time.perf_counter()

    for i, (method, rank, seed) in enumerate(todo, start=1):
        print(f"\n[{i}/{len(todo)}] method={method} rank={rank} seed={seed}  "
              f"elapsed_overall={time.perf_counter()-overall_start:.1f}s")
        cell = run_cell(method, rank, seed, args, device)
        results.append(asdict(cell))
        output_path.write_text(json.dumps({"meta": meta, "cells": results}, indent=2))
        print(f"  -> final_acc={cell.final_acc:.3f} flop_ratio={cell.backward_flop_ratio:.3f} "
              f"wall={cell.wall_clock_s:.1f}s")

    print(f"\nDone. {len(results)} cells written to {output_path}")


if __name__ == "__main__":
    main()
