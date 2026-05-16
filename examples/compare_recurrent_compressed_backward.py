from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from training_methods import (
    CompressedBackwardLinear,
    BF16AdamW,
    GatedDeltaFiLMRAGModel,
    Int8AdamW,
    ModelConfig,
    RelationalReasoningTask,
    estimate_linear_backward_cost,
    pick_device,
)


def evaluate(model, task, batch_size, think_steps, device, max_hops):
    model.eval()
    with torch.no_grad():
        batch = task.sample_batch(batch_size, device=device)
        logits = model(batch.input_ids, batch.retrieved_ids, think_steps=think_steps)
        loss = F.cross_entropy(logits[:, -1], batch.target_ids).item()
        pred = logits[:, -1].argmax(-1)
        acc = (pred == batch.target_ids).float().mean().item()
        per_hop = []
        for hop in range(1, max_hops + 1):
            mask = batch.hops == hop
            if mask.any():
                per_hop.append((hop, (pred[mask] == batch.target_ids[mask]).float().mean().item()))
    model.train()
    return loss, acc, per_hop


def run(mode, args, device):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    task = RelationalReasoningTask(args.n_entities, args.max_hops, args.n_distractors)
    rank = None if mode == "full" else args.rank
    compressed_mode = "full" if mode == "full" else mode
    model = GatedDeltaFiLMRAGModel(ModelConfig(
        vocab_size=task.vocab_size,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        film_source_layer=args.n_layers - 1,
        film_target_layer=0,
        compressed_backward_rank=rank,
        compressed_backward_mode=compressed_mode,
        compressed_backward_basis_refresh_every=args.basis_refresh_every,
    )).to(device)
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "bf16_adamw":
        opt = BF16AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "int8_adamw":
        opt = Int8AdamW(model.parameters(), lr=args.lr, block_size=args.int8_block_size)
    else:
        raise ValueError(args.optimizer)
    start = time.perf_counter()
    print(f"\n[{mode}] optimizer={args.optimizer} params={sum(p.numel() for p in model.parameters()):,}")
    estimates = [
        estimate_linear_backward_cost(
            token_count=args.batch_size * 4,
            in_features=module.in_features,
            out_features=module.out_features,
            rank=module.rank,
            mode=module.mode,
        )
        for module in model.modules()
        if isinstance(module, CompressedBackwardLinear)
    ]
    if estimates:
        dense_flops = sum(item.dense_backward_flops for item in estimates)
        method_flops = sum(item.method_backward_flops for item in estimates)
        dense_state = sum(item.dense_adam_state_elements for item in estimates)
        method_state = sum(item.method_adam_state_elements for item in estimates)
        print(
            f"theory token_count≈{args.batch_size * 4} "
            f"backward_flop_ratio={method_flops / dense_flops:.3f} "
            f"adam_state_ratio={method_state / dense_state:.3f}"
        )
    for step in range(1, args.steps + 1):
        batch = task.sample_batch(args.batch_size, device=device)
        logits = model(batch.input_ids, batch.retrieved_ids, think_steps=args.think_steps)
        loss = F.cross_entropy(logits[:, -1], batch.target_ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            val, acc, per_hop = evaluate(model, task, args.eval_batch_size, args.think_steps, device, args.max_hops)
            active, ratio = model.compressed_backward_stats()
            coverage, concentration, turnover, grad_concentration = model.selection_stats()
            state_bytes = (
                opt.state_nbytes()
                if hasattr(opt, "state_nbytes")
                else sum(v.numel() * v.element_size() for st in opt.state.values() for v in st.values() if torch.is_tensor(v))
            )
            hops = " ".join(f"h{k}={v:.2f}" for k, v in per_hop)
            diag = ""
            if hasattr(opt, "last_diagnostics") and opt.last_diagnostics:
                diag = " " + " ".join(f"{k}={v:.3g}" for k, v in opt.last_diagnostics.items())
            print(f"step={step:04d} train={loss.item():.4f} val={val:.4f} acc={acc:.3f} {hops} active={active:.3f} backward_flop_ratio={ratio:.3f} coverage={coverage:.3f} concentration={concentration:.3f} turnover={turnover:.3f} grad_concentration={grad_concentration:.3f} opt_state_bytes={state_bytes}{diag} elapsed={time.perf_counter()-start:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="full,batch_topk,ema_topk,random_lowrank,activation_sketch_lowrank")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--think-steps", type=int, default=3)
    p.add_argument("--max-hops", type=int, default=3)
    p.add_argument("--n-entities", type=int, default=16)
    p.add_argument("--n-distractors", type=int, default=3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-state", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--optimizer", choices=["adamw", "bf16_adamw", "int8_adamw"], default="bf16_adamw")
    p.add_argument("--basis-refresh-every", type=int, default=10)
    p.add_argument("--int8-block-size", type=int, default=256)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = pick_device(require_accelerator=True)
    print(f"device={device}")
    for method in args.methods.split(","):
        run(method.strip(), args, device)

if __name__ == "__main__":
    main()
