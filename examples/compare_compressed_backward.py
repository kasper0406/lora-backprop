from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from training_methods import CompressedBackwardMLP, CompressedMLPConfig, RelationalReasoningTask, RowSparseAdamW, pick_device


def evaluate(model, task, batch_size, device):
    model.eval()
    with torch.no_grad():
        batch = task.sample_batch(batch_size, device=device)
        logits = model(batch.input_ids, batch.retrieved_ids)
        loss = F.cross_entropy(logits, batch.target_ids).item()
        acc = (logits.argmax(-1) == batch.target_ids).float().mean().item()
    model.train()
    return loss, acc


def run(mode, args, device):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    task = RelationalReasoningTask(args.n_entities, args.max_hops, args.n_distractors)
    model = CompressedBackwardMLP(
        CompressedMLPConfig(task.vocab_size, args.d_model, args.d_hidden, args.n_layers, args.rank, mode)
    ).to(device)
    if mode.endswith("_sparseopt"):
        base_mode = mode.removesuffix("_sparseopt")
        for layer in model.layers:
            layer["up"].mode = base_mode
            layer["down"].mode = base_mode
        opt = RowSparseAdamW(model, lr=args.lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    print(f"\n[{mode}] params={sum(p.numel() for p in model.parameters()):,}")
    for step in range(1, args.steps + 1):
        batch = task.sample_batch(args.batch_size, device=device)
        logits = model(batch.input_ids, batch.retrieved_ids)
        loss = F.cross_entropy(logits, batch.target_ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            val, acc = evaluate(model, task, args.eval_batch_size, device)
            state_numel = opt.state_numel() if hasattr(opt, "state_numel") else sum(v.numel() for st in opt.state.values() for v in st.values() if torch.is_tensor(v))
            print(f"step={step:04d} train={loss.item():.4f} val={val:.4f} acc={acc:.3f} active={model.active_fraction():.3f} backward_flop_ratio={model.backward_flop_ratio():.3f} opt_state_numel={state_numel} elapsed={time.perf_counter()-start:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="full,batch_topk,ema_topk,batch_topk_sparseopt,ema_topk_sparseopt")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--max-hops", type=int, default=3)
    p.add_argument("--n-entities", type=int, default=16)
    p.add_argument("--n-distractors", type=int, default=3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-hidden", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
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
