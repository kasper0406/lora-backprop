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
    GatedDeltaFiLMRAGModel,
    ModelConfig,
    RelationalReasoningTask,
    pick_device,
)


@dataclass
class EvalResult:
    loss: float
    acc: float
    per_hop: dict[int, float]
    local_accs: list[float]


def evaluate(model, task, batch_size, think_steps, device, max_hops) -> EvalResult:
    model.eval()
    with torch.no_grad():
        batch = task.sample_batch(batch_size, device=device)
        logits, local_logits = model(
            batch.input_ids,
            batch.retrieved_ids,
            think_steps=think_steps,
            return_local_logits=True,
        )
        loss = F.cross_entropy(logits[:, -1], batch.target_ids).item()
        pred = logits[:, -1].argmax(dim=-1)
        acc = (pred == batch.target_ids).float().mean().item()
        per_hop = {}
        for hop in range(1, max_hops + 1):
            mask = batch.hops == hop
            if mask.any():
                per_hop[hop] = (pred[mask] == batch.target_ids[mask]).float().mean().item()
        local_accs = [
            (layer_logits[:, -1].argmax(dim=-1) == batch.target_ids).float().mean().item()
            for layer_logits in local_logits
        ]
    model.train()
    return EvalResult(loss=loss, acc=acc, per_hop=per_hop, local_accs=local_accs)


def run(method: str, args, device) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    task = RelationalReasoningTask(
        n_entities=args.n_entities,
        max_hops=args.max_hops,
        n_distractors=args.n_distractors,
    )
    model = GatedDeltaFiLMRAGModel(
        ModelConfig(
            vocab_size=task.vocab_size,
            d_model=args.d_model,
            d_state=args.d_state,
            n_layers=args.n_layers,
            film_source_layer=args.n_layers - 1,
            film_target_layer=0,
            use_log_skip=args.use_log_skip,
        )
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.perf_counter()

    print(f"\n[{method}] params={sum(p.numel() for p in model.parameters()):,}")
    initial_norms = {
        name: param.detach().norm().item()
        for name, param in model.named_parameters()
        if name.startswith("layers.") and name.endswith("to_candidate.weight")
    }
    for step in range(1, args.steps + 1):
        if args.curriculum == "hard":
            phase = min(args.max_hops, 1 + (step - 1) * args.max_hops // args.steps)
            batch = task.sample_batch(args.batch_size, device=device, hops=phase)
        elif args.curriculum == "mixed":
            progress = (step - 1) / max(1, args.steps - 1)
            hard_mass = progress
            probs = [max(0.05, 1.0 - hard_mass)]
            if args.max_hops >= 2:
                probs.append(max(0.05, 0.35 + 0.25 * progress))
            if args.max_hops >= 3:
                probs.append(max(0.05, 0.05 + 0.75 * progress))
            total = sum(probs[:args.max_hops])
            probs = [p / total for p in probs[:args.max_hops]]
            chosen_hops = random.choices(range(1, args.max_hops + 1), weights=probs, k=args.batch_size)
            rows = [task.sample_batch(1, device=device, hops=h) for h in chosen_hops]
            max_retrieved = max(r.retrieved_ids.shape[1] for r in rows)
            padded_retrieved = []
            for row in rows:
                pad = max_retrieved - row.retrieved_ids.shape[1]
                if pad > 0:
                    row_ids = F.pad(row.retrieved_ids, (0, pad))
                else:
                    row_ids = row.retrieved_ids
                padded_retrieved.append(row_ids)
            batch = type(rows[0])(
                input_ids=torch.cat([r.input_ids for r in rows], dim=0),
                target_ids=torch.cat([r.target_ids for r in rows], dim=0),
                retrieved_ids=torch.cat(padded_retrieved, dim=0),
                hops=torch.cat([r.hops for r in rows], dim=0),
            )
        else:
            batch = task.sample_batch(args.batch_size, device=device)
        if method == "bp":
            logits = model(batch.input_ids, batch.retrieved_ids, think_steps=args.think_steps)
            loss = F.cross_entropy(logits[:, -1], batch.target_ids)
        elif method == "local_heads":
            logits, local_logits = model(
                batch.input_ids,
                batch.retrieved_ids,
                think_steps=args.think_steps,
                return_local_logits=True,
            )
            global_loss = F.cross_entropy(logits[:, -1], batch.target_ids)
            local_losses = [F.cross_entropy(layer_logits[:, -1], batch.target_ids) for layer_logits in local_logits]
            weights = torch.linspace(args.local_weight_max, args.local_weight_min, len(local_losses), device=device)
            loss = global_loss + sum(w * l for w, l in zip(weights, local_losses)) / len(local_losses)
        else:
            raise ValueError(method)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            result = evaluate(model, task, args.eval_batch_size, args.think_steps, device, args.max_hops)
            elapsed = time.perf_counter() - start
            hops = " ".join(f"h{k}={v:.2f}" for k, v in result.per_hop.items())
            locals_ = " ".join(f"l{i+1}={v:.2f}" for i, v in enumerate(result.local_accs))
            layer_updates = []
            for name, param in model.named_parameters():
                if name in initial_norms:
                    rel = (param.detach().norm().item() - initial_norms[name]) / max(initial_norms[name], 1e-8)
                    layer_updates.append(rel)
            updates = " ".join(f"u{i+1}={v:+.2f}" for i, v in enumerate(layer_updates))
            print(
                f"step={step:04d} train={loss.item():.4f} val={result.loss:.4f} "
                f"acc={result.acc:.3f} {hops} | {locals_} | {updates} elapsed={elapsed:.1f}s"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="bp,local_heads")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--think-steps", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--n-entities", type=int, default=16)
    parser.add_argument("--n-distractors", type=int, default=3)
    parser.add_argument("--curriculum", choices=["none", "hard", "mixed"], default="none")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--use-log-skip", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--local-weight-max", type=float, default=0.6)
    parser.add_argument("--local-weight-min", type=float, default=0.15)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = pick_device(require_accelerator=True)
    print(f"device={device}")
    for method in args.methods.split(","):
        run(method.strip(), args, device)


if __name__ == "__main__":
    main()
