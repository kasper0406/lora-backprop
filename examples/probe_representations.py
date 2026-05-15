from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from training_methods import (
    GatedDeltaFiLMRAGModel,
    ModelConfig,
    RelationalReasoningTask,
    pick_device,
)


def make_mixed_batch(task, batch_size, step, total_steps, device, max_hops):
    progress = (step - 1) / max(1, total_steps - 1)
    probs = [max(0.05, 1.0 - progress)]
    if max_hops >= 2:
        probs.append(max(0.05, 0.35 + 0.25 * progress))
    if max_hops >= 3:
        probs.append(max(0.05, 0.05 + 0.75 * progress))
    total = sum(probs[:max_hops])
    probs = [p / total for p in probs[:max_hops]]
    chosen_hops = random.choices(range(1, max_hops + 1), weights=probs, k=batch_size)
    rows = [task.sample_batch(1, device=device, hops=h) for h in chosen_hops]
    max_retrieved = max(r.retrieved_ids.shape[1] for r in rows)
    padded = [F.pad(r.retrieved_ids, (0, max_retrieved - r.retrieved_ids.shape[1])) for r in rows]
    return type(rows[0])(
        input_ids=torch.cat([r.input_ids for r in rows], dim=0),
        target_ids=torch.cat([r.target_ids for r in rows], dim=0),
        retrieved_ids=torch.cat(padded, dim=0),
        hops=torch.cat([r.hops for r in rows], dim=0),
    )


def train_backbone(method, task, args, device):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = GatedDeltaFiLMRAGModel(
        ModelConfig(
            vocab_size=task.vocab_size,
            d_model=args.d_model,
            d_state=args.d_state,
            n_layers=args.n_layers,
            film_source_layer=args.n_layers - 1,
            film_target_layer=0,
        )
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(1, args.train_steps + 1):
        batch = make_mixed_batch(task, args.batch_size, step, args.train_steps, device, args.max_hops)
        if method == "bp":
            logits = model(batch.input_ids, batch.retrieved_ids, think_steps=args.think_steps)
            loss = F.cross_entropy(logits[:, -1], batch.target_ids)
        else:
            logits, local_logits = model(
                batch.input_ids,
                batch.retrieved_ids,
                think_steps=args.think_steps,
                return_local_logits=True,
            )
            global_loss = F.cross_entropy(logits[:, -1], batch.target_ids)
            local_losses = [F.cross_entropy(x[:, -1], batch.target_ids) for x in local_logits]
            weights = torch.linspace(args.local_weight_max, args.local_weight_min, len(local_losses), device=device)
            loss = global_loss + sum(w * l for w, l in zip(weights, local_losses)) / len(local_losses)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()


def fit_fresh_probes(model, task, args, device):
    for p in model.parameters():
        p.requires_grad_(False)
    probes = nn.ModuleList([nn.Linear(args.d_model, task.vocab_size) for _ in range(args.n_layers)]).to(device)
    opt = torch.optim.AdamW(probes.parameters(), lr=args.probe_lr)

    for step in range(1, args.probe_steps + 1):
        batch = task.sample_batch(args.batch_size, device=device)
        with torch.no_grad():
            _, states = model(
                batch.input_ids,
                batch.retrieved_ids,
                think_steps=args.think_steps,
                return_hidden_states=True,
            )
        losses = [F.cross_entropy(probe(state[:, -1]), batch.target_ids) for probe, state in zip(probes, states)]
        loss = sum(losses) / len(losses)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return probes.eval()


def evaluate(model, probes, task, args, device):
    batch = task.sample_batch(args.eval_batch_size, device=device)
    with torch.no_grad():
        logits, states = model(
            batch.input_ids,
            batch.retrieved_ids,
            think_steps=args.think_steps,
            return_hidden_states=True,
        )
        final_acc = (logits[:, -1].argmax(-1) == batch.target_ids).float().mean().item()
        probe_accs = [
            (probe(state[:, -1]).argmax(-1) == batch.target_ids).float().mean().item()
            for probe, state in zip(probes, states)
        ]
    return final_acc, probe_accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=450)
    p.add_argument("--probe-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=1024)
    p.add_argument("--think-steps", type=int, default=3)
    p.add_argument("--max-hops", type=int, default=3)
    p.add_argument("--n-entities", type=int, default=16)
    p.add_argument("--n-distractors", type=int, default=3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-state", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--probe-lr", type=float, default=3e-3)
    p.add_argument("--local-weight-max", type=float, default=0.6)
    p.add_argument("--local-weight-min", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = pick_device(require_accelerator=True)
    task = RelationalReasoningTask(args.n_entities, args.max_hops, args.n_distractors)
    print(f"device={device}")
    for method in ("bp", "local_heads"):
        print(f"\\n[{method}] training backbone")
        model = train_backbone(method, task, args, device)
        probes = fit_fresh_probes(model, task, args, device)
        final_acc, probe_accs = evaluate(model, probes, task, args, device)
        probes_str = " ".join(f"l{i+1}={acc:.3f}" for i, acc in enumerate(probe_accs))
        print(f"final_acc={final_acc:.3f} fresh_probes {probes_str}")


if __name__ == "__main__":
    main()
