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

from training_methods import LocalDepthConfig, LocalDepthSequenceModel, RelationalReasoningTask, pick_device


@dataclass
class EvalResult:
    loss: float
    final_acc: float
    layer_accs: list[float]


def allocated_memory(device: torch.device) -> int | None:
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def evaluate(model, task, batch_size, device) -> EvalResult:
    model.eval()
    with torch.no_grad():
        batch = task.sample_batch(batch_size, device=device)
        logits, all_logits = model.forward_global(batch.input_ids, batch.retrieved_ids)
        loss = F.cross_entropy(logits, batch.target_ids).item()
        final_acc = (logits.argmax(-1) == batch.target_ids).float().mean().item()
        layer_accs = [
            (layer_logits.argmax(-1) == batch.target_ids).float().mean().item()
            for layer_logits in all_logits
        ]
    model.train()
    return EvalResult(loss, final_acc, layer_accs)


def run(method: str, args, device) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    task = RelationalReasoningTask(args.n_entities, args.max_hops, args.n_distractors)
    model = LocalDepthSequenceModel(
        LocalDepthConfig(
            vocab_size=task.vocab_size,
            d_model=args.d_model,
            d_state=args.d_state,
            n_layers=args.n_layers,
            use_log_skip=(method in {"local_logskip", "edge_logskip"}),
        )
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    peak = allocated_memory(device) or 0

    print(f"\n[{method}] params={sum(p.numel() for p in model.parameters()):,}")
    for step in range(1, args.steps + 1):
        batch = task.sample_batch(args.batch_size, device=device)
        opt.zero_grad(set_to_none=True)
        if method == "bp":
            logits, _ = model.forward_global(batch.input_ids, batch.retrieved_ids)
            loss = F.cross_entropy(logits, batch.target_ids)
            loss.backward()
        elif method in {"local_chain", "local_logskip"}:
            loss, _ = model.local_step(batch.input_ids, batch.retrieved_ids, batch.target_ids)
        elif method in {"edge_chain", "edge_logskip"}:
            loss, _ = model.edge_local_step(batch.input_ids, batch.retrieved_ids, batch.target_ids)
        else:
            raise ValueError(method)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        current = allocated_memory(device)
        if current is not None:
            peak = max(peak, current)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            result = evaluate(model, task, args.eval_batch_size, device)
            layer_accs = " ".join(f"l{i+1}={acc:.2f}" for i, acc in enumerate(result.layer_accs))
            mem = f" peak_mb={peak / 1024**2:.1f}" if peak else ""
            print(
                f"step={step:04d} train={loss.item():.4f} val={result.loss:.4f} "
                f"acc={result.final_acc:.3f} {layer_accs}{mem} elapsed={time.perf_counter() - start:.1f}s"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="bp,local_chain,local_logskip,edge_chain,edge_logskip")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--max-hops", type=int, default=3)
    p.add_argument("--n-entities", type=int, default=16)
    p.add_argument("--n-distractors", type=int, default=3)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-state", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=8)
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
