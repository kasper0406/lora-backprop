"""Wall-clock microbenchmark for compressed backward methods at scale.

The recurrent relational-reasoning task uses small linears (d≈128) where
backward arithmetic is so cheap that per-layer launch overhead dominates
total time. To honestly test the claim that LoRA-shaped or top-k backward
trades arithmetic for wall-clock, we need to be in the regime where
backward matmul actually dominates: d ≥ ~1024 with a clean stack of
nn.Linear layers and no recurrent control flow.

This script builds a deep MLP at large width, runs each method for a
fixed number of training steps, and reports forward time, backward
time, total step time, FLOP ratio, and peak memory. The model and
data are synthetic — the science here is throughput, not accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
import torch.nn.functional as F

from training_methods import (
    CompressedBackwardLinear,
    estimate_linear_backward_cost,
    pick_device,
)


class DeepMLP(nn.Module):
    """A clean stack of linear+SiLU layers — no recurrence, no per-token loops."""

    def __init__(self, d_in: int, d_hidden: int, n_layers: int, d_out: int,
                 rank: int | None, mode: str, basis_refresh_every: int = 50):
        super().__init__()
        def make(in_f: int, out_f: int) -> nn.Module:
            if rank is None or mode == "full":
                return nn.Linear(in_f, out_f)
            return CompressedBackwardLinear(
                in_f, out_f, rank=min(rank, out_f), mode=mode,
                basis_refresh_every=basis_refresh_every,
            )
        self.in_proj = make(d_in, d_hidden)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "up": make(d_hidden, d_hidden * 4),
                "down": make(d_hidden * 4, d_hidden),
            }))
        self.out_proj = make(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.in_proj(x))
        for layer in self.layers:
            h = h + layer["down"](F.silu(layer["up"](h)))
        return self.out_proj(h)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class BenchResult:
    method: str
    rank: int
    d_hidden: int
    n_layers: int
    batch_size: int
    forward_ms: float
    backward_ms: float
    total_step_ms: float
    theory_flop_ratio: float
    theory_adam_state_ratio: float
    measured_flop_ratio: float
    peak_memory_mb: float
    params: int


def bench_cell(method: str, rank: int, args, device) -> BenchResult:
    torch.manual_seed(0)
    random.seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats()

    use_rank = rank if method != "full" else None
    mode = "full" if method == "full" else method
    model = DeepMLP(args.d_in, args.d_hidden, args.n_layers, args.d_out, use_rank, mode,
                    basis_refresh_every=args.basis_refresh_every).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Theoretical accounting over the compressed linears only (full uses nn.Linear).
    if method != "full":
        estimates = [
            estimate_linear_backward_cost(
                token_count=args.batch_size,
                in_features=m.in_features,
                out_features=m.out_features,
                rank=m.rank,
                mode=m.mode,
            )
            for m in model.modules()
            if isinstance(m, CompressedBackwardLinear)
        ]
        dense_flops = sum(e.dense_backward_flops for e in estimates)
        method_flops = sum(e.method_backward_flops for e in estimates)
        dense_state = sum(e.dense_adam_state_elements for e in estimates)
        method_state = sum(e.method_adam_state_elements for e in estimates)
        theory_flop_ratio = method_flops / dense_flops
        theory_state_ratio = method_state / dense_state
    else:
        theory_flop_ratio = 1.0
        theory_state_ratio = 1.0

    x = torch.randn(args.batch_size, args.d_in, device=device)
    y = torch.randn(args.batch_size, args.d_out, device=device)

    # Warmup — let cuBLAS pick kernels, cudnn pick algos, etc.
    for _ in range(args.warmup_steps):
        logits = model(x)
        loss = F.mse_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    cuda_sync()

    # Measure forward, backward, total separately.
    forward_times = []
    backward_times = []
    total_times = []
    for _ in range(args.bench_steps):
        cuda_sync()
        t0 = time.perf_counter()
        logits = model(x)
        loss = F.mse_loss(logits, y)
        cuda_sync()
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        cuda_sync()
        t2 = time.perf_counter()
        opt.step()
        cuda_sync()
        t3 = time.perf_counter()
        forward_times.append(t1 - t0)
        backward_times.append(t2 - t1)
        total_times.append(t3 - t0)

    # Use the median to avoid outliers.
    def median(xs):
        return sorted(xs)[len(xs)//2]

    measured_flop_ratio = 1.0
    if method != "full":
        # Sum over the compressed linears' reported per-call FLOPs.
        dense = sum(m.last_dense_backward_flops for m in model.modules() if isinstance(m, CompressedBackwardLinear))
        indexed = sum(m.last_indexed_backward_flops for m in model.modules() if isinstance(m, CompressedBackwardLinear))
        if dense:
            measured_flop_ratio = indexed / dense

    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
    n_params = sum(p.numel() for p in model.parameters())

    return BenchResult(
        method=method,
        rank=rank,
        d_hidden=args.d_hidden,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        forward_ms=median(forward_times) * 1e3,
        backward_ms=median(backward_times) * 1e3,
        total_step_ms=median(total_times) * 1e3,
        theory_flop_ratio=theory_flop_ratio,
        theory_adam_state_ratio=theory_state_ratio,
        measured_flop_ratio=measured_flop_ratio,
        peak_memory_mb=peak_mb,
        params=n_params,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="full,batch_topk,ema_topk,random_lowrank,activation_pca_lowrank,activation_sketch_lowrank")
    p.add_argument("--ranks", default="16,64,256")
    p.add_argument("--d-hidden", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--d-in", type=int, default=512)
    p.add_argument("--d-out", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--bench-steps", type=int, default=30)
    p.add_argument("--basis-refresh-every", type=int, default=50,
                   help="refresh the activation-PCA/sketch basis every N forwards; "
                        "high values amortize the eigh/QR cost, which is otherwise dominant")
    p.add_argument("--output", default="results/wallclock_bench.json")
    args = p.parse_args()

    device = pick_device(require_accelerator=True)
    print(f"device={device} torch={torch.__version__}")
    if device.type == "cuda":
        print(f"cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ranks = sorted({int(r) for r in args.ranks.split(",")})

    cells = []
    full_result: BenchResult | None = None
    for method in methods:
        if method == "full":
            print(f"\n--- {method} ---")
            r = bench_cell(method, ranks[-1], args, device)
            full_result = r
            cells.append(asdict(r))
            print(format_row(r, full_result))
        else:
            for rank in ranks:
                print(f"\n--- {method} rank={rank} ---")
                r = bench_cell(method, rank, args, device)
                cells.append(asdict(r))
                print(format_row(r, full_result))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "cells": cells}, indent=2))

    # Pretty summary.
    print("\n\n=== summary (d_hidden={}, n_layers={}, batch={}, gpu={}) ===".format(
        args.d_hidden, args.n_layers, args.batch_size,
        torch.cuda.get_device_name(0) if device.type == "cuda" else device.type))
    print(f"{'method':28s} {'rank':>5} {'fwd_ms':>8} {'bwd_ms':>8} {'tot_ms':>8} {'wall_x':>7} {'flop_ratio':>11} {'peak_MB':>8}")
    full_total = full_result.total_step_ms if full_result else None
    full_bwd = full_result.backward_ms if full_result else None
    for c in cells:
        wallx = (full_total / c['total_step_ms']) if full_total else 1.0
        print(f"{c['method']:28s} {c['rank']:>5} "
              f"{c['forward_ms']:>8.2f} {c['backward_ms']:>8.2f} {c['total_step_ms']:>8.2f} "
              f"{wallx:>7.2f} {c['measured_flop_ratio']:>11.3f} {c['peak_memory_mb']:>8.0f}")


def format_row(r: BenchResult, full: BenchResult | None) -> str:
    bwd_x = (full.backward_ms / r.backward_ms) if full and full.backward_ms > 0 else 1.0
    tot_x = (full.total_step_ms / r.total_step_ms) if full and full.total_step_ms > 0 else 1.0
    return (f"  fwd={r.forward_ms:.2f}ms bwd={r.backward_ms:.2f}ms tot={r.total_step_ms:.2f}ms  "
            f"bwd_speedup={bwd_x:.2f}x tot_speedup={tot_x:.2f}x  "
            f"flop_r={r.measured_flop_ratio:.3f} peak={r.peak_memory_mb:.0f}MB")


if __name__ == "__main__":
    main()
