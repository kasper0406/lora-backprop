# Plan: tighten the LoRA-backward wall-clock + accuracy claims

This is the queued work running automatically on the Hyperstack A6000 spot VM
(`812555`, branch `lora-backward-sweep`). All results land in
`results/`. When everything completes, a `results/ALL_DONE` marker is
touched; the VM is then torn down.

## 1. Finish accuracy sweep (in progress)

- `examples/sweep_lora_backward.py` at d=128, 6 layers, 1500 steps,
  5 methods × 3 ranks × 5 seeds, lr=3e-3, BF16AdamW.
- Output: `results/lora_backward_sweep.json`.
- Auto-resumes from saved cells if interrupted.

## 2. Lr-sensitivity check on full backward

The current sweep showed full backward plateauing around step 500 (val loss
2.61) while ema_topk kept improving. That plateau is suspicious — could be
lr too high for the larger model. Need to rule out before claiming
"compression beats full".

- Same task and model as the sweep, **method=full only**, 5 seeds.
- Sweep lr ∈ {1e-3, 5e-4} (3e-3 already covered by the main sweep).
- 1500 steps per cell.
- Output: `results/lr_sensitivity_full.json`.

## 3. Large-batch wall-clock bench

Push the ema_topk speedup story from 1.75× toward 2×+ by amplifying the
fwd+bwd share of total step time.

- d=4096, 6 layers, fused AdamW.
- Batch sizes: 1024, 2048.
- Methods: full, ema_topk (rank=64).
- Output: `results/wallclock_big_batch.json`.

## 4. Tear down

When `results/ALL_DONE` exists, SCP all `results/*.json` back to the
laptop, then delete VM 812555 via the Hyperstack MCP.
