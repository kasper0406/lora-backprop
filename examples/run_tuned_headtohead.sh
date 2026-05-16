#!/bin/bash
# Head-to-head: tuned ema_topk vs tuned full backward across lr ∈ {5e-4, 1e-3, 3e-3, 1e-2}.
# Touches results/TUNED_DONE when complete so the monitor can tear down.
set -eux
cd "$HOME/lora-backprop"
PY=.venv/bin/python
mkdir -p results/tuned

# 4 lrs × (full + 2 ranks ema) × 5 seeds = 60 cells. ~150s/cell ≈ 2.5h.
for LR in 5e-4 1e-3 3e-3 1e-2; do
    echo "=== lr=$LR @ $(date -Iseconds) ==="

    # full backward
    $PY examples/sweep_lora_backward.py \
        --steps 1500 \
        --d-model 128 --d-state 64 --n-layers 6 \
        --batch-size 64 --eval-batch-size 256 \
        --n-entities 16 --n-distractors 3 --max-hops 3 \
        --lr $LR \
        --methods full --ranks 32 --seeds 0,1,2,3,4 \
        --output results/tuned/full_lr${LR}.json \
        --log-every 500 2>&1 | tee -a /tmp/tuned_step1.log

    # ema_topk at 2 ranks
    $PY examples/sweep_lora_backward.py \
        --steps 1500 \
        --d-model 128 --d-state 64 --n-layers 6 \
        --batch-size 64 --eval-batch-size 256 \
        --n-entities 16 --n-distractors 3 --max-hops 3 \
        --lr $LR \
        --methods ema_topk --ranks 16,32 --seeds 0,1,2,3,4 \
        --output results/tuned/ema_topk_lr${LR}.json \
        --log-every 500 2>&1 | tee -a /tmp/tuned_step2.log
done

echo "=== TUNED DONE @ $(date -Iseconds) ==="
touch results/TUNED_DONE
