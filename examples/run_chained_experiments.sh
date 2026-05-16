#!/bin/bash
# Auto-chained experiment runner. Executes the queued work in PLAN_NEXT.md
# and touches results/ALL_DONE when finished.
set -eux
cd "$HOME/lora-backprop"
PY=.venv/bin/python

mkdir -p results

# ---- 1. Accuracy sweep (resume-capable; idempotent if already done) ----
echo "=== [1] accuracy sweep @ $(date -Iseconds) ==="
$PY examples/sweep_lora_backward.py \
    --steps 1500 \
    --d-model 128 --d-state 64 --n-layers 6 \
    --batch-size 64 --eval-batch-size 256 \
    --n-entities 16 --n-distractors 3 --max-hops 3 \
    --lr 3e-3 \
    --methods full,ema_topk,random_lowrank,activation_pca_lowrank,activation_sketch_lowrank \
    --ranks 8,16,32 --seeds 0,1,2,3,4 \
    --output results/lora_backward_sweep.json \
    --log-every 500 2>&1 | tee -a /tmp/chain_step1.log

# ---- 1b. PCA refresh-rate ablation ----
# Main sweep used basis_refresh_every=10 (the default). At rank=32 we got
# acc=0.196 vs ema_topk's 0.652. Test whether refresh frequency closes the gap.
echo "=== [1b] PCA refresh ablation @ $(date -Iseconds) ==="
for REFRESH in 1 50; do
    echo "--- basis_refresh_every=$REFRESH ---"
    $PY examples/sweep_lora_backward.py \
        --steps 1500 \
        --d-model 128 --d-state 64 --n-layers 6 \
        --batch-size 64 --eval-batch-size 256 \
        --n-entities 16 --n-distractors 3 --max-hops 3 \
        --lr 3e-3 \
        --methods activation_pca_lowrank --ranks 8,16,32 --seeds 0,1,2,3,4 \
        --basis-refresh-every $REFRESH \
        --output results/pca_refresh${REFRESH}.json \
        --log-every 500 2>&1 | tee -a /tmp/chain_step1b.log
done

# ---- 2. Lr-sensitivity on full backward ----
echo "=== [2] lr sensitivity (full backward) @ $(date -Iseconds) ==="
for LR in 1e-3 5e-4; do
    echo "--- lr=$LR ---"
    $PY examples/sweep_lora_backward.py \
        --steps 1500 \
        --d-model 128 --d-state 64 --n-layers 6 \
        --batch-size 64 --eval-batch-size 256 \
        --n-entities 16 --n-distractors 3 --max-hops 3 \
        --lr $LR \
        --methods full --ranks 32 --seeds 0,1,2,3,4 \
        --output results/lr_sensitivity_full_lr${LR}.json \
        --log-every 500 2>&1 | tee -a /tmp/chain_step2.log
done

# ---- 3. Large-batch wall-clock bench ----
echo "=== [3] large-batch wall-clock bench @ $(date -Iseconds) ==="
for B in 1024 2048; do
    echo "--- batch=$B ---"
    $PY examples/bench_wallclock_lora_backward.py \
        --d-hidden 4096 --n-layers 6 --d-in 1024 --d-out 1024 \
        --batch-size $B \
        --warmup-steps 5 --bench-steps 15 \
        --methods full,ema_topk --ranks 64 \
        --basis-refresh-every 200 \
        --optimizer adamw_fused \
        --output results/wallclock_big_batch_b${B}.json 2>&1 | tee -a /tmp/chain_step3.log
done

echo "=== ALL DONE @ $(date -Iseconds) ==="
touch results/ALL_DONE
