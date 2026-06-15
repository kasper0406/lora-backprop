#!/bin/bash
# Stochastic top-k follow-up: does Gumbel-top-k sampling close the gap between
# tuned ema_topk (0.684) and tuned full backward (0.896)?
#
# Fixed: lr=1e-3, rank=32, 5 seeds, 1500 steps (matches the tuned head-to-head).
# Variable: eta schedule (constant or annealed).
# Baseline: deterministic ema_topk @ lr=1e-3 r=32 from results/tuned/ema_topk_lr1e-3.json
set -eux
cd "$HOME/lora-backprop"
PY=.venv/bin/python
mkdir -p results/stochastic

COMMON_ARGS=(
    --steps 1500
    --d-model 128 --d-state 64 --n-layers 6
    --batch-size 64 --eval-batch-size 256
    --n-entities 16 --n-distractors 3 --max-hops 3
    --lr 1e-3
    --methods ema_topk --ranks 32 --seeds 0,1,2,3,4
    --log-every 500
)

# 1. Constant eta = 1.0  (substantial exploration throughout)
echo "=== eta_const_1.0 @ $(date -Iseconds) ==="
$PY examples/sweep_lora_backward.py "${COMMON_ARGS[@]}" \
    --eta-start 1.0 \
    --output results/stochastic/eta_const_1.0.json 2>&1 | tee -a /tmp/stochastic.log

# 2. Constant eta = 3.0  (modest exploration)
echo "=== eta_const_3.0 @ $(date -Iseconds) ==="
$PY examples/sweep_lora_backward.py "${COMMON_ARGS[@]}" \
    --eta-start 3.0 \
    --output results/stochastic/eta_const_3.0.json 2>&1 | tee -a /tmp/stochastic.log

# 3. Anneal 1 -> 10  (literal lightning schedule)
echo "=== eta_anneal_1_to_10 @ $(date -Iseconds) ==="
$PY examples/sweep_lora_backward.py "${COMMON_ARGS[@]}" \
    --eta-start 1.0 --eta-end 10.0 \
    --output results/stochastic/eta_anneal_1_to_10.json 2>&1 | tee -a /tmp/stochastic.log

# 4. Anneal 0.5 -> 20  (more aggressive: nearly uniform early, very sharp late)
echo "=== eta_anneal_0.5_to_20 @ $(date -Iseconds) ==="
$PY examples/sweep_lora_backward.py "${COMMON_ARGS[@]}" \
    --eta-start 0.5 --eta-end 20.0 \
    --output results/stochastic/eta_anneal_0.5_to_20.json 2>&1 | tee -a /tmp/stochastic.log

echo "=== STOCHASTIC DONE @ $(date -Iseconds) ==="
touch results/STOCHASTIC_DONE
