#!/bin/bash
# Follow-up: at 1500 steps the annealed Gumbel-top-k matched the deterministic
# baseline within 1 pstd and the training curves were still rising. This run
# doubles the budget to 3000 steps and adds a deterministic control at the
# same budget — the missing piece from the previous run, since without it we
# can't tell whether stochastic is catching up to a fixed ceiling or whether
# deterministic itself keeps improving.
set -eux
cd "$HOME/lora-backprop"
PY=.venv/bin/python
mkdir -p results/stochastic_3k

COMMON_ARGS=(
    --steps 3000
    --d-model 128 --d-state 64 --n-layers 6
    --batch-size 64 --eval-batch-size 256
    --n-entities 16 --n-distractors 3 --max-hops 3
    --lr 1e-3
    --methods ema_topk --ranks 32 --seeds 0,1,2,3,4
    --log-every 500
)

# A. Deterministic ema_topk @ 3000 steps (the control we should have run).
echo "=== det_3000 @ $(date -Iseconds) ==="
$PY examples/sweep_lora_backward.py "${COMMON_ARGS[@]}" \
    --output results/stochastic_3k/det_3000.json 2>&1 | tee -a /tmp/stochastic_3k.log

# B. Anneal 0.5 -> 20 over the full 3000 steps. Same target eta as the 1500-step
# best config, but the schedule now occupies twice as many steps so the
# diffuse phase actually has time to find diverse channels before the sharp
# exploit phase kicks in.
echo "=== anneal_0.5_to_20_3000 @ $(date -Iseconds) ==="
$PY examples/sweep_lora_backward.py "${COMMON_ARGS[@]}" \
    --eta-start 0.5 --eta-end 20.0 \
    --output results/stochastic_3k/anneal_0.5_to_20_3000.json 2>&1 | tee -a /tmp/stochastic_3k.log

# C. Anneal 0.5 -> 50. Validator's suggestion: a sharper final eta gives the
# exploit phase more time at the deterministic limit, which may close the
# residual gap if there is one.
echo "=== anneal_0.5_to_50_3000 @ $(date -Iseconds) ==="
$PY examples/sweep_lora_backward.py "${COMMON_ARGS[@]}" \
    --eta-start 0.5 --eta-end 50.0 \
    --output results/stochastic_3k/anneal_0.5_to_50_3000.json 2>&1 | tee -a /tmp/stochastic_3k.log

echo "=== STOCHASTIC 3K DONE @ $(date -Iseconds) ==="
touch results/STOCHASTIC_3K_DONE
