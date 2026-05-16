# Benchmark direction: retrieval-conditioned multi-hop reasoning

## Why this task

If the research goal is publishable local-learning results rather than a quick smoke test, simple image classification or one-hop key-value recall are too forgiving. A stronger benchmark should force the model to use all of the mechanisms we care about:

- persistent recurrent state
- sparse late-to-early FiLM feedback
- retrieval augmentation
- extra internal computation via a think gate
- credit assignment over both depth and time

## Proposed benchmark family

Use synthetic relational chains with retrieved distractor facts.

Example:

```text
query:      entity_3, hop_2
retrieved:  entity_3 -> entity_8
            entity_8 -> entity_1
            entity_7 -> entity_2
            entity_4 -> entity_9
answer:     entity_1
```

Difficulty scales naturally:

- 1-hop: retrieval / lookup
- 2-hop: composition
- 3-hop+: iterative reasoning and memory stress

## Why it is promising for local learning

This benchmark can expose whether local objectives help shallow layers acquire useful symbolic features earlier, while also measuring whether extra thought passes improve harder queries. It is richer than MNIST-style diagnostics but still controlled enough for mechanistic experiments.

## Publishable experimental arc

1. Controlled benchmark:
   - compare backprop, local heads, direct feedback alignment, hybrid warmup, and contract-based objectives
   - report accuracy by hop count, wall-clock, memory, and layerwise probes

2. Ablations:
   - remove FiLM feedback
   - remove retrieval
   - remove think passes
   - vary local-loss placement and weighting

3. Bridge task:
   - move from synthetic relational chains to a real retrieval benchmark once the mechanism is clear
   - candidate families: long-context QA / retrieval-intensive reasoning rather than generic classification

## Central claim to test

Local training signals may help most when the network must build reusable intermediate representations before the final task loss becomes informative. Multi-hop retrieval is a clean place to test that claim.

## First empirical pass — 2026-05-15

### Setup

- Model: 4-layer gated-delta recurrent stack, late→early FiLM broadcast, structured fact-vector RAG injection, 3 fixed thought passes
- Task: relational chains with 16 entities, 3 distractors, hop curriculum 1→2→3
- Hardware: M4 MacBook Pro via PyTorch MPS
- Comparison: ordinary backprop vs. deep supervision with layer-local classifier heads

### Early observations

A hard curriculum made the task learnable enough to expose behavior:

| Method | step 150 overall acc | step 450 overall acc | step 450 h1 | step 450 h2 | step 450 h3 |
|---|---:|---:|---:|---:|---:|
| Backprop | 0.336 | 0.148 | 0.13 | 0.15 | 0.16 |
| Local heads | 0.363 | 0.244 | 0.29 | 0.25 | 0.21 |

Interpretation: local supervision did not obviously help in the earliest trivial phase, but it retained more competence after the curriculum shifted toward harder reasoning. This is only a single-seed pilot, yet it suggests a more interesting hypothesis than "local heads learn faster": local objectives may stabilize reusable intermediate representations under changing task difficulty.

### Immediate next improvements

- Replace the hard 1→2→3 curriculum with a mixed curriculum so earlier skills do not disappear abruptly.
- Run at least 5 seeds before taking any difference seriously.
- Log per-layer probe accuracy and relative update norms; the mechanistic claim should be about representation formation and retention, not just endpoint accuracy.
- Add a no-FiLM and no-retrieval ablation so the task remains honest about which mechanism is buying what.

## Second empirical pass — mixed curriculum

### Setup

- Same architecture and task family as the first pass
- Mixed curriculum instead of hard phase switches: the training distribution gradually shifts toward harder hops while retaining easier examples
- Added layerwise readout diagnostics and relative layer-update norms

### Result

| Method | step 450 overall acc | h1 | h2 | h3 | local layer accuracies |
|---|---:|---:|---:|---:|---|
| Backprop | 0.742 | 0.76 | 0.68 | 0.78 | 0.03–0.04 |
| Local heads | 0.744 | 0.76 | 0.73 | 0.74 | 0.74–0.75 |

### Interpretation

The mixed curriculum removed the abrupt-forgetting artifact from the hard curriculum. Under this cleaner regime, deep supervision did **not** yet improve final accuracy over ordinary backprop, but it changed the internal organization of the model dramatically: every layer became directly predictive under local supervision, while ordinary backprop left intermediate layers nearly unreadable by the attached probes.

This reframes the strongest hypothesis:

> Local objectives may not chiefly improve endpoint accuracy on a stationary task; they may create distributed, reusable representations that become valuable under truncation, early exit, perturbation, or non-backprop training of lower layers.

### Best next tests

1. **Early exit / truncation:** evaluate accuracy if computation stops after layer 1, 2, or 3. Local learning should dominate if those layers truly contain useful representations.
2. **Layer freezing transfer:** after pretraining, freeze the first K layers and adapt the rest to a shifted task distribution. Distributed competence may transfer better.
3. **Hybrid local-to-global training:** train with local heads early, anneal their weight toward zero, and test whether we keep distributed representations without paying a final-loss tax.
4. **Non-backprop continuation:** pretrain locally, then continue with direct feedback alignment or a synthetic-gradient rule to see whether locally organized features make alternative credit assignment viable.

## Fair-probe correction

The first mixed-curriculum diagnostic overstated the difference between methods because the local-head model's intermediate classifiers were trained while the backprop model's attached heads remained random. A fairer test freezes both trained backbones and fits fresh identical linear probes from scratch on each layer.

### Fresh-probe result

| Method | final model accuracy | fresh probe L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|---:|
| Backprop | 0.712 | 0.723 | 0.719 | 0.722 | 0.723 |
| Local heads | 0.730 | 0.745 | 0.742 | 0.736 | 0.745 |

### Revised interpretation

The dramatic earlier contrast was mostly a measurement artifact: auxiliary heads trained with local supervision naturally become accurate, while untrained heads on a backprop model do not. Under fresh matched probes, ordinary backprop already produces highly linearly readable hidden states throughout the stack. Local supervision still shows a modest apparent gain, but this now needs multi-seed replication before any mechanistic claim is warranted.

### Methodological lesson

For representation claims, always distinguish:

1. **trained attached head accuracy** — what a head optimized during model training can read,
2. **fresh probe accuracy** — what information is linearly recoverable from a frozen representation,
3. **functional early-exit accuracy** — what the original model can do if computation is truncated.

Only the second and third speak directly to the quality of the underlying representation.

## Third empirical pass — bounded sparse-credit depth graphs

### Question

If full backpropagation is replaced with shallow local backward graphs, can sparse power-of-two depth edges recover some of the lost coordination while keeping memory low?

### Prototype

A simpler depth-only recurrent sequence model was added to isolate this question from the larger FiLM/RAG model. Three rules were compared on the same relational-reasoning task:

1. **Backprop:** ordinary end-to-end gradient through all layers.
2. **Edge-local chain:** each layer-local loss updates only the current layer and its immediate predecessor.
3. **Edge-local log-skip:** each layer-local loss updates the current layer plus direct parents at offsets 1, 2, 4, 8, ... .

The edge-local variants recompute only the bounded parent neighborhood needed for each local loss, so each backward graph stays shallow even though the full model is deep.

### First result — 8 layers, 180 steps, seed 0

| Method | final acc | peak allocated memory | wall-clock |
|---|---:|---:|---:|
| Backprop | 0.293 | 6.0 MB | 18.5 s |
| Edge-local chain | 0.297 | 2.7 MB | 38.5 s |
| Edge-local log-skip | 0.312 | 3.3 MB | 58.8 s |

### Interpretation

This first pass is encouraging but not yet decisive:

- Bounded local credit assignment preserved much more task competence than the stricter fully-detached local rule while using substantially less allocated memory than full backprop.
- Power-of-two parents gave a small apparent accuracy lift over the chain in this single seed, but they also cost extra recomputation time.
- The current implementation optimizes for conceptual cleanliness, not throughput; recomputing parent neighborhoods one local loss at a time is intentionally naive.

### Best next tests

1. Run at least 5 seeds before trusting the apparent log-skip edge.
2. Increase depth to 16–32 layers, where the graph diameter distinction should matter more.
3. Compare a single shared final objective with bounded sparse-credit routing against the current many-head local objective; the latter may make the task too forgiving.
4. Replace naive recomputation with a scheduled blockwise implementation before making any wall-clock claim.

## Fourth empirical pass — compressed backward channels and sparse optimizer

### Indexed backward

A custom linear layer now computes only selected output-channel rows during backpropagation. With rank 16 in the test MLP, the compressed layers execute `0.188×` the dense linear-backward FLOPs (about `5.3×` less arithmetic in those matmuls).

### Sparse optimizer

A prototype `RowSparseAdamW` stores first/second moments only for rows that have actually been active. On the 8-layer compressed-backward MLP at rank 16:

| Optimizer | step 300 optimizer-state elements |
|---|---:|
| Dense AdamW | 271,443 |
| Sparse AdamW, batch top-k | 159,854 |
| Sparse AdamW, EMA top-k | 117,254 |

The EMA version retained only about 43% of dense AdamW state by step 300. The current implementation is deliberately Pythonic and therefore much slower; it proves the storage model, not the throughput claim.

### Harder-task caveat

The first harder 3-hop MLP benchmark was not a useful judge because the dense baseline itself only reached about 10% accuracy after 300 steps. Future hardness tests should use the recurrent relational model, which has already shown meaningful learning on the same task family.

## Fifth empirical pass — recurrent FiLM/RAG with compressed backward

The compressed-backward linear layer was threaded through the recurrent FiLM/RAG architecture behind an opt-in model flag, so the comparison keeps recurrence, retrieval, and late→early FiLM feedback intact.

### First result — rank 16, 4 layers, 3 thought passes, seed 0

| Method | step 100 acc | step 200 acc | step 300 acc | backward FLOP ratio |
|---|---:|---:|---:|---:|
| Full backward | 0.254 | 0.246 | 0.164 | 1.000 |
| Batch top-k compressed backward | 0.156 | 0.180 | 0.203 | 0.305 |
| EMA top-k compressed backward | 0.152 | 0.176 | 0.203 | 0.305 |

### Interpretation

This single-seed pass is encouraging but not yet a claim:

- Compression survived the full recurrent + FiLM + retrieval stack without catastrophic failure.
- The compressed variants executed only about 30.5% of the dense linear-backward FLOPs in compressed modules.
- Dense training regressed late in this run, while compressed variants improved more steadily; that may indicate regularization, or simply seed noise.
- Multi-seed replication is now essential before interpreting the apparent late advantage.

## Sixth empirical pass — dense low-rank backward subspaces

### Question

Is the useful ingredient merely a rank-limited backward message, or specifically hard row sparsity?

### Prototype

`CompressedBackwardLinear` now also supports a dense output-channel basis `B ∈ R[out × r]`:

```text
z  = dy B
dx = z (Bᵀ W)
dW = B (zᵀ x)
```

This is intentionally distinct from GaLore-style post-hoc projection: the low-rank message is formed before `dx`, so it can reduce backward arithmetic, while the resulting parameter gradient is lifted back to dense `dW`.

### Theoretical accounting

For a linear layer with `T` token positions, input width `I`, output width `O`, and rank `r`:

| Method | backward FLOPs | ordinary AdamW state |
|---|---:|---:|
| Dense | `4 T O I` | `2 O I` |
| top-k rows | `4 T r I` | `2 r I` only with row-sparse optimizer |
| low-rank backward subspace | `2 T O r + 4 T r I + 4 O r I` | `2 O I` |

The low-rank path keeps dense optimizer state in the current implementation because it reconstructs dense `dW`; optimizer-memory savings would require a separate GaLore-like optimizer-state design in the subspace. Its FLOP benefit is also sequence-length dependent: projecting into and back out of the dense basis costs more than top-k row indexing, especially when `T` is small.

### First empirical comparison — rank 16, seed 0, 100 steps

| Method | step 100 acc | measured backward FLOP ratio | ordinary AdamW state ratio |
|---|---:|---:|---:|
| Full backward | 0.156 | 1.000 | 1.000 |
| Batch top-k | 0.172 | 0.305 | 1.000 in this run; `0.328` theoretically with row-sparse state |
| EMA top-k | 0.180 | 0.305 | 1.000 in this run; `0.328` theoretically with row-sparse state |
| Random low-rank basis | 0.180 | 0.441 | 1.000 |
| Activation-PCA low-rank basis | 0.219 | 0.441 | 1.000 |

Interpretation remains cautious because this is one seed and only 100 steps, but the first shape is encouraging:

- low-rank transport sits where expected computationally: denser and costlier than top-k, but still materially cheaper than full backward in the measured linear-backward arithmetic;
- a random dense subspace matched EMA top-k at step 100;
- activation-PCA low-rank was the strongest of this single-seed pass, though it was also much slower wall-clock because the current implementation performs eigendecompositions on CPU when running on MPS.

The next honest test is a 5-seed recurrent sweep before attributing meaning to the apparent PCA edge.

## Seventh empirical pass — int8 optimizer state

### Question

Can we shrink the persistent AdamW memory footprint without changing the model or backward rule?

### Prototype

`Int8AdamW` stores first and second moments as blockwise symmetric int8 tensors plus one floating-point scale per block. The implementation is intentionally clarity-first: it dequantizes states for the update, then requantizes them for storage between steps. This measures storage economics before any custom-kernel work.

### First smoke comparison — recurrent model, full backward, seed 0, 30 steps

| Optimizer | optimizer-state bytes | step 30 acc |
|---|---:|---:|
| AdamW | 722,640 | 0.078 |
| Int8AdamW prototype | 193,440 | 0.109 |

The int8 prototype used `0.268×` the optimizer-state bytes of dense AdamW in this run. These 30-step accuracies are only a smoke test; the current quantizer is simple symmetric blockwise storage, not yet a tuned optimizer-state quantizer. The immediate next test is to compare stability over longer runs, then improve the quantizer if needed before making any accuracy claim.

### Longer no-checkpoint comparison — recurrent model, full backward, seed 0, 300 steps

| Optimizer | optimizer-state bytes | step 100 acc | step 300 acc | behavior |
|---|---:|---:|---:|---|
| AdamW | 722,640 | 0.195 | 0.180 | stable |
| Int8AdamW prototype | 193,440 | 0.039 | 0.070 | diverged badly |

The first int8 design is therefore a useful negative result: it saves the intended memory, but the naïve symmetric quantizer does not preserve AdamW dynamics over a real run. By step 50, validation loss had already exploded from roughly `2.7` under AdamW to over `300` under the int8 prototype, and by step 300 it had reached the millions.

The next iteration should borrow more directly from established 8-bit optimizer designs rather than lowering precision indiscriminately:

1. use separate handling for first and second moments,
2. improve blockwise scaling / outlier behavior,
3. inspect dequantization error for `m` and especially `v`,
4. only after stability is recovered revisit more aggressive bit widths.

### Int8 revision — log-quantized second moment

Diagnostics on the failing prototype showed the key pathology: although average relative error looked small, about `20%` of the reconstructed second-moment entries had collapsed exactly to zero by step 50. That poisons AdamW's denominator.

The first repair keeps the signed first moment in blockwise int8 but stores the nonnegative second moment with blockwise log-space uint8 quantization. This removes the zero-collapse failure mode.

| Optimizer | optimizer-state bytes | step 100 acc | step 300 acc |
|---|---:|---:|---:|
| AdamW | 722,640 | 0.195 | 0.180 |
| BF16AdamW | 361,220 | 0.176 | — |
| Int8AdamW, naive `v` | 193,440 | 0.039 | 0.070 |
| Int8AdamW, log-quantized `v` | 194,928 | 0.125 | 0.219 |

The repaired int8 optimizer is now stable through 300 steps in the first seed while using `0.270×` AdamW's optimizer-state bytes. This is still a single-seed result, but it strongly supports the diagnosis that the original failure was variance quantization, not the idea of int8 state itself.

### Five-seed no-checkpoint sweep — recurrent model, full backward, 300 steps

| Optimizer | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean step-300 acc | optimizer-state bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | 0.156 | 0.230 | 0.184 | 0.160 | 0.230 | 0.192 | 722,640 |
| BF16AdamW | 0.168 | 0.242 | 0.191 | 0.207 | 0.215 | 0.205 | 361,220 |
| Int8AdamW, log-quantized `v` | 0.062 | 0.164 | 0.191 | 0.137 | 0.055 | 0.122 | 194,928 |

The multi-seed result changes the interpretation substantially:

- BF16 state looks like the strongest current memory tradeoff: roughly half the AdamW state bytes with no apparent loss in this sweep.
- The repaired int8 optimizer is numerically stable, but not yet learning-equivalent. Its mean accuracy is materially lower and its variance is much higher across seeds.
- The remaining problem is therefore subtler than catastrophic `v` collapse. The next int8 work should focus on optimizer fidelity, especially dynamic quantization for the signed first moment and richer error diagnostics over update directions rather than only tensor reconstruction error.

## Eighth empirical pass — BF16 state plus low-rank backward transport

With BF16 optimizer state looking like the practical memory baseline, the low-rank backward branch was re-centered around that optimizer rather than dense AdamW.

### Basis refresh variants

Two activation-conditioned dense bases are now available:

- `activation_pca_lowrank`: exact top principal directions, cached and refreshed periodically;
- `activation_sketch_lowrank`: randomized range-finder approximation, also cached and refreshed periodically.

Both use the same dense transport rule:

```text
z  = dy B
dx = z (Bᵀ W)
dW = B (zᵀ x)
```

### First BF16 comparison — rank 16, seed 0, 100 steps, basis refresh every 10 forwards

| Method | step 100 acc | measured backward FLOP ratio | optimizer-state bytes |
|---|---:|---:|---:|
| Full backward | 0.148 | 1.000 | 361,220 |
| EMA top-k | 0.156 | 0.305 | 361,220 |
| Random low-rank | 0.102 | 0.423 | 361,220 |
| Activation-PCA low-rank | 0.168 | 0.423 | 361,220 |
| Activation-sketch low-rank | 0.168 | 0.423 | 361,220 |

The first read is promising for the user's original idea: activation-conditioned dense low-rank transport again outperformed random low-rank and matched or exceeded the sparse top-k branch in this single seed, while BF16 state halves the optimizer-memory cost relative to AdamW.

The approximation did not yet improve wall-clock on MPS in this small prototype; the QR-heavy sketch path cost about the same as cached exact PCA here. That is an implementation result, not a conceptual defeat. The next fair test is multi-seed BF16 comparison of `full`, `ema_topk`, `activation_pca_lowrank`, and `activation_sketch_lowrank`.

### Five-seed BF16 sweep — rank 16, 300 steps, basis refresh every 10 forwards

| Method | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean step-300 acc | measured backward FLOP ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full backward | 0.086 | 0.223 | 0.184 | 0.199 | 0.199 | 0.178 | 1.000 |
| EMA top-k | 0.227 | 0.211 | 0.184 | 0.215 | 0.188 | 0.205 | 0.305 |
| Activation-PCA low-rank | 0.238 | 0.152 | 0.219 | 0.207 | 0.176 | 0.198 | 0.423 |
| Activation-sketch low-rank | 0.211 | 0.191 | 0.199 | 0.191 | 0.160 | 0.190 | 0.423 |

### Revised interpretation

The multi-seed result is more nuanced than the first seed:

- dense low-rank transport is viable and better than dense full backward on average in this sweep;
- however, EMA top-k is still the strongest current compressed method by mean accuracy, and it is cheaper in backward FLOPs;
- exact PCA low-rank remains close enough to be worth pursuing because it asks a different scientific question — dense rank-limited transport versus sparse coordinate transport — but it has not yet earned promotion over top-k;
- the randomized sketch approximation preserved much of the learning signal, but did not beat exact PCA and still carried poor wall-clock cost in the present implementation.

The next discriminating tests should vary rank and basis refresh cadence. If dense low-rank transport has a real advantage, it may emerge at tighter ranks where hard coordinate sparsity becomes too brittle, or with a more stable basis schedule than the current every-10-forward refresh.

## Ninth empirical pass — does EMA top-k starve the network?

### Question

The strong EMA top-k results raise a concern: is the method learning efficiently, or merely training a much smaller effective network while leaving most rows cold?

### Added diagnostics

For each compressed layer we now track:

- **coverage** — fraction of rows ever selected,
- **concentration** — Herfindahl-style concentration of cumulative selections,
- **turnover** — fraction of selected rows replaced since the previous selection.

### First recovery probe — rank 16, seed 0, 200 steps

| Method | step 200 acc | coverage | concentration | turnover |
|---|---:|---:|---:|---:|
| Full backward | 0.074 | 0.000 | 0.000 | 0.000 |
| EMA top-k | 0.074 | 0.741 | 0.049 | 0.015 |
| EMA top-k → full at step 101 | 0.109 | 0.707 | 0.050 | 0.011 |

### First interpretation

The starvation story is not as simple as “the same tiny subset trains forever.” In this run, EMA top-k had touched about `74%` of rows by step 200, while turnover stayed low but nonzero. That looks more like a slowly rotating specialist set than a permanently frozen subnetwork.

The delayed switch to full backward modestly improved step-200 accuracy over both always-full and always-EMA in this seed. That is only a first probe, but it suggests a potentially interesting curriculum interpretation: sparse backward early may help organize useful channels, while later dense training may broaden capacity.

The next honest test is multi-seed replication of the recovery experiment, plus cumulative **gradient/update mass** per row; selection counts alone tell us who was invited into the room, not who actually moved the furniture.

### Corrected five-seed recovery sweep — matched benchmark settings, rank 16, switch to full at step 151

| Method | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean step-300 acc |
|---|---:|---:|---:|---:|---:|---:|
| Full backward | 0.141 | 0.176 | 0.203 | 0.207 | 0.195 | 0.184 |
| EMA top-k | 0.160 | 0.195 | 0.215 | 0.195 | 0.215 | 0.196 |
| EMA top-k → full | 0.176 | 0.137 | 0.180 | 0.133 | 0.211 | 0.167 |
| Activation-PCA low-rank | 0.219 | 0.156 | 0.219 | 0.195 | 0.156 | 0.189 |
| Activation-sketch low-rank | 0.199 | 0.105 | 0.164 | 0.168 | 0.203 | 0.168 |

| Method | mean coverage | mean selection concentration | mean gradient concentration |
|---|---:|---:|---:|
| EMA top-k | 0.755 | 0.052 | 0.057 |
| EMA top-k → full | 0.729 | 0.053 | 0.026 |

### Revised interpretation

The first version of this sweep was invalid because the recovery script silently used the task defaults (`32` entities, `5` distractors) rather than the benchmark settings (`16`, `3`). The corrected sweep above matches the main recurrent comparison harness.

The corrected recovery sweep still does **not** support the optimistic sparse-then-dense curriculum story. Switching to full backward at step 151 substantially spreads gradient mass across rows, but it lowers mean accuracy relative to always-EMA and full backward.

The diagnostics do, however, answer the starvation question more precisely:

- EMA top-k is not training only a tiny fixed subset: roughly `76%` of rows were selected at least once by step 300.
- Yet update mass remains meaningfully concentrated: gradient concentration under always-EMA is more than twice that of the post-switch run.
- This suggests **partial specialization**, not total row starvation. The method is neither “just pruning itself” nor “fully using the width”; it is occupying a middle regime where many rows are visited but a smaller coalition receives most of the training signal.

The low-rank methods belong in the same picture:

- exact activation-PCA low-rank is competitive with full backward and close to EMA top-k, but not ahead on mean accuracy here;
- the sketch approximation lags in this sweep.

That makes the next mechanism question sharper: is EMA top-k winning because this specialization is useful, or because the task/model are effectively over-wide and do not need the dormant capacity?
