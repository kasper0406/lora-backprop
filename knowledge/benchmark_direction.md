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
