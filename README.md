# training-methods

A compact research workspace for experiments that weaken, reroute, or compress backpropagation.

The project currently revolves around one question:

> How much of backpropagation's cost is essential, and how much can be traded for locality, sparse routing, or compressed backward messages without losing useful learning?

## Current threads

### 1. Local learning and sparse credit flow

We compare ordinary end-to-end backpropagation with layer-local objectives and bounded sparse-credit graphs. The most interesting variant so far uses power-of-two depth parents so each local loss can update a logarithmic neighborhood instead of only the immediately previous layer.

### 2. Compressed backward signals

Instead of only compressing the final weight gradient, we now test compressed error messages themselves.

The most important branch is a **LoRA-shaped backward transport rule with full-weight optimization**. For a linear layer,

```text
forward:  y  = x Wᵀ

choose:   B ∈ R[out × r]      # one rank-r basis in output-channel space,
                              # chosen from forward activations

compress: z  = dy B
propagate:dx = z (Bᵀ W)
update:   dW = B (zᵀ x)
```

`B` is not a set of selected rows. Its columns are dense directions through the full output-channel space, so every output channel can participate through the low-rank basis. The gradient is low-rank for the current step, but it is lifted back into the full shape of `W`; all original weights remain trainable, and optimizers such as AdamW/BF16AdamW keep state for every original weight.

The repo also keeps a sparse `top-k` baseline, which chooses actual output rows and computes backward updates only through those rows. That baseline is cheaper, but it asks a different question: whether a small coordinate-aligned subset of channels is enough, rather than whether useful credit can travel through a dense low-rank subspace.

#### How this differs from GaLore

GaLore also keeps the original weights full-rank, but it is solving a different problem. GaLore forms the ordinary dense layer gradient first, projects that gradient into a low-rank space for optimizer updates, and stores optimizer state in the projected space to save memory. Its core target is optimizer-state memory reduction. citeturn0academia12

Our low-rank backward branch differs in four ways:

| Question | This repo | GaLore |
|---|---|---|
| **When is compression applied?** | Before layer backward work, by compressing `dy` | After the dense gradient has been formed |
| **What chooses the subspace?** | Forward activations (`activation_pca_lowrank` or sketches) | The gradient itself / its low-rank structure |
| **What becomes cheaper?** | The layer's backward transport arithmetic: both `dx` and `dW` are formed from rank-r messages | Optimizer memory; dense backward has already happened |
| **Where does optimizer state live?** | Full optimizer state for every original weight | Low-rank projected optimizer state |

So the cleanest one-line distinction is:

> **GaLore compresses gradients to save optimizer memory; our method compresses the backward message itself to test whether credit assignment can move through a low-rank activation-chosen subspace while preserving full-weight optimizer history.**

GaLore reports full-parameter learning with reduced optimizer-state memory; our current low-rank branch intentionally does **not** claim that memory saving, because preserving full per-weight optimizer state is part of the experiment. citeturn0academia12

### 3. Recurrent retrieval benchmark

The main benchmark is a synthetic multi-hop relational-reasoning task with:

- a gated recurrent stack,
- late→early FiLM feedback,
- retrieval-conditioned thought passes,
- controllable hop depth.

That benchmark is intentionally richer than a shallow classification toy: it stresses credit assignment over both depth and recurrent computation.

## Repository map

```text
src/training_methods/
  model.py                  recurrent FiLM/RAG model
  task.py                   synthetic relational benchmark
  local_depth.py            bounded local-credit experiments
  compressed_backward.py    indexed compressed-backward layers
  sparse_optim.py           row-sparse AdamW prototype
  quant_optim.py            blockwise int8 AdamW prototype
  activation_projection.py  activation-conditioned gradient projections

examples/
  compare_local_learning.py
  compare_sparse_gradient_flow.py
  compare_activation_projection.py
  compare_compressed_backward.py
  compare_recurrent_compressed_backward.py
  probe_representations.py

knowledge/
  non_backprop_training.md
  benchmark_direction.md
  literature_review.md
```

## What has happened so far

A few early signals are worth keeping in view:

- Local heads changed the internal organization of the recurrent model, but fair fresh-probe tests showed that ordinary backprop already made hidden states highly linearly readable.
- Bounded sparse-credit training kept much lower activation memory than full backprop, but naive power-of-two recomputation traded memory for substantial extra compute.
- Compressing the backward message itself is more promising than masking `dW` afterward. In a toy MLP, rank-16 backward channels used only `0.188×` the dense linear-backward FLOPs while remaining competitive on an easy 1-hop task.
- The compressed-backward mechanism now works inside the recurrent FiLM/RAG architecture. In the first single-seed 3-hop pass, compressed variants executed `0.305×` the dense linear-backward FLOPs in compressed modules and remained viable, but multi-seed replication is still required.

The detailed experimental log lives in [`knowledge/benchmark_direction.md`](knowledge/benchmark_direction.md).

## Running the project

The project uses `uv`.

```bash
uv run pytest -q
uv run python examples/compare_recurrent_compressed_backward.py
```

Most example scripts require Apple Silicon MPS by default so slow accidental CPU runs fail loudly rather than quietly consuming an afternoon.

## Near-term roadmap

1. Run 5-seed sweeps for recurrent compressed backward on the 3-hop benchmark.
2. Replace the Pythonic sparse optimizer with a more faithful sparse-state implementation if the learning signal survives replication.
3. Compare channel-selection rules beyond raw activation magnitude: sticky EMA, gradient-aware hybrids, and cheap sketches.
4. Keep separating three claims that are easy to conflate:
   - fewer executed backward FLOPs,
   - lower optimizer/activation memory,
   - equal or better sample efficiency.

## Reading

- [`knowledge/literature_review.md`](knowledge/literature_review.md)
- [`knowledge/non_backprop_training.md`](knowledge/non_backprop_training.md)
- [`knowledge/benchmark_direction.md`](knowledge/benchmark_direction.md)
