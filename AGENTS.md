# AGENTS.md

## Mission

This repository is a research sandbox for alternative training methods. Preserve experimental clarity over cleverness: comparisons should isolate one changed variable at a time.

## Working rules

- Keep dense backprop baselines healthy before adding exotic variants.
- Distinguish claims about FLOPs, memory, wall-clock, and accuracy; never let one stand in for another.
- Prefer small legible prototypes first, then optimize only ideas that survive contact with a fair benchmark.
- When adding a new method, add:
  1. a config flag or separate module,
  2. a comparison script,
  3. a smoke test,
  4. a short note in `knowledge/benchmark_direction.md` if it produces an interpretable result.
- Treat single-seed wins as hypotheses, not findings.
- Preserve the Apple Silicon/MPS workflow unless there is a strong reason not to.

## Current priorities

1. Multi-seed recurrent compressed-backward experiments.
2. More faithful sparse optimizer/state implementations if compressed backward remains competitive.
3. Better channel-selection rules than raw activation magnitude.

## Important files

- `src/training_methods/model.py` — recurrent FiLM/RAG architecture
- `src/training_methods/compressed_backward.py` — compressed backward channels
- `src/training_methods/sparse_optim.py` — sparse optimizer prototype
- `knowledge/benchmark_direction.md` — living experimental log
- `knowledge/literature_review.md` — conceptual map
