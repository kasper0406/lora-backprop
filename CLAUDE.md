# CLAUDE.md

This project studies training methods that trade global backpropagation for locality, sparse credit flow, or compressed backward messages.

Before changing code:

1. Read `README.md`.
2. Read `knowledge/literature_review.md`.
3. Read the latest sections of `knowledge/benchmark_direction.md`.

Research hygiene:

- Do not claim improvement from one seed.
- Separate executed FLOPs from wall-clock speed and optimizer-state memory.
- Prefer opt-in config flags over replacing existing baselines.
- Keep experiments reproducible and scripts small enough to audit.
- If a benchmark fails to train under the dense baseline, it is not a useful judge of an alternative method.

The most active line of work is compressed backward signaling inside the recurrent FiLM/RAG architecture.
