# Non-backpropagation training ideas

## Motivation

Standard neural-network training uses backpropagation plus an optimizer such as Adam or Muon. In some settings, later layers appear to learn much faster than early layers. That raises two related questions:

1. Can we add contracts or alignment objectives between layers so early layers receive stronger learning signals?
2. Can we replace or weaken backpropagation in exchange for faster wall-clock training, even if we lose some theoretical guarantees or final accuracy?

The practical target here is experimentation on an Apple Silicon laptop, where memory traffic and activation storage may matter as much as raw FLOPs.

## What biology suggests

The brain almost certainly does not implement textbook backpropagation exactly. Standard backprop has several biologically implausible aspects:

- It appears to require exact transpose weights in the feedback path.
- It assumes a clean forward pass followed by a backward pass.
- Synapses would need non-local information about distant errors.
- Neural communication is spike-based rather than dense signed tensors.

What is well supported experimentally is more local learning:

- Hebbian plasticity and spike-timing-dependent plasticity (STDP)
- Neuromodulator-gated learning, especially dopamine-like reward prediction errors
- Rich dendritic computation and feedback pathways that may support local approximations to credit assignment

The live scientific question is less "does the brain run GPU-style backprop?" and more "does biological learning approximate gradient-like credit assignment through local mechanisms?"

## Literature map

### 1. Local learning / deep supervision

Each block receives its own auxiliary loss or classifier head rather than relying only on the final task loss.

Why it matters:

- Gives shallow layers a direct training signal
- Reduces dependence on long backward chains
- Can reduce activation storage and become attractive on memory-constrained hardware

Representative ideas:

- Deep supervision / auxiliary classifiers
- Greedy or block-local learning
- Modern variants such as AugLocal and auxiliary-network methods that reduce the historical accuracy gap with backprop

Practical judgment:

This currently looks like the most promising family if the goal is actual speed on a laptop rather than biological elegance alone.

### 2. Feedback alignment and direct feedback alignment

Instead of using exact transposed forward weights during the backward pass, send errors through fixed random feedback matrices.

Why it matters:

- Avoids the weight-transport problem
- Simplifies the backward signal
- Direct feedback alignment can send an output error directly to hidden layers

Practical judgment:

Cheap and worth testing. Accuracy and scaling are less dependable than backprop, but it is one of the cleanest alternatives to prototype.

### 3. Synthetic gradients / decoupled neural interfaces

Train a small model to predict the gradient a layer would later receive, so the layer can update before the rest of the network finishes.

Why it matters:

- Breaks backward locking
- Potentially useful for pipelining or asynchronous training

Practical judgment:

Interesting where decoupling itself is the bottleneck. Less obviously useful for a single laptop unless the synthetic-gradient predictor is extremely cheap.

### 4. Target propagation

Propagate target activations rather than gradients. Each layer learns to move its output toward a target representation.

Why it matters:

- Conceptually close to a "contract between layers"
- More local than ordinary backprop

Practical judgment:

Very relevant intellectually, but implementation complexity is higher and empirical wins are less obvious than for local losses.

### 5. Predictive coding / equilibrium propagation

Use local prediction errors or energy minimization dynamics instead of a standard backward pass.

Why it matters:

- Strong connection to neuroscience
- Naturally framed as local consistency constraints between adjacent layers

Practical judgment:

Beautiful, but many versions require iterative settling or extra phases, which can erase any speed advantage on conventional hardware.

### 6. Forward-only methods

Examples include Hinton's Forward-Forward algorithm, forward-gradient approaches, and zeroth-order methods.

Why it matters:

- Avoid or reduce backward-mode autodiff
- Can reduce memory requirements
- Useful when gradients are unavailable or activation storage is the real bottleneck

Practical judgment:

Research-worthy, but not automatically faster. Extra forward passes or noisy estimators can make them slower in wall-clock terms even when they look cheaper on paper.

## Current ranking for experiments

1. Backprop baseline with strong diagnostics
2. Local supervised losses / AugLocal-lite
3. Direct feedback alignment
4. Synthetic gradients
5. Forward-Forward
6. Forward-gradient or zeroth-order hybrids
7. Predictive coding / equilibrium propagation as slower but conceptually rich controls

The key distinction is:

> Non-backpropagation is not the same thing as faster training.

A method only wins if its replacement learning signal is cheaper than the backward computation it removes, once memory traffic, extra passes, and convergence speed are counted.

## Original ideas worth testing

### Adjacent-layer agreement contract

Each block learns from the task loss plus a local objective encouraging its output to predict the next block's representation under stop-gradient.

Possible variants:

- Mean-squared prediction of the next representation
- Contrastive agreement objective
- Whitening / decorrelation regularizers to prevent trivial collapse

### Depth-weighted local supervision

Use stronger auxiliary losses for early layers, then anneal them away over training so the network can later coordinate globally.

### Warm-start with local or DFA signals, then switch to backprop

Use a cheap local rule to wake up early layers, then hand control to ordinary backprop once useful features exist.

### Hybrid trunk/head training

Train the early trunk with a local or forward-only rule while keeping backprop for the last few layers. This may capture much of the speed gain while preserving enough global coordination to remain accurate.

## Microbenchmark suite

### Benchmark A: Deep MLP stress test

- Dataset: MNIST or Fashion-MNIST
- Model: 16-32 layer MLP, intentionally without residuals
- Purpose: expose early-layer starvation cleanly

### Benchmark B: Small CNN

- Dataset: CIFAR-10
- Purpose: test whether methods survive useful vision structure

### Benchmark C: Small residual CNN

- Dataset: CIFAR-10
- Purpose: determine whether the method still helps once the architecture already mitigates vanishing gradients

### Benchmark D: Optional sequence task

- Example: tiny character-level language model or copy task
- Purpose: especially useful for synthetic gradients and locking-related ideas

## Metrics to track

Always compare methods by wall-clock, not just step count.

- Time to 90% of baseline accuracy
- Final validation accuracy
- Samples per second
- Peak unified memory
- Relative weight-update norm per layer
- Gradient norm per layer where applicable
- Linear-probe accuracy by layer
- Accuracy-vs-wall-clock curves

Useful derived probes:

- Time-to-feature: how long until each layer reaches a chosen linear-probe accuracy
- Frozen-tail test: freeze upper layers and measure how quickly lower layers can support a fresh head
- Local-vs-global gradient cosine similarity where both signals exist

## Initial practical recommendation

For the first real experiment, build one harness with identical models and compare:

1. Backprop + Adam baseline
2. Backprop + auxiliary local heads
3. Direct feedback alignment
4. Forward-Forward
5. Hybrid local-warmup then backprop

The likely early winner is not the most exotic method. It is probably a local-learning hybrid that preserves enough global coordination to remain useful while lowering activation-memory pressure.

## Starting references

- Lillicrap et al. — *Random synaptic feedback weights support error backpropagation for deep learning* / feedback alignment
- Nøkland & Eidnes — *Training Neural Networks with Local Error Signals*
- Jaderberg et al. — *Decoupled Neural Interfaces using Synthetic Gradients*
- Lee et al. / Bengio group — target propagation and difference target propagation
- Scellier & Bengio — equilibrium propagation
- Hinton — *The Forward-Forward Algorithm*
- Lillicrap et al. — *Backpropagation and the Brain*
- Recent local-learning work such as AugLocal and auxiliary-network methods

## Open questions

- Is the observed slow learning in early layers actually vanishing gradients, poor conditioning, optimizer behavior, or representation collapse?
- On Apple Silicon, do these methods reduce unified-memory pressure enough to beat backprop in wall-clock time?
- Which contracts help optimization without forcing adjacent layers into overly rigid representations?
- Are hybrid methods the sweet spot: local enough to be fast, global enough to stay competent?
