# Literature review

This project sits at the intersection of four literatures that are often discussed separately but are technically adjacent.

## 1. Local learning and alternatives to global error transport

Layer-local objectives are the closest precedent for the project's earliest experiments. Nøkland & Eidnes show that hidden layers can be trained from local losses rather than a single global loss, and Belilovsky et al. show that greedy layerwise learning can scale surprisingly far on ImageNet. Synthetic gradients attack a neighboring problem: instead of forcing a layer to wait for the true future gradient, Jaderberg et al. learn a local predictor for that signal. Hinton's Forward-Forward algorithm goes further and replaces the backward pass with separate positive and negative forward passes.

Relevant papers:

- Nøkland & Eidnes, *Training Neural Networks with Local Error Signals* (2019)
- Belilovsky, Eickenberg & Oyallon, *Greedy Layerwise Learning Can Scale to ImageNet* (2018)
- Jaderberg et al., *Decoupled Neural Interfaces using Synthetic Gradients* (2016)
- Hinton, *The Forward-Forward Algorithm: Some Preliminary Investigations* (2022)

## 2. Sparse depth connectivity

The power-of-two skip idea explored here is not a DenseNet clone so much as a sparse-aggregation idea: preserve short graph distances through depth without connecting every layer to every earlier layer. Zhu et al.'s sparsely aggregated networks are the closest architectural relative in the literature I found. Our distinct question is not merely whether sparse skips ease end-to-end training, but whether they help *bounded local credit assignment* recover some of the reach lost when full backpropagation is removed.

Relevant paper:

- Zhu et al., *Sparsely Aggregated Convolutional Networks* (2018)

## 3. Low-rank gradients and optimizer memory

GaLore is the cleanest match for the idea of projecting weight gradients into a low-rank subspace while keeping the underlying weights full-rank. PowerSGD studies a related low-rank compression problem for distributed communication rather than local optimizer memory. These methods establish that low-rank gradient structure can be practically useful, but they generally choose the subspace from the gradient itself or from a randomized sketch rather than from the forward computation.

Relevant papers:

- Zhao et al., *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection* (2024)
- Vogels et al., *PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization* (2019)

## 4. Compressed backward messages

The project's current main thread is slightly different from post-hoc gradient projection. For a linear layer,

```text
forward:  y  = x Wᵀ
backward: dx = dy W
update:   dW = dyᵀ x
```

If `dy` itself is compressed before it is propagated, then both the input-gradient computation and the weight update become cheaper. This is closer in spirit to low-rank or approximate feedback pathways than to merely compressing `dW` after dense backprop has already happened. Feedback-alignment work shows that exact backward transport is not always necessary; the present experiments ask a narrower systems question: whether a very cheap activation-chosen subset of backward channels can retain useful learning while reducing the arithmetic and state carried by the backward pass.

I did not find a canonical paper that is exactly "forward-pass-selected sparse backward channels for training ordinary neural nets." The closest neighbors are:

- low-rank gradient projection (GaLore),
- low-rank communication compression (PowerSGD),
- local or approximate feedback schemes,
- activation/statistics-aware second-order approximations such as K-FAC.

That absence is part of why the compressed-backward branch is worth testing carefully.

## Working synthesis

The project's current thesis is not that ordinary backpropagation is obsolete. It is narrower:

1. Some learning signals can be made more local than textbook backprop without immediately destroying useful representations.
2. Some backward information can be compressed substantially before catastrophic failure.
3. The important frontier is the shape of the tradeoff surface between:
   - backward information bandwidth,
   - memory footprint,
   - executed FLOPs,
   - sample efficiency,
   - task difficulty.

The recurrent FiLM/RAG benchmark is meant to make that tradeoff visible rather than letting easy tasks hide it.
