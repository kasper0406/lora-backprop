from __future__ import annotations

import torch


def pick_device(require_accelerator: bool = False) -> torch.device:
    """Prefer CUDA, then Apple Silicon MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if require_accelerator:
        raise RuntimeError(
            "No GPU accelerator available. Install a CUDA-enabled PyTorch on Linux "
            "or a native macOS arm64 PyTorch with MPS before training."
        )
    return torch.device("cpu")
