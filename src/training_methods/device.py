from __future__ import annotations

import torch


def pick_device(require_accelerator: bool = False) -> torch.device:
    """Prefer Apple Silicon acceleration via Metal Performance Shaders."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if require_accelerator:
        raise RuntimeError(
            "MPS is not available. Install a native macOS arm64 PyTorch build "
            "and verify torch.backends.mps.is_available() before training."
        )
    return torch.device("cpu")
