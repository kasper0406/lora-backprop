"""Experiments in alternative training methods and compact recurrent models."""

from .activation_projection import ActivationGradientProjector, ProjectionStats
from .compressed_backward import (
    BackwardCostEstimate,
    CompressedBackwardLinear,
    CompressedBackwardMLP,
    CompressedMLPConfig,
    estimate_linear_backward_cost,
)
from .device import pick_device
from .local_depth import LocalDepthConfig, LocalDepthSequenceModel
from .model import GatedDeltaFiLMRAGModel, ModelConfig
from .quant_optim import BF16AdamW, Int8AdamW
from .sparse_optim import RowSparseAdamW
from .task import RelationalReasoningBatch, RelationalReasoningTask

__all__ = [
    "ActivationGradientProjector",
    "BackwardCostEstimate",
    "BF16AdamW",
    "CompressedBackwardLinear",
    "CompressedBackwardMLP",
    "CompressedMLPConfig",
    "GatedDeltaFiLMRAGModel",
    "Int8AdamW",
    "LocalDepthConfig",
    "LocalDepthSequenceModel",
    "RelationalReasoningBatch",
    "RelationalReasoningTask",
    "ModelConfig",
    "ProjectionStats",
    "RowSparseAdamW",
    "estimate_linear_backward_cost",
    "pick_device",
]
