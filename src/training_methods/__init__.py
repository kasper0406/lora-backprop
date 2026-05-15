"""Experiments in alternative training methods and compact recurrent models."""

from .activation_projection import ActivationGradientProjector, ProjectionStats
from .compressed_backward import CompressedBackwardLinear, CompressedBackwardMLP, CompressedMLPConfig
from .device import pick_device
from .local_depth import LocalDepthConfig, LocalDepthSequenceModel
from .model import GatedDeltaFiLMRAGModel, ModelConfig
from .sparse_optim import RowSparseAdamW
from .task import RelationalReasoningBatch, RelationalReasoningTask

__all__ = [
    "ActivationGradientProjector",
    "CompressedBackwardLinear",
    "CompressedBackwardMLP",
    "CompressedMLPConfig",
    "GatedDeltaFiLMRAGModel",
    "LocalDepthConfig",
    "LocalDepthSequenceModel",
    "RelationalReasoningBatch",
    "RelationalReasoningTask",
    "ModelConfig",
    "ProjectionStats",
    "RowSparseAdamW",
    "pick_device",
]
