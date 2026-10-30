"""
Model architectures package.

Importing this package automatically registers all supported architectures in ARCHITECTURE_REGISTRY.
"""

from models.architectures.transformer import (
    BaseTransformerModel,
    TransformerBlock,
    Attention,
    FeedForward,
    RMSNorm,
    RotaryEmbedding,
)
from models.architectures.llama import LlamaModel
from models.architectures.qwen import Qwen2Model
from models.architectures.phi import Phi3Model
from models.architectures.gemma import Gemma2Model

__all__ = [
    "BaseTransformerModel",
    "TransformerBlock",
    "Attention",
    "FeedForward",
    "RMSNorm",
    "RotaryEmbedding",
    "LlamaModel",
    "Qwen2Model",
    "Phi3Model",
    "Gemma2Model",
]
