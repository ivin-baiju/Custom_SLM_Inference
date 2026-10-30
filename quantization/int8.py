"""
INT8 Weight Quantization module.

Provides symmetric 8-bit weight quantization, QuantizedLinearInt8 layer,
and in-place model quantization.
"""

from __future__ import annotations

import logging
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def quantize_tensor_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to symmetric int8.
    
    Formula:
        scale = max(|tensor|) / 127.0
        q = clip(round(tensor / scale), -127, 127).to(int8)
    
    Args:
        tensor: Floating point tensor (float32 or float16)
        
    Returns:
        (quantized_tensor, scale)
    """
    max_val = torch.max(torch.abs(tensor))
    scale = (max_val / 127.0).clamp(min=1e-8)
    q = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return q, scale.to(tensor.dtype)


def dequantize_tensor_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize an int8 tensor back to float."""
    return quantized.to(scale.dtype) * scale


class QuantizedLinearInt8(nn.Module):
    """Linear layer executing with INT8 weights and on-the-fly dequantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer("weight_scale", torch.ones((1,), dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinearInt8":
        """Convert a standard nn.Linear to QuantizedLinearInt8."""
        q_layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
        )
        with torch.no_grad():
            w_int8, scale = quantize_tensor_int8(linear.weight.data)
            q_layer.weight_int8.copy_(w_int8)
            q_layer.weight_scale.copy_(scale.to(torch.float16))
            if linear.bias is not None:
                q_layer.bias.copy_(linear.bias.data.to(torch.float16))
        return q_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight on the fly to match input dtype and device
        weight_float = self.weight_int8.to(dtype=x.dtype, device=x.device) * self.weight_scale.to(dtype=x.dtype, device=x.device)
        bias_float = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return F.linear(x, weight_float, bias_float)


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """Recursively replace all nn.Linear layers with QuantizedLinearInt8."""
    params = list(model.parameters())
    device = params[0].device if params else torch.device("cpu")
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            q_layer = QuantizedLinearInt8.from_linear(child).to(device)
            setattr(model, name, q_layer)
        else:
            quantize_model_int8(child)
    return model
