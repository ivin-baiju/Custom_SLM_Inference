"""
INT4 Group-wise Weight Quantization module.

Packs pairs of 4-bit unsigned integers into uint8 bytes with per-group scales and zero points.
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def quantize_tensor_int4(
    tensor: torch.Tensor, group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize float tensor to 4-bit integers with group-wise scaling.
    
    Args:
        tensor: Shape (out_features, in_features)
        group_size: Number of elements per quantization group
        
    Returns:
        (packed_uint8, scales, zeros)
    """
    orig_shape = tensor.shape
    out_features, in_features = orig_shape
    assert in_features % group_size == 0, f"in_features ({in_features}) must be divisible by group_size ({group_size})"

    # Reshape into groups: (out_features * (in_features // group_size), group_size)
    grouped = tensor.reshape(-1, group_size)
    min_val = grouped.min(dim=-1, keepdim=True).values
    max_val = grouped.max(dim=-1, keepdim=True).values

    scales = ((max_val - min_val) / 15.0).clamp(min=1e-8)
    zeros = min_val

    # Quantize to 0..15
    q = torch.clamp(torch.round((grouped - zeros) / scales), 0, 15).to(torch.uint8)

    # Pack 2 values per byte: low 4 bits, high 4 bits
    q = q.reshape(out_features, in_features)
    # in_features must be even
    q_low = q[:, 0::2]
    q_high = q[:, 1::2]
    packed = q_low | (q_high << 4)

    scales = scales.reshape(out_features, in_features // group_size)
    zeros = zeros.reshape(out_features, in_features // group_size)
    return packed, scales.to(tensor.dtype), zeros.to(tensor.dtype)


def dequantize_tensor_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Unpack and dequantize 4-bit packed weights."""
    out_features, half_in = packed.shape
    in_features = half_in * 2

    # Unpack low and high 4-bit nibbles
    q_low = (packed & 0x0F).to(scales.dtype)
    q_high = ((packed >> 4) & 0x0F).to(scales.dtype)

    # Interleave back to (out_features, in_features)
    unpacked = torch.empty((out_features, in_features), dtype=scales.dtype, device=packed.device)
    unpacked[:, 0::2] = q_low
    unpacked[:, 1::2] = q_high

    # Reshape to groups and apply scale and zeros
    num_groups = in_features // group_size
    unpacked_grouped = unpacked.reshape(out_features, num_groups, group_size)
    scales_grouped = scales.unsqueeze(-1)
    zeros_grouped = zeros.unsqueeze(-1)

    dequantized = (unpacked_grouped * scales_grouped) + zeros_grouped
    return dequantized.reshape(out_features, in_features)


class QuantizedLinearInt4(nn.Module):
    """Linear layer executing with INT4 packed weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        num_groups = in_features // group_size
        self.register_buffer(
            "weight_packed",
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8),
        )
        self.register_buffer(
            "weight_scales",
            torch.ones((out_features, num_groups), dtype=torch.float16),
        )
        self.register_buffer(
            "weight_zeros",
            torch.zeros((out_features, num_groups), dtype=torch.float16),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128) -> "QuantizedLinearInt4":
        # Ensure padding if needed
        in_feat = linear.in_features
        if in_feat % group_size != 0:
            pad = group_size - (in_feat % group_size)
            w = F.pad(linear.weight.data, (0, pad))
            in_feat = in_feat + pad
        else:
            w = linear.weight.data

        q_layer = cls(in_feat, linear.out_features, group_size=group_size, bias=linear.bias is not None)
        with torch.no_grad():
            packed, scales, zeros = quantize_tensor_int4(w, group_size=group_size)
            q_layer.weight_packed.copy_(packed)
            q_layer.weight_scales.copy_(scales.to(torch.float16))
            q_layer.weight_zeros.copy_(zeros.to(torch.float16))
            if linear.bias is not None:
                q_layer.bias.copy_(linear.bias.data.to(torch.float16))
        return q_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_float = dequantize_tensor_int4(
            self.weight_packed.to(x.device),
            self.weight_scales.to(dtype=x.dtype, device=x.device),
            self.weight_zeros.to(dtype=x.dtype, device=x.device),
            self.group_size,
        )
        if w_float.shape[1] > x.shape[-1]:
            w_float = w_float[:, : x.shape[-1]]
        bias_float = self.bias.to(dtype=x.dtype, device=x.device) if self.bias is not None else None
        return F.linear(x, w_float, bias_float)


def quantize_model_int4(model: nn.Module, group_size: int = 128) -> nn.Module:
    """Recursively replace nn.Linear layers with QuantizedLinearInt4."""
    params = list(model.parameters())
    device = params[0].device if params else torch.device("cpu")
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            q_layer = QuantizedLinearInt4.from_linear(child, group_size=group_size).to(device)
            setattr(model, name, q_layer)
        else:
            quantize_model_int4(child, group_size=group_size)
    return model
