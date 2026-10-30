"""
Unit tests for INT8 and INT4 quantization.
"""

import pytest
import torch
import torch.nn as nn
from quantization.int8 import quantize_tensor_int8, dequantize_tensor_int8, QuantizedLinearInt8
from quantization.int4 import quantize_tensor_int4, dequantize_tensor_int4, QuantizedLinearInt4


def test_int8_quantization_roundtrip():
    w = torch.randn(64, 64)
    q, scale = quantize_tensor_int8(w)
    deq = dequantize_tensor_int8(q, scale)

    # Max relative error for 8-bit should be small (< 0.05)
    abs_err = torch.abs(w - deq)
    assert abs_err.mean() < 0.05


def test_quantized_linear_int8():
    lin = nn.Linear(32, 16)
    qlin = QuantizedLinearInt8.from_linear(lin)
    x = torch.randn(2, 32, dtype=torch.float16)
    out = qlin(x)
    assert out.shape == (2, 16)


def test_int4_quantization_roundtrip():
    w = torch.randn(32, 128)
    packed, scales, zeros = quantize_tensor_int4(w, group_size=64)
    deq = dequantize_tensor_int4(packed, scales, zeros, group_size=64)
    abs_err = torch.abs(w - deq)
    assert abs_err.mean() < 0.2


def test_quantized_linear_int4():
    lin = nn.Linear(128, 64)
    qlin = QuantizedLinearInt4.from_linear(lin, group_size=64)
    x = torch.randn(2, 128, dtype=torch.float16)
    out = qlin(x)
    assert out.shape == (2, 64)
