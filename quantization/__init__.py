"""
Quantization module for custom SLM inference engine.
"""

from quantization.int8 import (
    quantize_tensor_int8,
    dequantize_tensor_int8,
    QuantizedLinearInt8,
    quantize_model_int8,
)
from quantization.int4 import (
    quantize_tensor_int4,
    dequantize_tensor_int4,
    QuantizedLinearInt4,
    quantize_model_int4,
)

__all__ = [
    "quantize_tensor_int8",
    "dequantize_tensor_int8",
    "QuantizedLinearInt8",
    "quantize_model_int8",
    "quantize_tensor_int4",
    "dequantize_tensor_int4",
    "QuantizedLinearInt4",
    "quantize_model_int4",
]
