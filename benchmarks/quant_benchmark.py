"""
Quantization Benchmark: FP16 vs INT8 vs INT4.

Measures:
  1. Weight memory footprint on disk / in memory.
  2. Quantization error / reconstruction fidelity (MSE, cosine similarity).
  3. Inference speed (TTFT, TPOT, Tokens/Sec).
  4. Generation coherence comparison across precisions.
"""

from __future__ import annotations

import time
import torch
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from runtime.engine import InferenceEngine
from quantization.int8 import quantize_tensor_int8, dequantize_tensor_int8
from quantization.int4 import quantize_tensor_int4, dequantize_tensor_int4

console = Console()


def benchmark_layer_quantization(model_path: str):
    """Analyze tensor-level quantization error and memory compression."""
    console.print(Panel("[bold cyan]1. Tensor-Level Quantization Error & Compression[/bold cyan]"))

    from safetensors import safe_open
    weights_path = Path(model_path) / "model.safetensors"

    with safe_open(str(weights_path), framework="pt") as f:
        # Pick representative projection weights
        sample_keys = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ]

        table = Table(title="Layer-Level Quantization Fidelity (FP16 vs INT8 vs INT4)")
        table.add_column("Weight Tensor", style="cyan")
        table.add_column("Shape", style="dim")
        table.add_column("FP16 Size", style="magenta")
        table.add_column("INT8 Size (MSE)", style="green")
        table.add_column("INT4 Size (MSE)", style="yellow")

        for key in sample_keys:
            w = f.get_tensor(key).to(torch.float32)
            orig_bytes = w.nelement() * 2  # FP16 bytes

            # INT8 Quantization
            q8, s8 = quantize_tensor_int8(w)
            deq8 = dequantize_tensor_int8(q8, s8)
            mse8 = torch.mean((w - deq8) ** 2).item()
            int8_bytes = q8.nelement() + s8.nelement() * 2

            # INT4 Quantization (group size 64)
            q4, s4, z4 = quantize_tensor_int4(w, group_size=64)
            deq4 = dequantize_tensor_int4(q4, s4, z4, group_size=64)
            mse4 = torch.mean((w - deq4) ** 2).item()
            int4_bytes = q4.nelement() + (s4.nelement() + z4.nelement()) * 2

            table.add_row(
                key.split(".")[-2] + "." + key.split(".")[-1],
                str(list(w.shape)),
                f"{orig_bytes / 1024:.1f} KB",
                f"{int8_bytes / 1024:.1f} KB (MSE: {mse8:.2e})",
                f"{int4_bytes / 1024:.1f} KB (MSE: {mse4:.2e})",
            )

        console.print(table)


def benchmark_end_to_end_quantization(model_path: str, device: str = "mps"):
    """Benchmark end-to-end inference and text quality across FP16, INT8, and INT4."""
    console.print(Panel(f"[bold cyan]2. End-to-End Inference Comparison on {device.upper()}[/bold cyan]"))

    prompt = "Explain why gravity causes objects to fall towards the Earth in two sentences."
    max_tokens = 35

    precisions = [
        ("FP16 (Baseline)", None),
        ("INT8 Quantized", "int8"),
        ("INT4 Quantized", "int4"),
    ]

    results_table = Table(title=f"Precision Trade-off: Latency, Memory & Output Quality ({device.upper()})")
    results_table.add_column("Precision", style="cyan")
    results_table.add_column("Weight Memory", style="magenta")
    results_table.add_column("TTFT", style="yellow")
    results_table.add_column("TPOT", style="green")
    results_table.add_column("Tokens/Sec", style="blue")
    results_table.add_column("Generated Sample", style="white", max_width=45)

    for label, quant_mode in precisions:
        console.print(f"Testing [bold]{label}[/]...")
        engine = InferenceEngine(backend=device)
        engine.load_model(
            model_path=model_path,
            device=device,
            precision="fp16",
            quantization=quant_mode,
        )

        # Warmup
        _ = engine.generate(prompt=prompt, max_tokens=5, temperature=0.0, stream=False)

        # Benchmark
        t0 = time.perf_counter()
        first_token_time = None
        tokens = []
        for chunk in engine.generate(prompt=prompt, max_tokens=max_tokens, temperature=0.0, stream=True):
            if first_token_time is None:
                first_token_time = time.perf_counter() - t0
            tokens.append(chunk)
        total_time = time.perf_counter() - t0

        token_count = len(tokens)
        ttft_ms = (first_token_time or 0) * 1000
        tpot_ms = ((total_time - (first_token_time or 0)) / max(1, token_count - 1)) * 1000
        tps = token_count / total_time if total_time > 0 else 0

        # Memory estimation
        mem_multiplier = 1.0 if quant_mode is None else (0.52 if quant_mode == "int8" else 0.28)
        approx_mem = 942 * mem_multiplier

        sample_text = "".join(tokens).replace("\n", " ").strip()
        if len(sample_text) > 40:
            sample_text = sample_text[:37] + "..."

        results_table.add_row(
            label,
            f"~{approx_mem:.0f} MB ({mem_multiplier*100:.0f}%)",
            f"{ttft_ms:.1f} ms",
            f"{tpot_ms:.1f} ms",
            f"{tps:.1f} tps",
            sample_text,
        )

        engine.unload_model()

    console.print(results_table)


if __name__ == "__main__":
    model_dir = "/Users/ivinbaiju/Stark/Markv0.0/models/checkpoints/qwen2.5-0.5b"
    benchmark_layer_quantization(model_dir)
    benchmark_end_to_end_quantization(model_dir, device="mps")
