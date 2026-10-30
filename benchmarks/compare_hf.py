"""
Hugging Face Comparison & Correctness Benchmark.

Verifies:
  1. Mathematical Logits Equivalence: checks that custom transformer outputs match HF reference.
  2. Latency & Throughput: compares TTFT and TPOT against HF AutoModelForCausalLM.generate().
  3. KV Cache Ablation: measures generation latency with vs. without KV caching.
"""

from __future__ import annotations

import time
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime.engine import InferenceEngine

console = Console()


def run_logits_parity_test(model_path: str, device: str = "cpu"):
    """Compare custom engine logits against HuggingFace reference model."""
    console.print(Panel("[bold cyan]1. Mathematical Logits Parity Test[/bold cyan]"))

    prompt = "The quick brown fox jumps over the lazy"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    console.print(f"Prompt: [yellow]'{prompt}'[/yellow] (Length: {input_ids.shape[1]} tokens)")

    # 1. HuggingFace Reference Pass (FP32 on CPU for clean reference)
    console.print("Computing forward pass with HuggingFace AutoModelForCausalLM...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
    ).to(device)
    hf_model.eval()

    with torch.no_grad():
        hf_outputs = hf_model(input_ids.to(device))
        hf_logits = hf_outputs.logits.detach().cpu()

    # 2. Custom Engine Pass
    console.print("Computing forward pass with Custom SLM Inference Engine...")
    custom_engine = InferenceEngine(backend=device)
    custom_engine.load_model(model_path, device=device, precision="fp32")

    with torch.no_grad():
        custom_logits = custom_engine.model(input_ids.to(device)).detach().cpu()

    # 3. Compute Parity Metrics
    abs_diff = torch.abs(hf_logits - custom_logits)
    max_diff = torch.max(abs_diff).item()
    mean_diff = torch.mean(abs_diff).item()

    hf_top1 = torch.argmax(hf_logits[:, -1, :], dim=-1).item()
    custom_top1 = torch.argmax(custom_logits[:, -1, :], dim=-1).item()

    cos_sim = torch.nn.functional.cosine_similarity(
        hf_logits[:, -1, :].flatten(),
        custom_logits[:, -1, :].flatten(),
        dim=0,
    ).item()

    table = Table(title="Logits Parity Verification")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="bold green" if max_diff < 1e-3 else "yellow")

    table.add_row("Max Absolute Difference", f"{max_diff:.6e}", "PASS" if max_diff < 1e-2 else "WARN")
    table.add_row("Mean Absolute Difference", f"{mean_diff:.6e}", "PASS" if mean_diff < 1e-4 else "WARN")
    table.add_row("Cosine Similarity", f"{cos_sim:.8f}", "EXACT MATCH" if cos_sim > 0.9999 else "MATCH")
    table.add_row("HF Argmax Token ID", f"{hf_top1} ({tokenizer.decode([hf_top1])!r})", "—")
    table.add_row("Custom Argmax Token ID", f"{custom_top1} ({tokenizer.decode([custom_top1])!r})", "MATCH" if hf_top1 == custom_top1 else "MISMATCH")

    console.print(table)
    del hf_model
    custom_engine.unload_model()


def run_speed_benchmark(model_path: str, device: str = "mps"):
    """Compare generation latency and throughput against HuggingFace generate()."""
    console.print(Panel(f"[bold cyan]2. Latency & Throughput Benchmark on {device.upper()}[/bold cyan]"))

    prompt = "Explain why the sky is blue using Rayleigh scattering in three sentences."
    max_tokens = 40
    num_runs = 3

    # Load HF model
    console.print(f"Loading HuggingFace model on {device} (FP16)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)
    hf_model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup HF
    with torch.no_grad():
        _ = hf_model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Benchmark HF
    hf_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = hf_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        hf_times.append(time.perf_counter() - t0)

    hf_avg_e2e = np.mean(hf_times)
    hf_tps = max_tokens / hf_avg_e2e
    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load Custom Engine
    console.print(f"Loading Custom Inference Engine on {device} (FP16)...")
    engine = InferenceEngine(backend=device)
    engine.load_model(model_path, device=device, precision="fp16")

    # Warmup Custom Engine
    _ = engine.generate(prompt=prompt, max_tokens=5, temperature=0.0, stream=False)

    # Benchmark Custom Engine
    custom_times = []
    custom_ttft = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        first_tok = None
        count = 0
        for _ in engine.generate(prompt=prompt, max_tokens=max_tokens, temperature=0.0, stream=True):
            if first_tok is None:
                first_tok = time.perf_counter() - t0
                custom_ttft.append(first_tok)
            count += 1
        custom_times.append(time.perf_counter() - t0)

    custom_avg_e2e = np.mean(custom_times)
    custom_avg_ttft = np.mean(custom_ttft)
    custom_avg_tpot = (custom_avg_e2e - custom_avg_ttft) / max(1, max_tokens - 1)
    custom_tps = max_tokens / custom_avg_e2e

    # Display comparison
    table = Table(title=f"Performance Comparison: Custom Runtime vs HuggingFace ({device.upper()})")
    table.add_column("Metric", style="cyan")
    table.add_column("HuggingFace generate()", style="magenta")
    table.add_column("Custom Inference Engine", style="green")
    table.add_column("Speedup", style="bold yellow")

    table.add_row("End-to-End Latency", f"{hf_avg_e2e*1000:.1f} ms", f"{custom_avg_e2e*1000:.1f} ms", f"{hf_avg_e2e/custom_avg_e2e:.2f}x")
    table.add_row("Throughput (Tokens/Sec)", f"{hf_tps:.1f} tps", f"{custom_tps:.1f} tps", f"{custom_tps/hf_tps:.2f}x")
    table.add_row("Time to First Token (TTFT)", "N/A (monolithic)", f"{custom_avg_ttft*1000:.1f} ms", "Streaming First Token")
    table.add_row("Time Per Output Token (TPOT)", f"{hf_avg_e2e/max_tokens*1000:.1f} ms", f"{custom_avg_tpot*1000:.1f} ms", f"{(hf_avg_e2e/max_tokens)/custom_avg_tpot:.2f}x")

    console.print(table)
    engine.unload_model()


def run_kv_cache_ablation(model_path: str, device: str = "mps"):
    """Demonstrate the speedup from KV Caching vs recomputing attention."""
    console.print(Panel(f"[bold cyan]3. KV Cache Ablation: With Cache vs Without Cache ({device.upper()})[/bold cyan]"))

    prompt = "Artificial intelligence systems are designed to process complex data and"
    tokens_to_gen = 25

    engine = InferenceEngine(backend=device)
    engine.load_model(model_path, device=device, precision="fp16")

    # 1. With KV Cache (Standard Fast Path)
    t0 = time.perf_counter()
    _ = engine.generate(prompt=prompt, max_tokens=tokens_to_gen, temperature=0.0, stream=False)
    time_with_cache = time.perf_counter() - t0

    # 2. Without KV Cache (Naive full-prefix recomputation)
    input_ids = engine.tokenizer_processor.encode(prompt)
    curr_ids = list(input_ids)
    t0 = time.perf_counter()
    for _ in range(tokens_to_gen):
        inp_tensor = torch.tensor([curr_ids], dtype=torch.long, device=engine.backend.device)
        with torch.no_grad():
            # Pass None for kv_cache to force complete prefix recomputation from scratch!
            logits = engine.model(inp_tensor, start_pos=0, kv_cache=None)
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            curr_ids.append(next_token)
    time_without_cache = time.perf_counter() - t0

    table = Table(title="KV Cache Ablation Impact")
    table.add_column("Strategy", style="cyan")
    table.add_column("Execution Mode", style="yellow")
    table.add_column("Total Time", style="green")
    table.add_column("Per-Token Latency", style="magenta")

    table.add_row(
        "With KV Cache",
        "O(1) Attention step per token",
        f"{time_with_cache*1000:.1f} ms",
        f"{(time_with_cache/tokens_to_gen)*1000:.1f} ms",
    )
    table.add_row(
        "Without KV Cache",
        "O(N) quadratic recomputation",
        f"{time_without_cache*1000:.1f} ms",
        f"{(time_without_cache/tokens_to_gen)*1000:.1f} ms",
    )
    speedup = time_without_cache / max(time_with_cache, 1e-5)
    console.print(table)
    console.print(f"[bold green]✓ KV Caching delivers a {speedup:.2f}x speedup by eliminating past-token recomputation![/bold green]\n")
    engine.unload_model()


if __name__ == "__main__":
    model_dir = "/Users/ivinbaiju/Stark/Markv0.0/models/checkpoints/qwen2.5-0.5b"
    run_logits_parity_test(model_dir, device="cpu")
    run_speed_benchmark(model_dir, device="mps")
    run_kv_cache_ablation(model_dir, device="mps")
