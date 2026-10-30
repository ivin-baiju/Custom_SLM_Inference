"""
Latency benchmarking module for measuring TTFT (Time to First Token) and TPOT (Time Per Output Token).
"""

from __future__ import annotations

import time
import statistics
from typing import List, Dict, Any
from runtime.engine import InferenceEngine


def benchmark_latency(
    engine: InferenceEngine,
    prompts: List[str],
    max_tokens: int = 50,
) -> Dict[str, Any]:
    """Measure TTFT, TPOT, and end-to-end latency."""
    ttft_list: List[float] = []
    tpot_list: List[float] = []
    e2e_list: List[float] = []

    for prompt in prompts:
        t0 = time.perf_counter()
        first_token_time: float = 0.0
        generated_count = 0

        for chunk in engine.generate(prompt=prompt, max_tokens=max_tokens, stream=True):
            if generated_count == 0:
                first_token_time = time.perf_counter() - t0
                ttft_list.append(first_token_time)
            generated_count += 1

        total_time = time.perf_counter() - t0
        e2e_list.append(total_time)

        if generated_count > 1:
            tpot = (total_time - first_token_time) / (generated_count - 1)
            tpot_list.append(tpot)

    return {
        "num_samples": len(prompts),
        "mean_ttft_ms": round(statistics.mean(ttft_list) * 1000, 2) if ttft_list else 0.0,
        "median_ttft_ms": round(statistics.median(ttft_list) * 1000, 2) if ttft_list else 0.0,
        "mean_tpot_ms": round(statistics.mean(tpot_list) * 1000, 2) if tpot_list else 0.0,
        "median_tpot_ms": round(statistics.median(tpot_list) * 1000, 2) if tpot_list else 0.0,
        "mean_e2e_ms": round(statistics.mean(e2e_list) * 1000, 2) if e2e_list else 0.0,
    }
