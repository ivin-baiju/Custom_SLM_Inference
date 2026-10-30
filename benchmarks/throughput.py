"""
Throughput benchmarking module for measuring generation speed (tokens/sec).
"""

from __future__ import annotations

import time
from typing import List, Dict, Any
from runtime.engine import InferenceEngine


def benchmark_throughput(
    engine: InferenceEngine,
    prompts: List[str],
    max_tokens: int = 100,
) -> Dict[str, Any]:
    """Measure generation throughput in tokens per second."""
    total_tokens = 0
    t0 = time.perf_counter()

    for prompt in prompts:
        tokens_in_prompt = 0
        for _ in engine.generate(prompt=prompt, max_tokens=max_tokens, stream=True):
            tokens_in_prompt += 1
        total_tokens += tokens_in_prompt

    total_duration = time.perf_counter() - t0
    tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0.0

    return {
        "total_tokens_generated": total_tokens,
        "total_duration_seconds": round(total_duration, 3),
        "tokens_per_second": round(tokens_per_second, 2),
    }
