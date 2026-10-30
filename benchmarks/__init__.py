"""
Benchmarks package for custom SLM inference engine.
"""

from benchmarks.latency import benchmark_latency
from benchmarks.throughput import benchmark_throughput
from benchmarks.memory import benchmark_memory

__all__ = ["benchmark_latency", "benchmark_throughput", "benchmark_memory"]
