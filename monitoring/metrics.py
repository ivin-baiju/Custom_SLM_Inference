"""
Prometheus and runtime metrics collection for custom SLM inference engine.
"""

from __future__ import annotations

import time
import threading
from typing import Dict, Any, List


class MetricsCollector:
    """Thread-safe singleton metrics collector."""

    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._requests_total = 0
        self._tokens_generated_total = 0
        self._total_prompt_tokens = 0
        self._total_generation_time_seconds = 0.0
        self._ttft_records: List[float] = []
        self._tpot_records: List[float] = []

    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = MetricsCollector()
            return cls._instance

    def record_request(
        self,
        prompt_tokens: int,
        generated_tokens: int,
        ttft: float,
        total_time: float,
    ) -> None:
        with self._lock:
            self._requests_total += 1
            self._total_prompt_tokens += prompt_tokens
            self._tokens_generated_total += generated_tokens
            self._total_generation_time_seconds += total_time
            self._ttft_records.append(ttft)
            if generated_tokens > 1 and total_time > ttft:
                tpot = (total_time - ttft) / (generated_tokens - 1)
                self._tpot_records.append(tpot)

            # Keep rolling window of last 200 requests
            if len(self._ttft_records) > 200:
                self._ttft_records = self._ttft_records[-200:]
            if len(self._tpot_records) > 200:
                self._tpot_records = self._tpot_records[-200:]

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            avg_ttft = sum(self._ttft_records) / max(1, len(self._ttft_records))
            avg_tpot = sum(self._tpot_records) / max(1, len(self._tpot_records))
            overall_tps = (
                self._tokens_generated_total / self._total_generation_time_seconds
                if self._total_generation_time_seconds > 0
                else 0.0
            )

            return {
                "requests_total": self._requests_total,
                "tokens_generated_total": self._tokens_generated_total,
                "prompt_tokens_total": self._total_prompt_tokens,
                "avg_ttft_ms": round(avg_ttft * 1000, 2),
                "avg_tpot_ms": round(avg_tpot * 1000, 2),
                "overall_tokens_per_second": round(overall_tps, 2),
            }

    def format_prometheus(self) -> str:
        """Export metrics in standard Prometheus exposition format."""
        s = self.get_summary()
        lines = [
            "# HELP slm_requests_total Total number of completed inference requests",
            "# TYPE slm_requests_total counter",
            f"slm_requests_total {s['requests_total']}",
            "",
            "# HELP slm_tokens_generated_total Total generated completion tokens",
            "# TYPE slm_tokens_generated_total counter",
            f"slm_tokens_generated_total {s['tokens_generated_total']}",
            "",
            "# HELP slm_avg_ttft_ms Average time to first token in milliseconds",
            "# TYPE slm_avg_ttft_ms gauge",
            f"slm_avg_ttft_ms {s['avg_ttft_ms']}",
            "",
            "# HELP slm_avg_tpot_ms Average time per output token in milliseconds",
            "# TYPE slm_avg_tpot_ms gauge",
            f"slm_avg_tpot_ms {s['avg_tpot_ms']}",
            "",
            "# HELP slm_tokens_per_second Throughput in tokens per second",
            "# TYPE slm_tokens_per_second gauge",
            f"slm_tokens_per_second {s['overall_tokens_per_second']}",
        ]
        return "\n".join(lines) + "\n"
