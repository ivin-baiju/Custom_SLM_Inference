"""
Inference Profiler for detailed latency and memory breakdown.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestTrace:
    request_id: str
    tokenization_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_steps_ms: List[float] = field(default_factory=list)
    detokenization_ms: float = 0.0
    total_ms: float = 0.0
    tokens_generated: int = 0
    prompt_tokens: int = 0


class InferenceProfiler:
    """Measures precise time and memory for each phase of LLM execution."""

    def __init__(self):
        self.traces: Dict[str, RequestTrace] = {}

    def start_trace(self, request_id: str) -> RequestTrace:
        trace = RequestTrace(request_id=request_id)
        self.traces[request_id] = trace
        return trace

    def format_trace(self, request_id: str) -> str:
        trace = self.traces.get(request_id)
        if not trace:
            return f"Trace {request_id} not found"

        avg_decode_ms = (
            sum(trace.decode_steps_ms) / len(trace.decode_steps_ms)
            if trace.decode_steps_ms
            else 0.0
        )
        return (
            f"=== Request #{trace.request_id} Trace ===\n"
            f"  Prompt tokens:     {trace.prompt_tokens}\n"
            f"  Generated tokens:  {trace.tokens_generated}\n"
            f"  Tokenization:      {trace.tokenization_ms:.2f} ms\n"
            f"  Prefill (TTFT):    {trace.prefill_ms:.2f} ms\n"
            f"  Avg Decode Step:   {avg_decode_ms:.2f} ms (TPOT)\n"
            f"  Total Duration:    {trace.total_ms:.2f} ms\n"
        )
