"""
Inference Scheduler coordinating request queue, continuous batching, and model execution.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, Optional

from runtime.engine import InferenceEngine
from runtime.request import InferenceRequest, InferenceResponse, RequestStatus
from scheduler.request_queue import RequestQueue
from scheduler.batching import ContinuousBatcher

logger = logging.getLogger(__name__)


class InferenceScheduler:
    """Async inference scheduler orchestrating request queues and batch execution."""

    def __init__(
        self,
        engine: InferenceEngine,
        max_batch_size: int = 8,
        queue_size: int = 1000,
    ):
        self.engine = engine
        self.queue = RequestQueue(maxsize=queue_size)
        self.batcher = ContinuousBatcher(max_batch_size=max_batch_size)
        self._running = False
        self._futures: Dict[str, asyncio.Future] = {}

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    async def submit(self, request: InferenceRequest, priority: int = 0) -> InferenceResponse:
        """Submit an inference request and await response."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._futures[request.request_id] = future

        enqueued = self.queue.enqueue(request, priority=priority)
        if not enqueued:
            request.status = RequestStatus.ERROR
            raise RuntimeError("Request queue is full")

        # Process in thread executor to avoid blocking the event loop
        return await future

    def step(self) -> None:
        """Execute one scheduling tick."""
        # 1. Admit new requests from queue if batcher has capacity
        while self.batcher.can_admit_request():
            req = self.queue.dequeue()
            if not req:
                break
            if not self.engine.tokenizer_processor:
                continue

            prompt_text = req.messages[0].content if req.messages else ""
            prompt_ids = self.engine.tokenizer_processor.encode(prompt_text)
            self.batcher.add_request(req, prompt_ids)

        # 2. Process prefill slots
        for slot in self.batcher.get_prefill_slots():
            try:
                # Run prefill using engine
                text_out = self.engine.generate(
                    prompt=slot.request.messages[0].content if slot.request.messages else "",
                    max_tokens=slot.request.generation_config.max_tokens or 128,
                    temperature=slot.request.generation_config.temperature,
                    top_k=slot.request.generation_config.top_k,
                    top_p=slot.request.generation_config.top_p,
                    stream=False,
                )
                slot.is_finished = True
                slot.request.status = RequestStatus.COMPLETE

                # Complete future
                future = self._futures.pop(slot.request.request_id, None)
                if future and not future.done():
                    resp = InferenceResponse(
                        request_id=slot.request.request_id,
                        model=self.engine.lifecycle.current_model or "",
                        text=text_out,
                        finish_reason="stop",
                    )
                    future.set_result(resp)
            except Exception as e:
                future = self._futures.pop(slot.request.request_id, None)
                if future and not future.done():
                    future.set_exception(e)

        # 3. Clean finished
        self.batcher.remove_finished()

    def get_stats(self) -> Dict[str, Any]:
        """Scheduler queue and batch status."""
        return {
            "queue_depth": self.queue.size,
            "active_batch_size": self.batcher.current_batch_size,
            "max_batch_size": self.batcher.max_batch_size,
        }
