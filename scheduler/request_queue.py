"""
Thread-safe Priority Request Queue for multi-user inference serving.
"""

from __future__ import annotations

import heapq
import threading
import time
from typing import Optional, List, Dict, Any

from runtime.request import InferenceRequest, RequestStatus


class RequestQueue:
    """Thread-safe priority queue with FIFO ordering for identical priority."""

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._heap: List[Tuple[int, float, InferenceRequest]] = []
        self._requests_by_id: Dict[str, InferenceRequest] = {}
        self._lock = threading.Lock()
        self._counter = 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._heap)

    def enqueue(self, request: InferenceRequest, priority: int = 0) -> bool:
        """Add request to queue with priority (lower number = higher priority)."""
        with self._lock:
            if len(self._heap) >= self.maxsize:
                return False
            self._counter += 1
            entry = (priority, time.time(), request)
            heapq.heappush(self._heap, entry)
            self._requests_by_id[request.request_id] = request
            request.status = RequestStatus.QUEUED
            return True

    def dequeue(self) -> Optional[InferenceRequest]:
        """Pop the highest-priority request."""
        with self._lock:
            while self._heap:
                _, _, request = heapq.heappop(self._heap)
                if request.status == RequestStatus.CANCELLED:
                    self._requests_by_id.pop(request.request_id, None)
                    continue
                self._requests_by_id.pop(request.request_id, None)
                return request
            return None

    def cancel(self, request_id: str) -> bool:
        """Cancel a pending request."""
        with self._lock:
            req = self._requests_by_id.get(request_id)
            if req and req.status == RequestStatus.QUEUED:
                req.status = RequestStatus.CANCELLED
                return True
            return False

    def get_pending(self) -> List[Dict[str, Any]]:
        """List current queued requests."""
        with self._lock:
            return [
                {
                    "request_id": req.request_id,
                    "model": req.model,
                    "status": req.status.value,
                    "created_at": req.created_at,
                }
                for _, _, req in self._heap
                if req.status != RequestStatus.CANCELLED
            ]
