"""
Model Lifecycle Management for the custom SLM inference engine.

Manages state transitions, memory tracking, and safe model switching / unloading.
"""

from __future__ import annotations

import gc
import enum
import logging
import threading
from typing import Optional, Dict, Any
import torch

from backends.base import ExecutionBackend

logger = logging.getLogger(__name__)


class LifecycleState(enum.Enum):
    """Model lifecycle state machine."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    UNLOADING = "unloading"


class ModelLifecycle:
    """Thread-safe lifecycle controller managing model state and memory resources."""

    def __init__(self, backend: ExecutionBackend):
        self.backend = backend
        self._state = LifecycleState.UNLOADED
        self._active_requests = 0
        self._lock = threading.RLock()
        self._model_path: Optional[str] = None
        self._model_name: Optional[str] = None

    @property
    def state(self) -> LifecycleState:
        with self._lock:
            return self._state

    @property
    def active_requests(self) -> int:
        with self._lock:
            return self._active_requests

    @property
    def current_model(self) -> Optional[str]:
        return self._model_name

    def begin_loading(self, model_name: str, model_path: str) -> None:
        """Transition to LOADING state."""
        with self._lock:
            if self._state in (LifecycleState.LOADING, LifecycleState.BUSY):
                raise RuntimeError(f"Cannot load model while in state {self._state.value}")
            self._state = LifecycleState.LOADING
            self._model_name = model_name
            self._model_path = model_path
            logger.info("Lifecycle: Beginning loading of model '%s'", model_name)

    def set_ready(self) -> None:
        """Mark model as READY to accept inference requests."""
        with self._lock:
            self._state = LifecycleState.READY
            logger.info("Lifecycle: Model '%s' is now READY", self._model_name)

    def acquire_inference(self) -> None:
        """Increment active requests when starting inference."""
        with self._lock:
            if self._state not in (LifecycleState.READY, LifecycleState.BUSY):
                raise RuntimeError(f"Cannot run inference in state {self._state.value}")
            self._active_requests += 1
            self._state = LifecycleState.BUSY

    def release_inference(self) -> None:
        """Decrement active requests when completing inference."""
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            if self._active_requests == 0 and self._state == LifecycleState.BUSY:
                self._state = LifecycleState.READY

    def unload(self, target_model: Any = None) -> None:
        """Safely drain active requests, free model weights, and release backend memory."""
        with self._lock:
            self._state = LifecycleState.UNLOADING
            logger.info("Lifecycle: Unloading model '%s'...", self._model_name)

            # Wait briefly if there are active requests
            import time
            wait_deadline = time.time() + 5.0
            while self._active_requests > 0 and time.time() < wait_deadline:
                time.sleep(0.05)

            # Explicitly delete references
            del target_model
            gc.collect()

            # Empty backend device caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

            self._state = LifecycleState.UNLOADED
            self._model_name = None
            self._model_path = None
            self._active_requests = 0
            logger.info("Lifecycle: Model successfully unloaded and device memory reclaimed.")

    def get_status(self) -> Dict[str, Any]:
        """Summary of lifecycle status and memory."""
        with self._lock:
            mem_info = self.backend.memory_info()
            return {
                "state": self._state.value,
                "model_name": self._model_name,
                "model_path": self._model_path,
                "active_requests": self._active_requests,
                "memory": mem_info,
            }
