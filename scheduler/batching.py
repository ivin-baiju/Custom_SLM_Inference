"""
Batching strategies for concurrent inference execution.

Provides StaticBatcher and ContinuousBatcher (iteration-level batching).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import torch

from runtime.request import InferenceRequest, RequestStatus

logger = logging.getLogger(__name__)


@dataclass
class BatchSlot:
    """Represents an active request slot in continuous batching."""
    request: InferenceRequest
    prompt_ids: List[int]
    generated_ids: List[int] = field(default_factory=list)
    current_pos: int = 0
    is_prefill_done: bool = False
    is_finished: bool = False


class ContinuousBatcher:
    """Manages dynamic batching where requests join and leave between decoding iterations."""

    def __init__(self, max_batch_size: int = 8, max_context_len: int = 2048):
        self.max_batch_size = max_batch_size
        self.max_context_len = max_context_len
        self.active_slots: Dict[str, BatchSlot] = {}

    @property
    def current_batch_size(self) -> int:
        return len(self.active_slots)

    def can_admit_request(self) -> bool:
        """Check if batcher has capacity for another request."""
        return len(self.active_slots) < self.max_batch_size

    def add_request(self, request: InferenceRequest, prompt_ids: List[int]) -> bool:
        """Admit a new request into active batch."""
        if not self.can_admit_request():
            return False

        slot = BatchSlot(
            request=request,
            prompt_ids=prompt_ids,
            current_pos=len(prompt_ids),
        )
        self.active_slots[request.request_id] = slot
        request.status = RequestStatus.PREFILLING
        return True

    def get_prefill_slots(self) -> List[BatchSlot]:
        """Get slots that still need prefill."""
        return [s for s in self.active_slots.values() if not s.is_prefill_done]

    def get_decode_slots(self) -> List[BatchSlot]:
        """Get slots ready for decode step."""
        return [s for s in self.active_slots.values() if s.is_prefill_done and not s.is_finished]

    def mark_prefill_done(self, request_id: str, first_token: int) -> None:
        slot = self.active_slots.get(request_id)
        if slot:
            slot.is_prefill_done = True
            slot.generated_ids.append(first_token)
            slot.request.status = RequestStatus.DECODING

    def mark_token_generated(self, request_id: str, token_id: int) -> None:
        slot = self.active_slots.get(request_id)
        if slot:
            slot.generated_ids.append(token_id)
            slot.current_pos += 1

    def remove_finished(self) -> List[BatchSlot]:
        """Remove and return finished sequences from the batch."""
        finished = []
        for req_id, slot in list(self.active_slots.items()):
            max_tokens = slot.request.generation_config.max_tokens or 256
            if slot.is_finished or len(slot.generated_ids) >= max_tokens:
                slot.is_finished = True
                slot.request.status = RequestStatus.COMPLETE
                finished.append(slot)
                del self.active_slots[req_id]
        return finished
