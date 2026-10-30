"""
Server module for custom SLM inference engine.
"""

from server.api import app, global_engine
from server.endpoints import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelLoadRequest,
    SwitchModelRequest,
)

__all__ = [
    "app",
    "global_engine",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "ModelLoadRequest",
    "SwitchModelRequest",
]
