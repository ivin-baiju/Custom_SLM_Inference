"""
Server-Sent Events (SSE) streaming generator for OpenAI-compatible client integration.
"""

from __future__ import annotations

import json
import logging
import uuid
import time
from typing import AsyncGenerator
from sse_starlette.sse import ServerSentEvent

from server.endpoints import ChatCompletionRequest, ChatCompletionChunk, StreamChoice, DeltaMessage
from runtime.engine import InferenceEngine

logger = logging.getLogger(__name__)


async def stream_chat_completion(
    request: ChatCompletionRequest,
    engine: InferenceEngine,
) -> AsyncGenerator[ServerSentEvent, None]:
    """Stream chat completions as Server-Sent Events."""
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())
    model_name = engine.lifecycle.current_model or "slm"

    # Send initial role chunk
    initial_chunk = ChatCompletionChunk(
        id=req_id,
        created=created_time,
        model=model_name,
        choices=[StreamChoice(index=0, delta=DeltaMessage(role="assistant"))],
    )
    yield ServerSentEvent(data=initial_chunk.model_dump_json())

    # Format prompt from messages
    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
    stop_seqs = [request.stop] if isinstance(request.stop, str) else (request.stop or [])

    if not engine.tokenizer_processor:
        raise RuntimeError("No tokenizer available in engine.")

    prompt = engine.tokenizer_processor.apply_chat_template(
        messages_dicts, add_generation_prompt=True
    )

    last_telemetry = None
    for _, delta_text, telemetry in engine.generate_with_telemetry(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        stop_sequences=stop_seqs,
        request_id=req_id.replace("chatcmpl-", ""),
    ):
        last_telemetry = telemetry
        chunk = ChatCompletionChunk(
            id=req_id,
            created=created_time,
            model=model_name,
            choices=[StreamChoice(index=0, delta=DeltaMessage(content=delta_text))],
            telemetry=telemetry,
        )
        yield ServerSentEvent(data=chunk.model_dump_json(exclude_none=True))

    # Send final finish chunk
    final_chunk = ChatCompletionChunk(
        id=req_id,
        created=created_time,
        model=model_name,
        choices=[StreamChoice(index=0, delta=DeltaMessage(), finish_reason="stop")],
        telemetry=last_telemetry,
    )
    yield ServerSentEvent(data=final_chunk.model_dump_json(exclude_none=True))
    yield ServerSentEvent(data="[DONE]")
