"""Typed request / response containers and generation configuration.

Every public surface that touches user input or model output flows through the
data-classes defined here so that the rest of the engine can stay decoupled
from serialisation details.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Request status lifecycle
# ---------------------------------------------------------------------------

class RequestStatus(enum.Enum):
    """Status of an individual inference request as it moves through the pipeline."""

    QUEUED = "queued"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Generation configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """All tuneable knobs for the autoregressive generation loop.

    Parameters
    ----------
    max_tokens:
        Hard cap on the number of tokens to generate.  ``0`` means the model's
        own ``max_position_embeddings`` is used as the ceiling.
    temperature:
        Softmax temperature.  ``0.0`` forces greedy (argmax) decoding.
    top_k:
        Retain only the *k* highest-probability tokens before sampling.
        ``0`` disables top-k filtering.
    top_p:
        Nucleus sampling — keep the smallest set whose cumulative probability
        exceeds *p*.  ``1.0`` disables nucleus filtering.
    repetition_penalty:
        Multiplicative penalty applied to tokens that already appear in the
        context.  ``1.0`` means no penalty.
    stop_sequences:
        Generation halts when any of these strings is produced.
    seed:
        RNG seed for reproducibility.  ``None`` means non-deterministic.
    """

    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    stop_sequences: tuple[str, ...] = ()
    seed: int | None = None


# ---------------------------------------------------------------------------
# Chat message
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single turn in a multi-turn conversation.

    Parameters
    ----------
    role:
        One of ``"system"``, ``"user"``, or ``"assistant"``.
    content:
        The textual content of the turn.
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Serialise to a plain dict (compatible with HF chat templates)."""
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# Inference request
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class InferenceRequest:
    """Encapsulates everything needed for a single inference call.

    Parameters
    ----------
    messages:
        Chat-style message list (``[{role, content}, ...]``).  Converted to a
        prompt string by the tokenizer's chat template.
    model:
        Model identifier or path.  The engine resolves this to a concrete
        model directory.
    temperature:
        Sampling temperature override (folds into ``gen_config``).
    max_tokens:
        Maximum tokens to generate.
    top_k:
        Top-k sampling cutoff.
    top_p:
        Nucleus sampling threshold.
    stop_sequences:
        Additional stop strings beyond EOS.
    stream:
        If ``True`` the engine yields :class:`StreamChunk` objects instead of
        blocking until completion.
    gen_config:
        Full generation config — if supplied, scalar overrides above are
        ignored.
    """

    messages: list[ChatMessage]
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 512
    top_k: int = 50
    top_p: float = 0.9
    stop_sequences: tuple[str, ...] = ()
    stream: bool = False
    gen_config: GenerationConfig | None = None

    # ── internal bookkeeping ──────────────────────────────────────────
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: RequestStatus = RequestStatus.QUEUED
    created_at: float = field(default_factory=time.monotonic)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolve_gen_config(self) -> GenerationConfig:
        """Return the effective :class:`GenerationConfig`.

        If an explicit ``gen_config`` was set, return it directly; otherwise
        build one from the per-field overrides on this request.
        """
        if self.gen_config is not None:
            return self.gen_config
        return GenerationConfig(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            stop_sequences=self.stop_sequences,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InferenceRequest:
        """Build a request from a raw JSON-like dict (e.g. from the API layer).

        Expects ``messages`` as a list of ``{role, content}`` dicts.
        """
        messages = [
            ChatMessage(role=m["role"], content=m["content"])
            for m in data.get("messages", [])
        ]
        return cls(
            messages=messages,
            model=data.get("model", ""),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 512),
            top_k=data.get("top_k", 50),
            top_p=data.get("top_p", 0.9),
            stop_sequences=tuple(data.get("stop_sequences", ())),
            stream=data.get("stream", False),
        )


# ---------------------------------------------------------------------------
# Inference response
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class InferenceResponse:
    """Returned to the caller after generation completes (non-streaming path).

    Parameters
    ----------
    request_id:
        Echoed from the originating :class:`InferenceRequest`.
    text:
        The generated text (assistant turn only, no prompt echo).
    prompt_tokens:
        Number of tokens in the prompt after tokenization.
    completion_tokens:
        Number of tokens actually generated.
    total_tokens:
        ``prompt_tokens + completion_tokens``.
    time_to_first_token_ms:
        Wall-clock time from request receipt to the first generated token
        (prefill latency).
    tokens_per_second:
        Decode throughput measured as ``completion_tokens / decode_time``.
    total_time_ms:
        End-to-end wall-clock time for the entire generation.
    finish_reason:
        Why generation stopped — ``"stop"`` (EOS or stop sequence),
        ``"length"`` (hit ``max_tokens``), or ``"error"``.
    model:
        The model identifier that served this request.
    """

    request_id: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    time_to_first_token_ms: float = 0.0
    tokens_per_second: float = 0.0
    total_time_ms: float = 0.0
    finish_reason: str = "stop"
    model: str = ""


# ---------------------------------------------------------------------------
# Streaming chunk
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StreamChunk:
    """A single incremental piece emitted during streaming generation.

    Parameters
    ----------
    request_id:
        Ties the chunk back to its originating request.
    delta:
        The newly generated text fragment (may be a partial token for
        byte-level tokenizers).
    token_id:
        The raw token id that produced ``delta``.
    finish_reason:
        ``None`` while generation continues; one of ``"stop"`` /
        ``"length"`` / ``"error"`` on the final chunk.
    """

    request_id: str
    delta: str
    token_id: int | None = None
    finish_reason: str | None = None
