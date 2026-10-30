"""
Autoregressive generation loop with Prefill and Decode stages, KV Cache management,
and advanced sampling (greedy, temperature, top-k, top-p, repetition penalty).
"""

from __future__ import annotations

import time
import logging
from typing import Generator, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F

from models.architectures.transformer import BaseTransformerModel
from cache.kv_cache import KVCache
from runtime.request import GenerationConfig, StreamChunk

logger = logging.getLogger(__name__)


def sample_token(
    logits: torch.Tensor,
    gen_config: GenerationConfig,
    generated_tokens: Optional[List[int]] = None,
) -> int:
    """Sample next token from logits according to GenerationConfig.
    
    Supports:
        - Greedy decoding (temperature == 0.0)
        - Repetition penalty
        - Temperature scaling
        - Top-k filtering
        - Top-p (nucleus) filtering
        - Multinomial sampling
    """
    # logits shape: (1, vocab_size) or (vocab_size,)
    if logits.dim() == 2:
        logits = logits.squeeze(0)

    # 1. Repetition penalty
    if gen_config.repetition_penalty != 1.0 and generated_tokens:
        for prev_token in set(generated_tokens):
            if logits[prev_token] < 0:
                logits[prev_token] *= gen_config.repetition_penalty
            else:
                logits[prev_token] /= gen_config.repetition_penalty

    # 2. Greedy decoding
    if gen_config.temperature <= 1e-5:
        return int(torch.argmax(logits, dim=-1).item())

    # 3. Temperature scaling
    logits = logits / gen_config.temperature

    # 4. Top-K filtering
    if gen_config.top_k > 0:
        top_k = min(gen_config.top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        min_value = values[-1]
        logits = torch.where(logits < min_value, torch.tensor(float("-inf"), device=logits.device), logits)

    # 5. Top-P (nucleus) filtering
    if gen_config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > gen_config.top_p
        # Shift the indices to keep at least the first token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    # 6. Sample from distribution
    probs = F.softmax(logits, dim=-1)
    if gen_config.seed is not None:
        generator = torch.Generator(device=logits.device).manual_seed(gen_config.seed)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator).item()
    else:
        next_token = torch.multinomial(probs, num_samples=1).item()

    return int(next_token)


def prefill(
    input_ids: torch.Tensor,
    model: BaseTransformerModel,
    kv_cache: KVCache,
) -> Tuple[torch.Tensor, float]:
    """Execute Prefill stage: process entire prompt, build KV cache.
    
    Args:
        input_ids: Tensor of shape (1, seq_len)
        model: Transformer model
        kv_cache: Initialized KV cache
        
    Returns:
        (logits for last token, prefill_duration_seconds)
    """
    t0 = time.perf_counter()
    with torch.no_grad():
        # logits shape: (1, seq_len, vocab_size)
        logits = model(input_ids, start_pos=0, kv_cache=kv_cache)
        last_token_logits = logits[:, -1, :]  # (1, vocab_size)
    duration = time.perf_counter() - t0
    return last_token_logits, duration


def decode_step(
    token_id: int,
    position: int,
    model: BaseTransformerModel,
    kv_cache: KVCache,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Execute single Decode step: process one token using existing KV cache.
    
    Args:
        token_id: Token ID to forward
        position: Current position index in the sequence
        model: Transformer model
        kv_cache: KV cache holding past keys and values
        device: Device where model is located
        
    Returns:
        (logits for next token, step_duration_seconds)
    """
    t0 = time.perf_counter()
    token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(token_tensor, start_pos=position, kv_cache=kv_cache)
        next_logits = logits[:, 0, :]  # (1, vocab_size)
    duration = time.perf_counter() - t0
    return next_logits, duration


def generate_tokens(
    prompt_ids: List[int],
    model: BaseTransformerModel,
    kv_cache: KVCache,
    gen_config: GenerationConfig,
    tokenizer: Any,
    device: torch.device,
) -> Generator[Tuple[int, str, dict], None, None]:
    """Yields (token_id, token_text, metrics_dict) autoregressively.
    
    Yields per-step metrics including TTFT on first token and TPOT on subsequent tokens.
    """
    kv_cache.reset()

    # 1. Prefill stage
    input_ids_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    last_logits, prefill_time = prefill(input_ids_tensor, model, kv_cache)

    generated_ids: List[int] = []
    current_pos = len(prompt_ids)

    # First token selection (Time to First Token - TTFT)
    first_token = sample_token(last_logits, gen_config, generated_ids)
    generated_ids.append(first_token)
    first_text = tokenizer.decode([first_token])

    yield first_token, first_text, {
        "step": 0,
        "ttft": prefill_time,
        "step_time": prefill_time,
        "is_prefill": True,
    }

    # Stop check
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and first_token == eos_id:
        return

    # Check stop sequences
    if any(stop_seq in first_text for stop_seq in gen_config.stop_sequences):
        return

    # 2. Decode stage (token by token)
    max_new_tokens = gen_config.max_tokens if gen_config.max_tokens > 0 else 256
    current_token = first_token

    accumulated_text = first_text

    for step in range(1, max_new_tokens):
        next_logits, step_time = decode_step(
            current_token, current_pos, model, kv_cache, device
        )
        current_pos += 1

        next_token = sample_token(next_logits, gen_config, prompt_ids + generated_ids)
        generated_ids.append(next_token)

        # Stop conditions
        if eos_id is not None and next_token == eos_id:
            break

        token_text = tokenizer.decode([next_token])
        accumulated_text += token_text

        yield next_token, token_text, {
            "step": step,
            "step_time": step_time,
            "is_prefill": False,
        }

        # Check stop sequence
        if any(stop_seq in accumulated_text for stop_seq in gen_config.stop_sequences):
            break

        current_token = next_token
