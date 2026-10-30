"""
TokenProcessor providing text encoding, decoding, special token management,
and chat template application.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class TokenProcessor:
    """Handles all tokenization and text processing operations."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Convert string to list of token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert list of token IDs back to human-readable string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_token(self, token_id: int) -> str:
        """Convert single token ID to text string."""
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format a list of conversation messages into a prompt string using chat template."""
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception as e:
                logger.warning("Error using tokenizer chat template (%s); falling back to ChatML", e)

        # Fallback to standard ChatML format
        # <|im_start|>role\ncontent<|im_end|>\n
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"
        return formatted

    def get_special_tokens(self) -> Dict[str, Any]:
        """Return dictionary of special token IDs."""
        return {
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
