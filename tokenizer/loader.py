"""
Tokenizer loader for custom SLM inference engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def load_tokenizer(model_path: Union[str, Path]):
    """Load a HuggingFace tokenizer from model directory.
    
    Supports BPE, SentencePiece, Unigram tokenizers via transformers AutoTokenizer
    with fast Rust implementation.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    logger.info("Loading tokenizer from %s", path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        trust_remote_code=True,
        use_fast=True,
    )
    return tokenizer
