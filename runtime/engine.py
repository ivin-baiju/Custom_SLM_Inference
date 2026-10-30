"""
Core InferenceEngine orchestrator for custom SLM inference runtime.

Provides universal high-performance inference across multiple SLM architectures,
device backends, KV caching strategies, and quantization modes.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Generator, List, Optional, Dict, Any, Union
import torch

from models.config import ModelConfig, load_config
from models.loader import ModelLoader
from backends.base import ExecutionBackend, get_backend
from tokenizer.loader import load_tokenizer
from tokenizer.processor import TokenProcessor
from cache.kv_cache import KVCache, DynamicKVCache, StaticKVCache
from runtime.lifecycle import ModelLifecycle, LifecycleState
from runtime.generation import generate_tokens
from runtime.request import GenerationConfig, InferenceRequest, InferenceResponse, StreamChunk
from quantization.int8 import quantize_model_int8
from quantization.int4 import quantize_model_int4

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Universal Small Language Model Inference Engine."""

    def __init__(self, backend: Optional[str] = "auto"):
        self.backend: ExecutionBackend = get_backend(backend)
        self.lifecycle = ModelLifecycle(self.backend)

        self.model: Optional[torch.nn.Module] = None
        self.config: Optional[ModelConfig] = None
        self.tokenizer_processor: Optional[TokenProcessor] = None
        self.kv_cache: Optional[KVCache] = None
        self.model_path: Optional[str] = None
        self.precision: str = "fp16"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.lifecycle.state == LifecycleState.READY

    def load_model(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        precision: str = "fp16",
        cache_type: str = "dynamic",
        max_seq_len: int = 2048,
        quantization: Optional[str] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """Load model, tokenizer, and initialize KV cache.
        
        Args:
            model_path: Path to model directory
            device: 'auto', 'cpu', 'mps', or 'cuda'
            precision: 'fp32', 'fp16', or 'bf16'
            cache_type: 'dynamic' or 'static'
            max_seq_len: Context window limit for KV cache
            quantization: None, 'int8', or 'int4'
            tokenizer: Optional pre-loaded or mock tokenizer instance
        """
        path = Path(model_path)
        model_name = path.name

        if device:
            self.backend = get_backend(device)
            self.lifecycle = ModelLifecycle(self.backend)

        self.lifecycle.begin_loading(model_name, str(path))
        self.precision = precision
        self.model_path = str(path)

        try:
            # 1. Load config
            self.config = load_config(path)

            # 2. Load model weights and architecture (with optional quantization and zero-copy streaming)
            self.model = ModelLoader.load_model(
                path,
                device=self.backend.device,
                precision=precision,
                quantization=quantization,
            )

            # 4. Load tokenizer
            if tokenizer is not None:
                if isinstance(tokenizer, TokenProcessor):
                    self.tokenizer_processor = tokenizer
                else:
                    self.tokenizer_processor = TokenProcessor(tokenizer)
            else:
                raw_tokenizer = load_tokenizer(path)
                self.tokenizer_processor = TokenProcessor(raw_tokenizer)

            # 5. Initialize KV Cache
            num_kv_heads = self.config.num_key_value_heads or self.config.num_attention_heads
            head_dim = self.config.head_dim or (self.config.hidden_size // self.config.num_attention_heads)

            dtype_map = {
                "fp32": torch.float32,
                "float32": torch.float32,
                "fp16": torch.float16,
                "float16": torch.float16,
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
            }
            cache_dtype = dtype_map.get(precision.lower(), torch.float16)

            if cache_type == "static":
                self.kv_cache = StaticKVCache(
                    num_layers=self.config.num_hidden_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    max_seq_len=max_seq_len,
                    dtype=cache_dtype,
                    device=self.backend.device,
                )
            else:
                self.kv_cache = DynamicKVCache(
                    num_layers=self.config.num_hidden_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    max_seq_len=max_seq_len,
                    dtype=cache_dtype,
                    device=self.backend.device,
                )

            self.lifecycle.set_ready()
            logger.info("Model '%s' successfully loaded and initialized", model_name)

        except Exception as e:
            logger.error("Failed to load model '%s': %s", model_name, e)
            self.unload_model()
            raise

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response for a prompt string.
        
        If stream=True, returns a generator yielding strings.
        If stream=False, returns the complete generated string.
        """
        if not self.is_loaded or self.model is None or self.tokenizer_processor is None:
            raise RuntimeError("No model is currently loaded. Call load_model() first.")

        gen_config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_sequences=tuple(stop_sequences or ()),
        )

        input_ids = self.tokenizer_processor.encode(prompt)

        def _generator() -> Generator[str, None, None]:
            self.lifecycle.acquire_inference()
            try:
                for _, text_chunk, _ in generate_tokens(
                    prompt_ids=input_ids,
                    model=self.model,
                    kv_cache=self.kv_cache,
                    gen_config=gen_config,
                    tokenizer=self.tokenizer_processor,
                    device=self.backend.device,
                ):
                    yield text_chunk
            finally:
                self.lifecycle.release_inference()

        if stream:
            return _generator()
        else:
            return "".join(_generator())

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response for a chat conversation using model's chat template."""
        if not self.tokenizer_processor:
            raise RuntimeError("No tokenizer available. Call load_model() first.")

        prompt = self.tokenizer_processor.apply_chat_template(messages, add_generation_prompt=True)
        return self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stream=stream,
        )

    def switch_model(self, new_model_path: Union[str, Path], **kwargs) -> None:
        """Hot model switching without restarting runtime."""
        logger.info("Hot switching model to '%s'", new_model_path)
        self.unload_model()
        self.load_model(new_model_path, **kwargs)

    def unload_model(self) -> None:
        """Release loaded model and KV cache resources."""
        self.lifecycle.unload(self.model)
        self.model = None
        self.config = None
        self.tokenizer_processor = None
        self.kv_cache = None

    def get_status(self) -> Dict[str, Any]:
        """Engine runtime status, KV cache stats, and device metrics."""
        status = self.lifecycle.get_status()
        status["backend"] = self.backend.name
        status["device"] = str(self.backend.device)
        status["precision"] = self.precision

        if self.config:
            status["config"] = {
                "architecture": self.config.architecture_type,
                "vocab_size": self.config.vocab_size,
                "hidden_size": self.config.hidden_size,
                "layers": self.config.num_hidden_layers,
                "attention_heads": self.config.num_attention_heads,
                "kv_heads": self.config.num_key_value_heads,
            }

        if self.kv_cache:
            status["kv_cache"] = {
                "memory_bytes": self.kv_cache.memory_usage(),
                "seq_length": self.kv_cache.get_seq_length(),
                "max_seq_len": self.kv_cache.max_seq_len,
            }

        return status
