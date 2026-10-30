"""
Universal Model Loader for custom SLM inference engine.

Loads configurations, safetensors weights (single or sharded), resolves architectures
from the registry, and instantiates initialized models ready for inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, Type
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from safetensors import safe_open

from models.config import ModelConfig, load_config
from models.registry import get_architecture
# Import architectures to ensure registration
import models.architectures  # noqa: F401
from backends.base import detect_best_backend
from quantization.int8 import quantize_model_int8
from quantization.int4 import quantize_model_int4

logger = logging.getLogger(__name__)


class ModelLoader:
    """Universal model loader supporting Safetensors, HF formats, and multiple architectures."""

    @staticmethod
    def find_weight_files(model_path: Path) -> list[Path]:
        """Locate all weight files in model directory."""
        # 1. Check for sharded safetensors index
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            filenames = sorted(list(set(weight_map.values())))
            files = [model_path / fn for fn in filenames if (model_path / fn).exists()]
            if files:
                return files

        # 2. Check for single safetensors
        single_safetensors = list(model_path.glob("*.safetensors"))
        if single_safetensors:
            return sorted(single_safetensors)

        # 3. Check for PyTorch bin files
        bin_files = list(model_path.glob("*.bin"))
        if bin_files:
            return sorted(bin_files)

        raise FileNotFoundError(f"No valid weight files found in {model_path}")

    @classmethod
    def load_state_dict(
        cls,
        files: list[Path],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Load state dict from multiple weight files into target device and precision."""
        state_dict: Dict[str, torch.Tensor] = {}
        for file in files:
            logger.info("Loading weights from %s", file.name)
            if file.suffix == ".safetensors":
                tensors = load_safetensors(str(file), device=str(device))
            else:
                tensors = torch.load(str(file), map_location=device)

            for k, v in tensors.items():
                state_dict[k] = v.to(dtype=dtype, device=device)
        return state_dict

    @classmethod
    def load_model(
        cls,
        model_path: Union[str, Path],
        device: Optional[str] = "auto",
        precision: str = "fp16",
        quantization: Optional[str] = None,
    ) -> nn.Module:
        """Load complete model ready for inference.
        
        Uses zero-copy streaming via safe_open and CPU staging when quantizing
        to guarantee minimal peak memory overhead on Apple Silicon / unified RAM.
        """
        import gc
        path = Path(model_path)
        if not path.exists():
            # Check if in local checkpoints
            checkpoint_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / str(model_path)
            if checkpoint_dir.exists():
                path = checkpoint_dir
            elif "/" in str(model_path):
                # Auto-download from Hugging Face Hub
                logger.info("Model '%s' not found locally. Downloading from Hugging Face Hub...", model_path)
                try:
                    from huggingface_hub import snapshot_download
                    clean_name = str(model_path).split("/")[-1]
                    target_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / clean_name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    path = Path(snapshot_download(repo_id=str(model_path), local_dir=str(target_dir)))
                except Exception as e:
                    raise FileNotFoundError(f"Could not find or download model '{model_path}': {e}")
            else:
                raise FileNotFoundError(f"Model path does not exist: {path}")

        # 1. Parse Config
        config = load_config(path)
        logger.info(
            "Loaded model config: architecture=%s, vocab_size=%d, layers=%d, hidden_size=%d",
            config.architecture_type,
            config.vocab_size,
            config.num_hidden_layers,
            config.hidden_size,
        )

        # 2. Resolve Device & Dtype
        if not device or device == "auto":
            device_str = detect_best_backend()
        else:
            device_str = device
        target_device = torch.device(device_str)

        precision_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        target_dtype = precision_map.get(precision.lower(), torch.float16)

        # 3. Resolve Architecture
        arch_cls = get_architecture(config.architecture_type)
        logger.info("Instantiating architecture: %s", arch_cls.__name__)

        # 4. Instantiate Model on CPU first to prevent MPS double-allocation spikes
        model = arch_cls(config)
        model.to(device=torch.device("cpu"), dtype=target_dtype)

        # 5. Load Weights using zero-copy safe_open streaming when possible
        weight_files = cls.find_weight_files(path)
        remap_fn = getattr(arch_cls, "remap_weight_key", lambda k: k)

        all_safetensors = all(f.suffix == ".safetensors" for f in weight_files)
        if all_safetensors:
            logger.info("Using safe_open zero-copy weight streaming across %d files...", len(weight_files))
            model_state = model.state_dict()
            for fpath in weight_files:
                with safe_open(str(fpath), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        remapped_k = remap_fn(key)
                        if remapped_k in model_state:
                            model_state[remapped_k].copy_(f.get_tensor(key).to(dtype=target_dtype))
            del model_state
        else:
            raw_state_dict = cls.load_state_dict(weight_files, torch.device("cpu"), target_dtype)
            remapped_state_dict = {remap_fn(k): v for k, v in raw_state_dict.items()}
            del raw_state_dict
            model.load_state_dict(remapped_state_dict, strict=False)
            del remapped_state_dict

        # Check for tied word embeddings
        if config.tie_word_embeddings:
            model_state = model.state_dict()
            if "lm_head.weight" in model_state and "embed_tokens.weight" in model_state:
                model_state["lm_head.weight"].copy_(model_state["embed_tokens.weight"])
            del model_state

        gc.collect()

        # 6. Apply Quantization if requested before moving to target accelerator device
        if quantization == "int8":
            logger.info("Applying INT8 quantization...")
            model = quantize_model_int8(model)
        elif quantization == "int4":
            logger.info("Applying INT4 quantization...")
            model = quantize_model_int4(model)

        # 7. Move to target device
        model.to(device=target_device)
        if target_device.type == "mps":
            torch.mps.empty_cache()

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        logger.info(
            "Successfully loaded model %s on %s in %s (quantization=%s)",
            config.architecture_type,
            target_device,
            target_dtype,
            quantization,
        )
        return model
