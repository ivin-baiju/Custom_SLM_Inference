"""
Native PyTorch LoRA Fine-Tuning & Direct Weight Surgery Pipeline for Custom SLM.

Trains and alters Qwen2ForCausalLM weights across 7 specialized domains:
  1. Unslop UI/UX Redline Auditing & 9-Pass Design Engineering (@heydamir/unslop)
  2. Formal Deductive Logic & Algorithmic Correctness Invariants
  3. Hardware & Quantization Efficiency (Zero-Copy GQA, Per-Channel INT8, Roofline)
  4. Clinical Pharmacology & Hemodynamics
  5. Cinema Theory & Relativistic Narrative Architecture
  6. High-Performance C++20 & Python 3.12 Systems Concurrency
  7. Academic Research Methodology & Meta-Analysis

Executes native PyTorch LoRA adaptation (q_proj, k_proj, v_proj, o_proj, gate_proj,
up_proj, down_proj) + SVD spectral conditioning and weight merging directly into
Safetensors checkpoints on Apple Silicon MPS / CPU without external wrapper bloat.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

from models.config import load_config
from models.architectures.qwen import Qwen2Model
from quantization.int8 import quantize_tensor_int8, dequantize_tensor_int8


class NativeLoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) wrapper for nn.Linear with direct weight merge & SVD surgery."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 32,
        lora_alpha: float = 64.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = min(r, self.in_features, self.out_features)
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / float(self.r)
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        device = base_layer.weight.device
        # Keep LoRA trainable parameters in FP32 for high-precision gradient accumulation
        self.lora_A = nn.Parameter(torch.empty((self.r, self.in_features), dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.r), dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Joint Full + Low-Rank (DoRA/LoRA Hybrid) adaptation so both base weights and low-rank subspace optimize
        self.base_layer.weight.requires_grad = True
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        x_fp32 = self.dropout(x.to(torch.float32))
        lora_delta = (x_fp32 @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return base_out + lora_delta.to(base_out.dtype)

    @torch.no_grad()
    def merge_and_condition(
        self,
        spectral_blend: float = 0.15,
        prune_threshold: float = 1e-4,
    ) -> Tuple[nn.Linear, float, float]:
        """Merge LoRA delta (B @ A) into base weight and apply SVD spectral conditioning.

        Returns:
            (merged_linear_layer, delta_frobenius_norm, sparsity_fraction)
        """
        orig_dtype = self.base_layer.weight.dtype
        w_base = self.base_layer.weight.data.to(torch.float32)
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        w_merged = w_base + delta_w

        # SVD Spectral Conditioning: sharpen dominant logical singular vectors
        # while attenuating high-frequency tail noise for cleaner INT8/INT4 quantization
        if spectral_blend > 0.0 and min(w_merged.shape) >= 8:
            w_cpu = w_merged.cpu()
            u, s, vh = torch.linalg.svd(w_cpu, full_matrices=False)
            # Soft spectral gate: preserve top singular components, dampen noise floor
            median_s = torch.median(s)
            spectral_gate = torch.sigmoid((s - 0.35 * median_s) / (0.15 * median_s + 1e-6))
            s_conditioned = s * ((1.0 - spectral_blend) + spectral_blend * spectral_gate)
            # Normalize energy to preserve layer variance stability
            energy_scale = (torch.norm(s) / (torch.norm(s_conditioned) + 1e-8)).clamp(0.85, 1.15)
            w_merged = (u @ torch.diag(s_conditioned * energy_scale) @ vh).to(w_merged.device)

        # Prune sub-threshold micro-weights to improve sparsity & quantization SNR
        if prune_threshold > 0.0:
            mask = torch.abs(w_merged) >= prune_threshold
            sparsity = float((~mask).float().mean().item())
            w_merged = w_merged * mask
        else:
            sparsity = 0.0

        delta_norm = float(torch.norm(w_merged - w_base, p="fro").item())
        self.base_layer.weight.data.copy_(w_merged.to(orig_dtype))
        return self.base_layer, delta_norm, sparsity


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def inject_lora_adapters(
    model: nn.Module,
    r: int = 32,
    lora_alpha: float = 64.0,
    lora_dropout: float = 0.02,
) -> Tuple[int, int]:
    """Replace target nn.Linear projections with NativeLoRALinear and return (trainable_params, total_params)."""
    for p in model.parameters():
        p.requires_grad = True

    def _replace(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name in TARGET_MODULES:
                setattr(module, name, NativeLoRALinear(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
            else:
                _replace(child)

    _replace(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def merge_lora_adapters(
    model: nn.Module,
    spectral_blend: float = 0.15,
    prune_threshold: float = 1e-4,
) -> Tuple[float, float, int]:
    """Merge all NativeLoRALinear layers back into standard nn.Linear layers in-place."""
    total_delta_norm = 0.0
    total_sparsity = 0.0
    merged_count = 0

    def _merge(module: nn.Module) -> None:
        nonlocal total_delta_norm, total_sparsity, merged_count
        for name, child in module.named_children():
            if isinstance(child, NativeLoRALinear):
                base_lin, d_norm, spar = child.merge_and_condition(
                    spectral_blend=spectral_blend,
                    prune_threshold=prune_threshold,
                )
                setattr(module, name, base_lin)
                total_delta_norm += d_norm
                total_sparsity += spar
                merged_count += 1
            else:
                _merge(child)

    _merge(model)
    avg_sparsity = (total_sparsity / merged_count) if merged_count > 0 else 0.0
    return total_delta_norm, avg_sparsity, merged_count


def encode_text_deterministic(text: str, vocab_size: int = 512, max_len: int = 96) -> List[int]:
    """Deterministic subword hash tokenizer mapping domain text into [0, vocab_size)."""
    import re
    tokens = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", text.lower())
    ids = [1]  # BOS = 1
    for tok in tokens[: max_len - 2]:
        # Deterministic polynomial rolling hash across characters (independent of PYTHONHASHSEED)
        h = 2166136261
        for ch in tok:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        ids.append(3 + (h % (vocab_size - 4)))
    ids.append(2)  # EOS = 2
    return ids


def build_training_batches(
    jsonl_path: Path,
    vocab_size: int = 512,
    seq_len: int = 64,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """Load conversational JSONL dataset and build padded token ID tensors for causal LM training."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    sequences: List[torch.Tensor] = []
    for row in rows:
        msgs = row.get("messages", [])
        combined = " ".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs)
        ids = encode_text_deterministic(combined, vocab_size=vocab_size, max_len=seq_len)
        if len(ids) < seq_len:
            ids = ids + [0] * (seq_len - len(ids))
        else:
            ids = ids[:seq_len]
        sequences.append(torch.tensor(ids, dtype=torch.long, device=device))

    # Stack into mini-batches of 4 sequences
    batches: List[torch.Tensor] = []
    batch_size = 4
    for i in range(0, len(sequences), batch_size):
        batches.append(torch.stack(sequences[i : i + batch_size], dim=0))
    return batches


def run_lora_training(
    epochs: int = 45,
    lora_r: int = 32,
    lora_alpha: float = 64.0,
    learning_rate: float = 4e-3,
    spectral_blend: float = 0.08,
    device_str: Optional[str] = None,
    reset_base_weights: bool = False,
) -> Dict[str, Any]:
    """Execute LoRA fine-tuning + SVD weight surgery and save updated Safetensors checkpoints."""
    t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parent.parent
    dataset_file = repo_root / "training" / "dataset.jsonl"
    micro_dir = repo_root / "models" / "checkpoints" / "qwen2.5-micro-synthetic"
    expert_dir = repo_root / "models" / "checkpoints" / "qwen2.5-unslop-expert"
    metrics_file = repo_root / "training" / "training_metrics.json"

    if device_str is None or device_str == "auto":
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)

    # Ensure base micro checkpoint exists before loading
    if reset_base_weights or not (micro_dir / "config.json").exists() or not (micro_dir / "model.safetensors").exists():
        if (micro_dir / "model.safetensors").exists() and reset_base_weights:
            (micro_dir / "model.safetensors").unlink()
        from runtime.engine import InferenceEngine
        eng = InferenceEngine()
        eng.load_synthetic_demo_model(device="cpu", quantization=None)
        eng.unload_model()

    config = load_config(micro_dir)
    model = Qwen2Model(config)

    # Load existing weights (in FP32 for stable training gradients)
    raw_weights = load_file(str(micro_dir / "model.safetensors"), device="cpu")
    remapped = {Qwen2Model.remap_weight_key(k): v.to(torch.float32) for k, v in raw_weights.items()}
    model.load_state_dict(remapped, strict=False)
    model.to(device=device, dtype=torch.float32)

    # Measure baseline INT8 quantization error on q_proj before weight conditioning
    sample_w_before = model.layers[0].self_attn.q_proj.weight.data.detach().cpu()
    q_b, s_b = quantize_tensor_int8(sample_w_before)
    quant_err_before = float(torch.abs(sample_w_before - dequantize_tensor_int8(q_b, s_b)).mean().item())

    # Build multi-domain dataset batches
    batches = build_training_batches(
        dataset_file,
        vocab_size=config.vocab_size,
        seq_len=64,
        device=device,
    )
    with open(dataset_file, "r", encoding="utf-8") as f:
        num_samples = sum(1 for line in f if line.strip())

    # Inject Native LoRA Adapters
    trainable_params, total_params = inject_lora_adapters(
        model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.02
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.1)

    model.train()
    epoch_losses: List[float] = []
    initial_acc = 0.0
    final_acc = 0.0

    for epoch in range(epochs):
        total_loss = 0.0
        correct_tokens = 0
        valid_tokens = 0

        for batch in batches:
            optimizer.zero_grad()
            # Causal LM shift: predict batch[:, 1:] from batch[:, :-1]
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(inputs, start_pos=0, kv_cache=None)
            vocab_sz = logits.size(-1)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_sz),
                targets.reshape(-1),
                ignore_index=0,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = targets != 0
                correct_tokens += int(((preds == targets) & mask).sum().item())
                valid_tokens += int(mask.sum().item())

        scheduler.step()
        avg_loss = total_loss / max(1, len(batches))
        epoch_losses.append(round(avg_loss, 4))
        acc = (correct_tokens / max(1, valid_tokens)) * 100.0
        if epoch == 0:
            initial_acc = round(acc, 2)
        if epoch == epochs - 1:
            final_acc = round(acc, 2)

    # Merge LoRA adapters & apply SVD spectral conditioning into base weights
    model.eval()
    delta_fro_norm, avg_sparsity, merged_layers = merge_lora_adapters(
        model, spectral_blend=spectral_blend, prune_threshold=1e-4
    )

    # Measure post-conditioning per-channel INT8 quantization error
    sample_w_after = model.layers[0].self_attn.q_proj.weight.data.detach().cpu()
    q_a, s_a = quantize_tensor_int8(sample_w_after)
    quant_err_after = float(torch.abs(sample_w_after - dequantize_tensor_int8(q_a, s_a)).mean().item())

    # Serialize altered FP16 weights to both qwen2.5-micro-synthetic and qwen2.5-unslop-expert
    state_to_save: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        prefixed_key = k if k.startswith("lm_head") else f"model.{k}"
        state_to_save[prefixed_key] = v.detach().cpu().to(torch.float16).contiguous()

    save_file(state_to_save, str(micro_dir / "model.safetensors"))

    expert_dir.mkdir(parents=True, exist_ok=True)
    with open(micro_dir / "config.json", "r", encoding="utf-8") as f:
        cfg_json = json.load(f)
    with open(expert_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, indent=2)
    save_file(state_to_save, str(expert_dir / "model.safetensors"))

    elapsed_sec = round(time.perf_counter() - t0, 2)
    initial_loss = epoch_losses[0] if epoch_losses else 0.0
    final_loss = epoch_losses[-1] if epoch_losses else 0.0
    loss_reduction_pct = round(((initial_loss - final_loss) / max(1e-6, initial_loss)) * 100.0, 1)
    ppl_before = round(math.exp(min(10.0, initial_loss)), 2)
    ppl_after = round(math.exp(min(10.0, final_loss)), 2)

    # Select 5 milestone epochs for compact UI display
    if len(epoch_losses) >= 5:
        idxs = [0, len(epoch_losses) // 4, len(epoch_losses) // 2, (3 * len(epoch_losses)) // 4, len(epoch_losses) - 1]
        milestones = [epoch_losses[i] for i in idxs]
    else:
        milestones = epoch_losses

    summary = {
        "status": "completed",
        "device": device_str.upper(),
        "dataset_samples": num_samples,
        "domains": [
            "unslop UI/UX Redline & 9-Pass Design",
            "Formal Deductive Logic & Loop Invariants",
            "Hardware & Quantization Efficiency",
            "Clinical Pharmacology",
            "Cinema Theory",
            "Systems C++20 / Python 3.12",
            "Research Methodology",
        ],
        "lora_rank": lora_r,
        "lora_alpha": lora_alpha,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "merged_projection_layers": merged_layers,
        "epochs": epochs,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": loss_reduction_pct,
        "perplexity_before": ppl_before,
        "perplexity_after": ppl_after,
        "token_accuracy_before_pct": initial_acc,
        "token_accuracy_after_pct": final_acc,
        "weight_delta_frobenius_norm": round(delta_fro_norm, 4),
        "structured_sparsity_pct": round(avg_sparsity * 100.0, 2),
        "int8_mae_before": round(quant_err_before, 6),
        "int8_mae_after": round(quant_err_after, 6),
        "epoch_loss_curve": epoch_losses,
        "milestone_losses": milestones,
        "elapsed_seconds": elapsed_sec,
        "checkpoints_updated": [
            str(micro_dir / "model.safetensors"),
            str(expert_dir / "model.safetensors"),
        ],
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    print("=== Starting Native PyTorch LoRA Training & SVD Weight Surgery ===")
    summary = run_lora_training(reset_base_weights=True)
    print(f"Device               : {summary['device']}")
    print(f"Dataset Samples      : {summary['dataset_samples']} multi-domain conversations")
    print(f"LoRA Config          : r={summary['lora_rank']}, alpha={summary['lora_alpha']} ({summary['merged_projection_layers']} projections merged)")
    print(f"Cross-Entropy Loss   : {summary['initial_loss']:.4f} -> {summary['final_loss']:.4f} (-{summary['loss_reduction_pct']}%)")
    print(f"Perplexity           : {summary['perplexity_before']} -> {summary['perplexity_after']}")
    print(f"Next-Token Accuracy  : {summary['token_accuracy_before_pct']}% -> {summary['token_accuracy_after_pct']}%")
    print(f"Weight Delta ||dW||_F: {summary['weight_delta_frobenius_norm']} (Sparsity: {summary['structured_sparsity_pct']}%)")
    print(f"Checkpoints Updated  : {', '.join(summary['checkpoints_updated'])}")
    print(f"Completed in {summary['elapsed_seconds']}s.")


if __name__ == "__main__":
    main()
