"""
CLI interface for custom SLM inference engine.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from runtime.engine import InferenceEngine
from server.api import app, global_engine
from models.registry import list_architectures

console = Console()


@click.group()
def cli():
    """Custom SLM Inference Engine CLI."""
    pass


@cli.command("architectures")
def show_architectures():
    """List supported model architectures."""
    archs = list_architectures()
    table = Table(title="Supported SLM Architectures")
    table.add_column("Architecture Class", style="cyan")
    table.add_column("Description", style="green")

    descriptions = {
        "LlamaForCausalLM": "Llama 2, Llama 3, Mistral, CodeLlama family",
        "Qwen2ForCausalLM": "Qwen 2 / Qwen 2.5 family models",
        "Phi3ForCausalLM": "Microsoft Phi-3 / Phi-3.5 family models",
        "Gemma2ForCausalLM": "Google Gemma & Gemma 2 family models",
    }
    for arch in archs:
        table.add_row(arch, descriptions.get(arch, "Decoder-only transformer"))
    console.print(table)


CONFIG_FILE = Path.home() / ".slm_active_model.json"


def save_active_model(info: dict) -> None:
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
    except Exception:
        pass


def load_active_model() -> Optional[dict]:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


@cli.command("load")
@click.argument("model_path")
@click.option("--device", default="auto", help="Execution device: auto, cpu, mps, cuda")
@click.option("--precision", default="fp16", help="Precision: fp16, fp32, bf16")
@click.option("--quantization", default=None, help="Quantization: int8, int4")
def load_cmd(model_path, device, precision, quantization):
    """Load and persist active model configuration."""
    resolved_path = str(Path(model_path).resolve())
    console.print(f"[bold blue]Loading model from:[/] {resolved_path} (device={device}, precision={precision})...")
    try:
        global_engine.load_model(
            model_path=resolved_path,
            device=device,
            precision=precision,
            quantization=quantization,
        )
        save_active_model({
            "path": resolved_path,
            "device": device,
            "precision": precision,
            "quantization": quantization,
        })
        console.print("[bold green]✓ Model loaded and set as active default![/]")
    except Exception as e:
        console.print(f"[bold red]✗ Error loading model:[/] {e}")
        sys.exit(1)


@cli.command("run")
@click.option("--prompt", "-p", required=True, help="Input prompt")
@click.option("--model", default=None, help="Path to model (defaults to active model)")
@click.option("--device", default="auto", help="Execution device")
@click.option("--max-tokens", "-m", default=128, help="Max tokens to generate")
@click.option("--temperature", "-t", default=0.7, help="Sampling temperature")
@click.option("--stream/--no-stream", default=True, help="Stream token output")
def run_cmd(prompt, model, device, max_tokens, temperature, stream):
    """Run inference generation on the model."""
    if not global_engine.is_loaded:
        model_path = model
        precision = "fp16"
        quantization = None
        target_device = device

        if not model_path:
            active = load_active_model()
            if active:
                model_path = active.get("path")
                target_device = target_device if target_device != "auto" else active.get("device", "auto")
                precision = active.get("precision", "fp16")
                quantization = active.get("quantization")

        if not model_path:
            console.print("[bold red]No model specified or active. Pass --model <path> or run 'slm load <path>' first.[/]")
            sys.exit(1)

        console.print(f"[dim]Loading active model: {model_path} on {target_device}...[/dim]")
        global_engine.load_model(
            model_path=model_path,
            device=target_device,
            precision=precision,
            quantization=quantization,
        )

    console.print(f"\n[bold yellow]Prompt:[/] {prompt}\n[bold green]Response:[/] ", end="")
    if stream:
        for token in global_engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            console.print(token, end="")
        console.print()
    else:
        output = global_engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        console.print(output)


@cli.command("status")
def status_cmd():
    """Display runtime status and hardware statistics."""
    status = global_engine.get_status()
    console.print(Panel.fit(str(status), title="SLM Engine Status", border_style="cyan"))


@cli.command("train")
@click.option("--epochs", "-e", default=45, help="Number of LoRA training epochs")
@click.option("--rank", "-r", default=32, help="LoRA rank r")
@click.option("--alpha", "-a", default=64.0, help="LoRA scaling alpha")
@click.option("--device", default="auto", help="Training device: auto, mps, cpu")
def train_cmd(epochs, rank, alpha, device):
    """Run LoRA fine-tuning and SVD spectral weight conditioning."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from training.train_lora import run_lora_training
    console.print(f"[bold blue]Starting LoRA + SVD Weight Surgery (epochs={epochs}, r={rank}, alpha={alpha})...[/]")
    summary = run_lora_training(
        epochs=epochs,
        lora_r=rank,
        lora_alpha=alpha,
        device_str=device,
        reset_base_weights=True,
    )
    table = Table(title="SLM Weight Training & SVD Conditioning Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Measured Result", style="green")
    table.add_row("Accelerator Device", summary["device"])
    table.add_row("Dataset Samples", f"{summary['dataset_samples']} multi-domain conversations (incl. unslop, logic, efficiency)")
    table.add_row("LoRA Rank / Projections", f"r={summary['lora_rank']}, α={summary['lora_alpha']} ({summary['merged_projection_layers']} layers merged)")
    table.add_row("Cross-Entropy Loss", f"{summary['initial_loss']:.4f} → {summary['final_loss']:.4f} (-{summary['loss_reduction_pct']}%)")
    table.add_row("Perplexity", f"{summary['perplexity_before']} → {summary['perplexity_after']}")
    table.add_row("Next-Token Accuracy", f"{summary['token_accuracy_before_pct']}% → {summary['token_accuracy_after_pct']}%")
    table.add_row("Weight Delta ||ΔW||_F", f"{summary['weight_delta_frobenius_norm']} (Sparsity: {summary['structured_sparsity_pct']}%)")
    table.add_row("Elapsed Time", f"{summary['elapsed_seconds']}s")
    console.print(table)


@cli.command("serve")
@click.option("--host", default="0.0.0.0", help="Host interface to bind")
@click.option("--port", default=8000, help="Port to listen on")
@click.option("--model", default=None, help="Optional model path to auto-load")
@click.option("--demo", is_flag=True, help="Auto-bootstrap synthetic micro Qwen2 model if no model is loaded")
def serve_cmd(host, port, model, demo):
    """Start the OpenAI-compatible HTTP/SSE API server."""
    import uvicorn
    if model:
        console.print(f"[bold blue]Pre-loading model:[/] {model}")
        global_engine.load_model(model)
    else:
        active = load_active_model()
        if active and active.get("path") and Path(active["path"]).exists():
            console.print(f"[bold blue]Pre-loading active model:[/] {active['path']} (quantization={active.get('quantization')})")
            global_engine.load_model(
                model_path=active["path"],
                device=active.get("device", "auto"),
                precision=active.get("precision", "fp16"),
                quantization=active.get("quantization"),
            )
        elif demo:
            console.print("[bold cyan]Bootstrapping synthetic micro Qwen2 model (INT8)...[/]")
            global_engine.load_synthetic_demo_model(quantization="int8")

    console.print(f"[bold green]Starting SLM Inference Server on http://{host}:{port}[/]")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
