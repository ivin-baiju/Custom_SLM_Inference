"""
FastAPI Server for custom SLM inference engine.

Exposes OpenAI-compatible endpoints:
  POST /v1/chat/completions
  GET  /v1/models
  POST /v1/models/load
  POST /v1/models/switch
  POST /v1/models/unload
  GET  /v1/status
  GET  /health
  GET  /metrics
"""

from __future__ import annotations

import logging
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from runtime.engine import InferenceEngine
from server.endpoints import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
    ModelLoadRequest,
    SwitchModelRequest,
    DemoLoadRequest,
)
from server.streaming import stream_chat_completion

logger = logging.getLogger(__name__)

# Global inference engine singleton
global_engine = InferenceEngine()

app = FastAPI(
    title="Custom SLM Inference Engine",
    description="High-performance, hardware-aware execution system for Small Language Models",
    version="0.1.0",
)

# Enable CORS for web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_dashboard():
    dashboard_path = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path))
    return {"message": "Custom SLM Inference Engine is online"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "engine_ready": global_engine.is_loaded,
        "backend": global_engine.backend.name,
    }


@app.get("/v1/models")
def list_models():
    current = global_engine.lifecycle.current_model
    data = []
    if current:
        data.append({
            "id": current,
            "object": "model",
            "owned_by": "custom-slm-runtime",
            "permission": [],
        })
    return {"object": "list", "data": data}


@app.get("/v1/models/available")
def list_available_models():
    """Discover all local checkpoints available on disk for instant hot-switching."""
    checkpoints_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoints"
    available = []
    if checkpoints_dir.exists():
        for d in sorted(checkpoints_dir.iterdir()):
            if d.is_dir() and (d / "config.json").exists():
                try:
                    with open(d / "config.json", "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    size_mb = sum(f.stat().st_size for f in d.glob("*.safetensors")) / (1024**2)
                    is_active = (global_engine.model_path and Path(global_engine.model_path).name == d.name)
                    available.append({
                        "name": d.name,
                        "path": str(d),
                        "architecture": cfg.get("architectures", [cfg.get("model_type", "Unknown")])[0],
                        "size_mb": round(size_mb, 1),
                        "hidden_size": cfg.get("hidden_size"),
                        "num_layers": cfg.get("num_hidden_layers"),
                        "is_active": bool(is_active),
                    })
                except Exception as e:
                    logger.debug("Failed parsing config for %s: %s", d.name, e)
    return {"models": available}


@app.get("/v1/status")
def get_engine_status():
    return global_engine.get_status()


@app.post("/v1/models/demo")
def load_demo_model_endpoint(req: Optional[DemoLoadRequest] = None):
    """Bootstrap an instant micro Qwen2ForCausalLM safetensors checkpoint for live hardware verification."""
    cfg = req or DemoLoadRequest()
    try:
        model_name = global_engine.load_synthetic_demo_model(
            device=cfg.device,
            precision=cfg.precision,
            quantization=cfg.quantization,
            cache_type=cfg.cache_type,
        )
        return {
            "status": "loaded",
            "model": model_name,
            "is_synthetic_demo": True,
        }
    except Exception as e:
        logger.exception("Failed to bootstrap synthetic demo model")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/v1/models/load")
def load_model_endpoint(req: ModelLoadRequest):
    try:
        if req.path in ("qwen2.5-micro-synthetic", "synthetic-demo", "demo"):
            model_name = global_engine.load_synthetic_demo_model(
                device=req.device,
                precision=req.precision,
                quantization=req.quantization,
                cache_type=req.cache_type,
            )
            return {"status": "loaded", "model": model_name}

        model_p = Path(req.path)
        if not model_p.exists():
            checkpoint_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / req.path
            if checkpoint_dir.exists():
                model_p = checkpoint_dir

        if global_engine.is_loaded:
            global_engine.unload_model()

        global_engine.load_model(
            model_path=str(model_p),
            device=req.device,
            precision=req.precision,
            cache_type=req.cache_type,
            quantization=req.quantization,
        )
        return {"status": "loaded", "model": global_engine.lifecycle.current_model}
    except Exception as e:
        logger.exception("Failed to load model")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/v1/models/switch")
def switch_model_endpoint(req: SwitchModelRequest):
    try:
        if req.path in ("qwen2.5-micro-synthetic", "synthetic-demo", "demo"):
            model_name = global_engine.load_synthetic_demo_model(
                device=req.device,
                precision=req.precision,
                quantization=req.quantization,
                cache_type=req.cache_type,
            )
            return {"status": "switched", "model": model_name}

        model_p = Path(req.path)
        if not model_p.exists():
            checkpoint_dir = Path(__file__).resolve().parent.parent / "models" / "checkpoints" / req.path
            if checkpoint_dir.exists():
                model_p = checkpoint_dir

        global_engine.switch_model(
            new_model_path=str(model_p),
            device=req.device,
            precision=req.precision,
            cache_type=req.cache_type,
            quantization=req.quantization,
        )
        return {"status": "switched", "model": global_engine.lifecycle.current_model}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/v1/models/unload")
def unload_model_endpoint():
    global_engine.unload_model()
    return {"status": "unloaded"}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not global_engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No model is currently loaded in the inference engine.",
        )

    if req.stream:
        return EventSourceResponse(stream_chat_completion(req, global_engine))

    # Non-streaming execution with telemetry
    messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]
    stop_seqs = [req.stop] if isinstance(req.stop, str) else (req.stop or [])
    try:
        if not global_engine.tokenizer_processor:
            raise RuntimeError("No tokenizer available in engine.")

        prompt = global_engine.tokenizer_processor.apply_chat_template(
            messages_dicts, add_generation_prompt=True
        )
        chunks = []
        last_meta: Dict[str, Any] = {}
        for _, piece, meta in global_engine.generate_with_telemetry(
            prompt=prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            stop_sequences=stop_seqs,
        ):
            chunks.append(piece)
            last_meta = meta

        response_text = "".join(chunks)
        prompt_tok = int(last_meta.get("prompt_tokens", len(global_engine.tokenizer_processor.encode(prompt))))
        comp_tok = int(last_meta.get("completion_tokens", len(chunks)))

        return ChatCompletionResponse(
            model=global_engine.lifecycle.current_model or "slm",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                total_tokens=prompt_tok + comp_tok,
            ),
            telemetry=last_meta or None,
        )
    except Exception as e:
        logger.exception("Error executing generation")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/metrics", response_class=PlainTextResponse)
def metrics_endpoint():
    """Prometheus-compatible plain text metrics."""
    from monitoring.metrics import MetricsCollector
    return PlainTextResponse(MetricsCollector.get_instance().format_prometheus())
