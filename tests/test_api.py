"""
Unit tests for FastAPI REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from server.api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "backend" in data


def test_list_models_empty():
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"


def test_status_endpoint():
    response = client.get("/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert "state" in data
    assert "backend" in data
    assert "metrics" in data
    assert "recent_traces" in data


def test_demo_model_bootstrap_and_chat_completions():
    # Bootstrap in-repo synthetic micro Qwen2 model
    load_resp = client.post(
        "/v1/models/demo",
        json={"device": "cpu", "precision": "fp16", "quantization": "int8", "cache_type": "dynamic"},
    )
    assert load_resp.status_code == 200
    assert load_resp.json()["status"] == "loaded"
    assert load_resp.json()["is_synthetic_demo"] is True

    # Non-streaming chat completion with telemetry and repetition_penalty
    chat_resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Explain the mechanism of ACE inhibitors"}],
            "max_tokens": 12,
            "temperature": 0.7,
            "repetition_penalty": 1.1,
            "stream": False,
        },
    )
    assert chat_resp.status_code == 200
    body = chat_resp.json()
    assert len(body["choices"]) == 1
    assert len(body["choices"][0]["message"]["content"]) > 0
    assert body["usage"]["completion_tokens"] > 0
    assert body["telemetry"] is not None
    assert "ttft_ms" in body["telemetry"]

    # Check Prometheus metrics recorded the request
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "slm_requests_total" in metrics_resp.text
    assert "slm_tokens_generated_total" in metrics_resp.text

    # Streaming SSE chat completion with per-chunk telemetry
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Explain std::move in C++"}],
            "max_tokens": 10,
            "temperature": 0.7,
            "stream": True,
        },
    ) as stream_resp:
        assert stream_resp.status_code == 200
        lines = [line for line in stream_resp.iter_lines() if line.startswith("data: ")]
        assert len(lines) >= 10
        assert lines[-1] == "data: [DONE]"

    # Unload model cleanly
    unload_resp = client.post("/v1/models/unload")
    assert unload_resp.status_code == 200
    assert unload_resp.json()["status"] == "unloaded"
