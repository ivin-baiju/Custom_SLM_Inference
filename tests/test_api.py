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
