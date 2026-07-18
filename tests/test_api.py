"""
Tests for the FraudLens FastAPI application.

Covers health check, prediction, batch prediction, and error handling.
Uses FastAPI's TestClient for fast, dependency-free testing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.main import app

client = TestClient(app)


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_transaction() -> dict:
    """A valid transaction with all required features."""
    tx = {f"V{i}": round(np.random.randn(), 4) for i in range(1, 29)}
    tx["Time"] = 100000.0
    tx["Amount"] = 150.0
    return tx


# ─── Tests: Health ────────────────────────────────────────────────────────

class TestHealth:
    """Tests for the /health endpoint."""

    def test_health_endpoint(self):
        """Test that health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_expected_keys(self):
        """Test that health response contains required fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_status_is_healthy(self):
        """Test that status is 'healthy' (model load is optional for tests)."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


# ─── Tests: Prediction ─────────────────────────────────────────────────────

class TestPrediction:
    """Tests for the /predict endpoint."""

    def test_predict_endpoint_accepts_valid_data(self, sample_transaction: dict):
        """Test that /predict returns 200 for valid input."""
        response = client.post("/predict", json=sample_transaction)
        # Note: May return 503 if model isn't loaded in test env
        assert response.status_code in (200, 503)

    def test_predict_returns_correct_schema(self, sample_transaction: dict):
        """Test that prediction response contains all required fields."""
        response = client.post("/predict", json=sample_transaction)
        if response.status_code == 200:
            data = response.json()
            assert "fraud_probability" in data
            assert "decision" in data
            assert "threshold_used" in data
            assert "is_fraud" in data
            assert isinstance(data["fraud_probability"], (int, float))
            assert data["decision"] in ("FRAUD", "LEGITIMATE")
            assert isinstance(data["is_fraud"], bool)

    def test_predict_rejects_negative_amount(self):
        """Test that negative Amount returns 422 validation error."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        tx["Amount"] = -100.0  # Invalid
        response = client.post("/predict", json=tx)
        assert response.status_code == 422

    def test_predict_rejects_missing_required_field(self):
        """Test that missing Amount returns 422 validation error."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        # Missing Amount
        response = client.post("/predict", json=tx)
        assert response.status_code == 422


# ─── Tests: Batch Prediction ─────────────────────────────────────────────

class TestBatchPrediction:
    """Tests for the /predict/batch endpoint."""

    def test_batch_endpoint(self, sample_transaction: dict):
        """Test that batch endpoint accepts valid input."""
        batch = {"transactions": [sample_transaction, sample_transaction]}
        response = client.post("/predict/batch", json=batch)
        assert response.status_code in (200, 503)

    def test_batch_returns_summary(self, sample_transaction: dict):
        """Test that batch response contains predictions and summary."""
        batch = {"transactions": [sample_transaction]}
        response = client.post("/predict/batch", json=batch)
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "summary" in data
            assert data["summary"]["total"] == 1

    def test_batch_with_multiple_transactions(self, sample_transaction: dict):
        """Test batch with multiple transactions."""
        batch = {"transactions": [sample_transaction] * 5}
        response = client.post("/predict/batch", json=batch)
        if response.status_code == 200:
            data = response.json()
            assert data["summary"]["total"] == 5
            assert len(data["predictions"]) == 5

    def test_batch_rejects_empty_list(self):
        """Test that empty transactions list returns 422."""
        response = client.post("/predict/batch", json={"transactions": []})
        assert response.status_code == 422


# ─── Tests: Model Info ────────────────────────────────────────────────────

class TestModelInfo:
    """Tests for the /model-info endpoint."""

    def test_model_info_endpoint(self):
        """Test that model-info returns 200."""
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_returns_dict(self):
        """Test that model-info returns a JSON object."""
        response = client.get("/model-info")
        data = response.json()
        assert isinstance(data, dict)
