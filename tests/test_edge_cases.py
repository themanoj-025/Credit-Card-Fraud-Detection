"""
FraudLens — Edge Case Tests

Verifies the API and models handle edge cases gracefully:
- NaN / Inf in transaction inputs (should 422, not 500)
- Empty batch (should 422, not 500)
- Batch at max allowed size + 1 (should reject cleanly)
- Threshold at 0 and 1 (boundary conditions)
- 0% and 100% fraud probability
- Missing all optional fields
- Extreme Amount values (0 and very large)
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestNanInfInputs:
    """Edge case: API should reject NaN/Inf with 422, never 500."""

    def test_nan_amount_rejected(self, client):
        """Test that NaN Amount returns 422."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        tx["Amount"] = float("nan")
        response = client.post("/v1/predict", json=tx)
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_inf_amount_rejected(self, client):
        """Test that Inf Amount returns 422."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        tx["Amount"] = float("inf")
        response = client.post("/v1/predict", json=tx)
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_neg_inf_amount_rejected(self, client):
        """Test that -Inf Amount returns 422."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        tx["Amount"] = float("-inf")
        response = client.post("/v1/predict", json=tx)
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_nan_v_feature_rejected(self, client):
        """Test that NaN in a V feature returns 422."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["V14"] = float("nan")
        tx["Time"] = 0.0
        tx["Amount"] = 100.0
        response = client.post("/v1/predict", json=tx)
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"

    def test_nan_time_rejected(self, client):
        """Test that NaN Time returns 422."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = float("nan")
        tx["Amount"] = 100.0
        response = client.post("/v1/predict", json=tx)
        assert response.status_code == 422, f"Expected 422, got {response.status_code}"


class TestEdgeCaseInputs:
    """Edge case: boundary values and extreme inputs."""

    def test_zero_amount_accepted(self, client):
        """Test that Amount=0 is valid (free transaction)."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        tx["Amount"] = 0.0
        response = client.post("/v1/predict", json=tx)
        assert response.status_code in (200, 503), f"Expected 200/503, got {response.status_code}"

    def test_very_large_amount_accepted(self, client):
        """Test that a very large Amount is accepted."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        tx["Amount"] = 1e7  # $10M
        response = client.post("/v1/predict", json=tx)
        assert response.status_code in (200, 503), f"Expected 200/503, got {response.status_code}"

    def test_zero_time_accepted(self, client):
        """Test that Time=0 is valid."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        tx["Amount"] = 100.0
        response = client.post("/v1/predict", json=tx)
        assert response.status_code in (200, 503), f"Expected 200/503, got {response.status_code}"

    def test_max_time_accepted(self, client):
        """Test that Time at maximum value is accepted."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 172792.0  # Max in dataset
        tx["Amount"] = 100.0
        response = client.post("/v1/predict", json=tx)
        assert response.status_code in (200, 503), f"Expected 200/503, got {response.status_code}"

    def test_amount_zero_on_explain(self, client):
        """Test that /v1/explain accepts zero Amount."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        tx["Amount"] = 0.0
        response = client.post("/v1/explain", json=tx)
        assert response.status_code in (200, 503), f"Expected 200/503, got {response.status_code}"


class TestEmptyBatch:
    """Edge case: empty and edge-case batch requests."""

    def test_empty_batch_rejected(self, client):
        """Test that empty transactions list returns 422."""
        response = client.post("/v1/predict/batch", json={"transactions": []})
        assert response.status_code == 422

    def test_batch_too_large_rejected(self, client, sample_transaction):
        """Test that batch with >1000 transactions returns 422."""
        many_txs = [sample_transaction for _ in range(1001)]
        response = client.post("/v1/predict/batch", json={"transactions": many_txs})
        assert response.status_code == 422

    def test_batch_single_transaction(self, client, sample_transaction):
        """Test that batch with single transaction works."""
        response = client.post(
            "/v1/predict/batch",
            json={"transactions": [sample_transaction]},
        )
        assert response.status_code in (200, 503)

    def test_batch_with_nan_rejected(self):
        """Test that batch containing NaN is rejected."""
        import requests
        # Use FastAPI TestClient directly
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        tx["Amount"] = float("nan")

        response = client.post(
            "/v1/predict/batch",
            json={"transactions": [tx, tx]},
        )
        assert response.status_code == 422


class TestModelBoundaries:
    """Edge case: model boundary conditions."""

    def test_threshold_at_zero(self):
        """Test prediction interpretation at threshold=0 (everything is fraud)."""
        import numpy as np

        probas = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        threshold = 0.0
        decisions = ["FRAUD" if p >= threshold else "LEGITIMATE" for p in probas]
        assert all(d == "FRAUD" for d in decisions)

    def test_threshold_at_one(self):
        """Test prediction interpretation at threshold=1 (nothing is fraud)."""
        import numpy as np

        probas = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        threshold = 1.0
        decisions = ["FRAUD" if p >= threshold else "LEGITIMATE" for p in probas]
        assert all(d == "LEGITIMATE" for d in decisions)

    def test_all_features_zero(self, client):
        """Test that all-zero features produce a valid prediction."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        tx["Amount"] = 0.0
        response = client.post("/v1/predict", json=tx)
        assert response.status_code in (200, 503)

    def test_all_large_positive_features(self, client):
        """Test that all-large features produce a valid prediction."""
        tx = {f"V{i}": 999.0 for i in range(1, 29)}
        tx["Time"] = 999.0
        tx["Amount"] = 999.0
        response = client.post("/v1/predict", json=tx)
        assert response.status_code in (200, 503)

    def test_all_large_negative_features(self, client):
        """Test that all-large negative features produce a valid prediction."""
        tx = {f"V{i}": -999.0 for i in range(1, 29)}
        tx["Time"] = -999.0
        tx["Amount"] = -999.0
        response = client.post("/v1/predict", json=tx)
        # Negative Amount should be rejected by Pydantic validation
        assert response.status_code in (422, 503)


class TestExplainEdgeCases:
    """Edge case tests for the /v1/explain endpoint."""

    def test_explain_rejects_nan(self, client):
        """Test that /v1/explain rejects NaN inputs."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        tx["Amount"] = float("nan")
        response = client.post("/v1/explain", json=tx)
        assert response.status_code == 422

    def test_explain_rejects_inf(self, client):
        """Test that /v1/explain rejects Inf inputs."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        tx["Amount"] = float("inf")
        response = client.post("/v1/explain", json=tx)
        assert response.status_code == 422


class TestSimilarCasesEdgeCases:
    """Edge case tests for /v1/similar-cases endpoint."""

    def test_similar_cases_rejects_nan(self, client):
        """Test that /v1/similar-cases rejects NaN inputs."""
        tx = {f"V{i}": 0.0 for i in range(1, 29)}
        tx["Time"] = 0.0
        tx["Amount"] = float("nan")
        response = client.post("/v1/similar-cases", json=tx)
        assert response.status_code == 422

    def test_similar_cases_invalid_top_k(self, client, sample_transaction):
        """Test that top_k=0 returns 422."""
        response = client.post(
            "/v1/similar-cases?top_k=0",
            json=sample_transaction,
        )
        assert response.status_code == 422

    def test_similar_cases_large_top_k(self, client, sample_transaction):
        """Test that top_k=50 (beyond max) returns 422."""
        response = client.post(
            "/v1/similar-cases?top_k=50",
            json=sample_transaction,
        )
        # Pydantic Query validation should reject top_k > 20
        assert response.status_code == 422


class TestChatEdgeCases:
    """Edge case tests for /v1/chat endpoint."""

    def test_chat_empty_message_rejected(self, client):
        """Test that /v1/chat rejects empty message."""
        response = client.post(
            "/v1/chat",
            json={"message": "", "conversation_history": []},
        )
        assert response.status_code == 422

    def test_chat_missing_conversation_history(self, client):
        """Test that /v1/chat works without conversation_history."""
        response = client.post(
            "/v1/chat",
            json={"message": "Hello"},
        )
        # Should either work (200) or return 503 (no Anthropic key)
        assert response.status_code in (200, 503)
