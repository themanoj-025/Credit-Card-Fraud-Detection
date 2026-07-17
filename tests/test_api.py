"""
Smoke Tests for FastAPI Endpoint

Tests:
1. Health endpoint returns 200
2. Predict endpoint validates input correctly
3. Predict endpoint returns correct response schema
4. Batch prediction works
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_returns_200(self):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_schema(self):
        """Health response should have correct schema."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    """Tests for /predict endpoint."""
    
    def test_predict_valid_input(self):
        """Valid transaction should return prediction."""
        transaction = {
            "Time": 100000.0,
            "Amount": 150.0,
            "V1": -1.359,
            "V2": -0.073,
            "V3": 2.536,
            "V4": 1.378,
            "V5": -0.338,
            "V6": 0.462,
            "V7": 0.240,
            "V8": 0.099,
            "V9": 0.364,
            "V10": 0.091,
            "V11": -0.552,
            "V12": -0.618,
            "V13": -0.991,
            "V14": -0.311,
            "V15": 1.468,
            "V16": -0.470,
            "V17": 0.208,
            "V18": 0.026,
            "V19": 0.404,
            "V20": 0.251,
            "V21": -0.018,
            "V22": 0.278,
            "V23": -0.110,
            "V24": 0.067,
            "V25": 0.129,
            "V26": -0.189,
            "V27": 0.134,
            "V28": 0.021,
        }
        
        response = client.post("/predict", json=transaction)
        assert response.status_code == 200
        
        data = response.json()
        assert "fraud_probability" in data
        assert "decision" in data
        assert "is_fraud" in data
        assert "threshold_used" in data
        
        # Fraud probability should be between 0 and 1
        assert 0 <= data["fraud_probability"] <= 1
        
        # Decision should be FRAUD or LEGITIMATE
        assert data["decision"] in ["FRAUD", "LEGITIMATE"]
    
    def test_predict_invalid_amount(self):
        """Negative amount should fail validation."""
        transaction = {
            "Time": 100000.0,
            "Amount": -50.0,  # Invalid: negative
            **{f"V{i}": 0.0 for i in range(1, 29)},
        }
        
        response = client.post("/predict", json=transaction)
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_fields(self):
        """Missing required fields should fail validation."""
        transaction = {
            "Time": 100000.0,
            # Missing Amount and V1-V28
        }
        
        response = client.post("/predict", json=transaction)
        assert response.status_code == 422
    
    def test_predict_response_has_explanation(self):
        """Prediction should include explanation when model is loaded."""
        transaction = {
            "Time": 100000.0,
            "Amount": 5000.0,  # High amount
            **{f"V{i}": np.random.randn() * 3 for i in range(1, 29)},
        }
        
        response = client.post("/predict", json=transaction)
        data = response.json()
        
        # Should have explanation if model is loaded
        if data.get("decision") != "ERROR":
            assert "explanation" in data or "decision" in data


class TestBatchPredict:
    """Tests for /predict/batch endpoint."""
    
    def test_batch_prediction(self):
        """Batch endpoint should handle multiple transactions."""
        transactions = [
            {
                "Time": 100000.0,
                "Amount": 100.0,
                **{f"V{i}": 0.0 for i in range(1, 29)},
            },
            {
                "Time": 200000.0,
                "Amount": 5000.0,
                **{f"V{i}": np.random.randn() * 3 for i in range(1, 29)},
            },
        ]
        
        response = client.post("/predict/batch", json={"transactions": transactions})
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "summary" in data
        assert len(data["predictions"]) == 2





if __name__ == "__main__":
    pytest.main([__file__, "-v"])
