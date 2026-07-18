"""
Tests for the Anomaly Detection module.

Verifies anomaly scores separate obviously-anomalous points from normal ones,
output shapes are correct, and edge cases are handled.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fraudshield.models.anomaly import IsolationForestDetector


# ─── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def normal_data() -> pd.DataFrame:
    """Normal (non-anomalous) data points."""
    np.random.seed(42)
    return pd.DataFrame(np.random.randn(200, 10))


@pytest.fixture
def anomalous_point() -> pd.DataFrame:
    """A clearly anomalous point far from the normal distribution."""
    return pd.DataFrame([[10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0]])


# ─── Tests: IsolationForestDetector ─────────────────────────────────────

class TestIsolationForestDetector:
    """Tests for IsolationForestDetector."""

    def test_fit_without_labels(self, normal_data):
        """Test that fit works without y_train (unsupervised)."""
        detector = IsolationForestDetector()
        detector.fit(normal_data)
        assert detector.model is not None

    def test_fit_on_legit_only(self, normal_data):
        """Test that fit with labels filters to legit transactions."""
        y = pd.Series(np.zeros(200))
        y.iloc[:5] = 1  # 5 fraud samples
        detector = IsolationForestDetector()
        detector.fit(normal_data, y)
        assert detector.model is not None

    def test_predict_returns_array(self, normal_data, anomalous_point):
        """Test that predict returns numpy array with -1/1 values."""
        detector = IsolationForestDetector(contamination=0.1)
        detector.fit(normal_data)
        preds = detector.predict(anomalous_point)
        assert isinstance(preds, np.ndarray)
        assert preds[0] in (-1, 1)  # -1 = anomaly, 1 = normal

    def test_predict_before_fit_raises(self):
        """Test that predict raises error before fit."""
        detector = IsolationForestDetector()
        with pytest.raises(ValueError):
            detector.predict(pd.DataFrame([[0.0, 0.0]]))

    def test_score_returns_float_array(self, normal_data):
        """Test that score returns an array of floats."""
        detector = IsolationForestDetector(contamination=0.1)
        detector.fit(normal_data)
        scores = detector.score(normal_data.head(5))
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 5
        assert scores.dtype in (np.float64, np.float32)

    def test_anomalous_point_lower_score(self, normal_data, anomalous_point):
        """Test that anomalous points get lower (more negative) anomaly scores."""
        detector = IsolationForestDetector(contamination=0.1)
        detector.fit(normal_data)
        normal_score = detector.score(normal_data.head(1))[0]
        anom_score = detector.score(anomalous_point)[0]
        # Anomalous points should have lower scores in Isolation Forest
        # (more negative = more anomalous)
        assert anom_score < normal_score


# ─── Tests: Fraud Probability Conversion ────────────────────────────────

class TestFraudProbability:
    """Tests for predict_proba_as_fraud method."""

    def test_proba_returns_values_in_range(self, normal_data):
        """Test that fraud probabilities are in [0, 1] range."""
        detector = IsolationForestDetector(contamination=0.1)
        detector.fit(normal_data)
        probas = detector.predict_proba_as_fraud(normal_data)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_anomaly_higher_proba(self, normal_data, anomalous_point):
        """Test that anomalous points get higher or equal fraud probability."""
        detector = IsolationForestDetector(contamination=0.1)
        detector.fit(normal_data)
        normal_proba = detector.predict_proba_as_fraud(normal_data.head(1))[0]
        anom_proba = detector.predict_proba_as_fraud(anomalous_point)[0]
        # Anomalous points should be at least as high as normal points
        assert anom_proba >= normal_proba


# ─── Tests: Edge Cases ───────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case tests for anomaly detection."""

    def test_single_feature(self):
        """Test with single feature dimension."""
        X = pd.DataFrame(np.random.randn(100, 1))
        detector = IsolationForestDetector()
        detector.fit(X)
        scores = detector.score(pd.DataFrame([[5.0]]))
        assert len(scores) == 1

    def test_very_small_dataset(self):
        """Test with very small dataset."""
        X = pd.DataFrame(np.random.randn(10, 5))
        detector = IsolationForestDetector()
        detector.fit(X)
        preds = detector.predict(pd.DataFrame([[0.0] * 5]))
        assert len(preds) == 1


# ─── Tests: AutoencoderDetector ─────────────────────────────────────────

class TestAutoencoderDetector:
    """Tests for AutoencoderDetector (if TensorFlow is available)."""

    def test_build_model_structure(self):
        """Test that the model builds with correct input/output dimensions."""
        try:
            from src.fraudshield.models.anomaly import AutoencoderDetector

            detector = AutoencoderDetector(encoding_dim=4, epochs=2, batch_size=8)
            detector._build_model(10)
            assert detector.model is not None
            assert detector.model.input_shape[-1] == 10
            assert detector.model.output_shape[-1] == 10
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_fit_and_predict(self):
        """Test that fit trains and predict returns scores."""
        try:
            from src.fraudshield.models.anomaly import AutoencoderDetector

            np.random.seed(42)
            X = pd.DataFrame(np.random.randn(100, 10))
            detector = AutoencoderDetector(encoding_dim=4, epochs=2, batch_size=8)
            detector.fit(X)
            scores = detector.score(pd.DataFrame(np.random.randn(5, 10)))
            assert len(scores) == 5
            assert np.all(scores >= 0)
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_score_before_fit_raises(self):
        """Test that score raises error before fit."""
        try:
            from src.fraudshield.models.anomaly import AutoencoderDetector

            detector = AutoencoderDetector()
            with pytest.raises(ValueError):
                detector.score(pd.DataFrame([[0.0] * 10]))
        except ImportError:
            pytest.skip("TensorFlow not installed")
