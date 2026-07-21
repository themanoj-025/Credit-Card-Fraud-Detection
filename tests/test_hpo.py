"""
FraudLens — Hyperparameter Optimization Tests (HPO)

Tests for HyperparameterOptimizer in models/hpo.py.
Uses mocked optuna to avoid expensive real optimization runs.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.fraudlens.models.hpo import HyperparameterOptimizer


@pytest.fixture
def sample_data():
    """Create a small labeled dataset for HPO tests."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(
        {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    )
    X["Time"] = np.random.uniform(0, 172800, n)
    X["Amount"] = np.random.exponential(100, n)
    y = pd.Series(np.random.choice([0, 1], n, p=[0.95, 0.05]))
    return X, y


class TestHyperparameterOptimizer:
    """Tests for HyperparameterOptimizer."""

    def test_init_defaults(self):
        """Test default initialization parameters."""
        optimizer = HyperparameterOptimizer()
        assert optimizer.n_trials == 50
        assert optimizer.cv_folds == 3
        assert optimizer.random_state == 42
        assert optimizer.timeout_seconds is None
        assert optimizer.best_params == {}
        assert optimizer.best_score == 0.0
        assert optimizer.study is None

    def test_init_custom(self):
        """Test custom initialization parameters."""
        optimizer = HyperparameterOptimizer(
            n_trials=10, cv_folds=5, random_state=123, timeout_seconds=60
        )
        assert optimizer.n_trials == 10
        assert optimizer.cv_folds == 5
        assert optimizer.random_state == 123
        assert optimizer.timeout_seconds == 60

    def test_compute_scale_pos_weight_balanced(self):
        """Test scale_pos_weight with equal classes."""
        optimizer = HyperparameterOptimizer()
        y = pd.Series([0, 0, 0, 1, 1, 1])
        weight = optimizer._compute_scale_pos_weight(y)
        assert weight == 1.0

    def test_compute_scale_pos_weight_imbalanced(self):
        """Test scale_pos_weight with imbalanced classes."""
        optimizer = HyperparameterOptimizer()
        y = pd.Series([0] * 95 + [1] * 5)
        weight = optimizer._compute_scale_pos_weight(y)
        assert weight == 19.0

    def test_compute_scale_pos_weight_no_positives(self):
        """Test scale_pos_weight with no positive class (edge case)."""
        optimizer = HyperparameterOptimizer()
        y = pd.Series([0] * 100)
        weight = optimizer._compute_scale_pos_weight(y)
        assert weight == 1.0

    def test_cv_score(self, sample_data):
        """Test _cv_score returns a float between 0 and 1."""
        X, y = sample_data

        # Use a small subset for speed
        X_small = X.iloc[:50]
        y_small = y.iloc[:50]

        from sklearn.linear_model import LogisticRegression

        optimizer = HyperparameterOptimizer(n_trials=2, cv_folds=2)
        score = optimizer._cv_score(
            X_small, y_small, LogisticRegression, {"random_state": 42, "max_iter": 100}
        )

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_tune_xgboost_optuna_not_installed(self, sample_data):
        """Test tune_xgboost fallback when optuna is not installed."""
        X, y = sample_data

        with patch("src.fraudlens.models.hpo.logger") as mock_logger:
            with patch.dict("sys.modules", {"optuna": None}):
                import importlib

                import src.fraudlens.models.hpo as hpo_module

                importlib.reload(hpo_module)
                optimizer = hpo_module.HyperparameterOptimizer(n_trials=2)

                params = optimizer.tune_xgboost(X, y)
                assert params["n_estimators"] == 200
                assert params["max_depth"] == 6
                assert params["learning_rate"] == 0.1

    @patch("src.fraudlens.models.hpo.optuna")
    def test_tune_xgboost_success(self, mock_optuna, sample_data):
        """Test tune_xgboost with mocked optuna returns best params."""
        X, y = sample_data

        # Mock optuna study
        mock_study = MagicMock()
        mock_study.best_params = {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "min_child_weight": 3,
            "gamma": 1.0,
        }
        mock_study.best_value = 0.85
        mock_study.best_trials = []

        mock_optuna.create_study.return_value = mock_study

        optimizer = HyperparameterOptimizer(n_trials=2, cv_folds=2)
        params = optimizer.tune_xgboost(X, y)

        assert params["n_estimators"] == 300
        assert params["max_depth"] == 8
        assert params["learning_rate"] == 0.05
        assert params["scale_pos_weight"] > 0
        assert params["random_state"] == 42
        assert optimizer.best_score == 0.85

    @patch("src.fraudlens.models.hpo.optuna")
    def test_tune_lightgbm_success(self, mock_optuna, sample_data):
        """Test tune_lightgbm with mocked optuna returns best params."""
        X, y = sample_data

        # Mock optuna study
        mock_study = MagicMock()
        mock_study.best_params = {
            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "min_child_samples": 20,
            "num_leaves": 31,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
        mock_study.best_value = 0.82
        mock_study.best_trials = []

        mock_optuna.create_study.return_value = mock_study

        optimizer = HyperparameterOptimizer(n_trials=2, cv_folds=2)
        params = optimizer.tune_lightgbm(X, y)

        assert params["n_estimators"] == 250
        assert params["is_unbalance"] is True
        assert params["random_state"] == 42
        assert optimizer.best_score == 0.82

    def test_get_trials_dataframe_no_study(self):
        """Test get_trials_dataframe returns None when no study exists."""
        optimizer = HyperparameterOptimizer()
        assert optimizer.get_trials_dataframe() is None

    @patch("src.fraudlens.models.hpo.optuna")
    def test_get_trials_dataframe_with_study(self, mock_optuna, sample_data):
        """Test get_trials_dataframe returns DataFrame when study exists."""
        X, y = sample_data

        mock_study = MagicMock()
        mock_study.best_params = {"n_estimators": 300, "max_depth": 8}
        mock_study.best_value = 0.85
        mock_study.best_trials = []
        mock_study.trials_dataframe.return_value = pd.DataFrame(
            {"value": [0.8, 0.85, 0.82]}
        )

        mock_optuna.create_study.return_value = mock_study

        optimizer = HyperparameterOptimizer(n_trials=2, cv_folds=2)
        optimizer.tune_xgboost(X, y)

        df = optimizer.get_trials_dataframe()
        assert df is not None
        assert len(df) == 3
