"""
Tests for the model selection module.

Verifies auto-selection logic picks the correct model,
raises appropriate errors, and handles edge cases.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fraudlens.models.model_selection import ModelSelector

# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def comparison_df() -> pd.DataFrame:
    """A model comparison table for testing."""
    return pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
            "PR-AUC": [0.72, 0.84, 0.88],
            "ROC-AUC": [0.97, 0.98, 0.97],
            "F1": [0.62, 0.56, 0.71],
            "Net Benefit ($)": [12140, 12130, 12445],
        }
    )


@pytest.fixture
def trained_models() -> dict:
    """Dict of trained model objects for testing."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=100, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42),
        "XGBoost": LogisticRegression(max_iter=100, random_state=42),  # Placeholder
    }


# ─── Tests ────────────────────────────────────────────────────────────────


class TestModelSelector:
    """Tests for ModelSelector class."""

    def test_selects_highest_pr_auc(
        self, comparison_df: pd.DataFrame, trained_models: dict
    ):
        """Test that selector picks the model with highest PR-AUC."""
        selector = ModelSelector(metric="PR-AUC", higher_is_better=True)
        result = selector.select(comparison_df, trained_models)

        assert result["best_model_name"] == "XGBoost"
        assert result["metric_value"] == 0.88

    def test_selects_lowest_value(
        self, comparison_df: pd.DataFrame, trained_models: dict
    ):
        """Test with higher_is_better=False picks the lowest value."""
        selector = ModelSelector(metric="PR-AUC", higher_is_better=False)
        result = selector.select(comparison_df, trained_models)

        assert result["best_model_name"] == "Logistic Regression"

    def test_ranking_is_sorted(self, comparison_df: pd.DataFrame, trained_models: dict):
        """Test that ranking DataFrame is sorted by metric."""
        selector = ModelSelector(metric="PR-AUC")
        result = selector.select(comparison_df, trained_models)

        ranking = result["ranking"]
        assert ranking["PR-AUC"].iloc[0] >= ranking["PR-AUC"].iloc[-1]

    def test_raises_error_for_missing_metric(
        self, comparison_df: pd.DataFrame, trained_models: dict
    ):
        """Test that missing metric column raises ValueError."""
        selector = ModelSelector(metric="non_existent")
        with pytest.raises(ValueError):
            selector.select(comparison_df, trained_models)

    def test_raises_error_for_missing_model(
        self, comparison_df: pd.DataFrame, trained_models: dict
    ):
        """Test that missing model in trained_models raises KeyError."""
        bad_models = {"Wrong Model": LogisticRegression()}
        selector = ModelSelector(metric="PR-AUC")
        with pytest.raises(KeyError):
            selector.select(comparison_df, bad_models)

    def test_reasoning_includes_winner(
        self, comparison_df: pd.DataFrame, trained_models: dict
    ):
        """Test that reasoning string mentions the winner."""
        selector = ModelSelector(metric="PR-AUC")
        result = selector.select(comparison_df, trained_models)

        assert "XGBoost" in result["reasoning"]
        assert "PR-AUC" in result["reasoning"]
        assert "0.88" in result["reasoning"]

    def test_get_summary_before_select(self):
        """Test that getting summary before selection returns appropriate message."""
        selector = ModelSelector()
        assert "No model selected" in selector.get_selection_summary()

    def test_save_before_select_raises_error(self):
        """Test that saving before selection raises error."""
        selector = ModelSelector()
        with pytest.raises(ValueError):
            selector.save_best_model()

    def test_single_model_always_selected(self, trained_models: dict):
        """Test that with one model, it's always selected."""
        single_df = pd.DataFrame({"Model": ["XGBoost"], "PR-AUC": [0.88]})
        selector = ModelSelector(metric="PR-AUC")
        result = selector.select(single_df, {"XGBoost": trained_models["XGBoost"]})

        assert result["best_model_name"] == "XGBoost"
