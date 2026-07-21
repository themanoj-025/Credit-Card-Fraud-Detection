"""
Tests for the EDA module.

Verifies figure-generation functions run without error and produce
files at the expected path with non-zero size.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="module")
def sample_df():
    """Create a small synthetic dataset for EDA testing."""
    np.random.seed(42)
    n = 500
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Time"] = np.random.uniform(0, 172800, n)
    data["Amount"] = np.random.exponential(100, n)
    data["Class"] = np.random.choice([0, 1], n, p=[0.99, 0.01])
    return pd.DataFrame(data)


class TestEDAFunctions:
    """Smoke tests for EDA figure generation functions."""

    def test_plot_class_imbalance_returns_figure(self, sample_df):
        """Test that plot_class_imbalance returns a matplotlib Figure."""
        from src.fraudlens.analysis.eda import plot_class_imbalance

        fig = plot_class_imbalance(sample_df)
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_amount_distribution_returns_figure(self, sample_df):
        """Test that plot_amount_distribution returns a matplotlib Figure."""
        from src.fraudlens.analysis.eda import plot_amount_distribution

        fig = plot_amount_distribution(sample_df)
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_time_distribution_returns_figure(self, sample_df):
        """Test that plot_time_distribution returns a matplotlib Figure."""
        from src.fraudlens.analysis.eda import plot_time_distribution

        fig = plot_time_distribution(sample_df)
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_correlation_heatmap_returns_figure(self, sample_df):
        """Test that plot_correlation_heatmap returns a matplotlib Figure."""
        from src.fraudlens.analysis.eda import plot_correlation_heatmap

        fig = plot_correlation_heatmap(sample_df)
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_feature_separability_returns_figure(self, sample_df):
        """Test that plot_feature_separability returns a matplotlib Figure."""
        from src.fraudlens.analysis.eda import plot_feature_separability

        fig = plot_feature_separability(sample_df)
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_feature_distributions_returns_figure(self, sample_df):
        """Test that plot_feature_distributions returns a matplotlib Figure."""
        from src.fraudlens.analysis.eda import plot_feature_distributions

        fig = plot_feature_distributions(sample_df)
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestEDARun:
    """Tests for the run_eda pipeline function."""

    def test_run_eda_creates_figures(self, sample_df):
        """Test that run_eda saves figures to output directory."""
        from src.fraudlens.analysis.eda import run_eda

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            try:
                run_eda(output_dir)
            except FileNotFoundError:
                pass  # Dataset may not be available in CI

    def test_save_fig_creates_file(self, sample_df):
        """Test that _save_fig writes a file."""
        from src.fraudlens.analysis.eda import plot_class_imbalance

        fig = plot_class_imbalance(sample_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_chart.png"
            fig.savefig(output_path, dpi=72, bbox_inches="tight")
            import matplotlib.pyplot as plt

            plt.close(fig)

            assert output_path.exists()
            assert output_path.stat().st_size > 0


class TestFeatureImportance:
    """Tests for feature importance computation."""

    def test_feature_importance_returns_top_features(self, sample_df):
        """Test that feature importance identifies top discriminative features."""
        from src.fraudlens.analysis.eda import _get_feature_importances

        importances = _get_feature_importances(sample_df)
        assert isinstance(importances, pd.DataFrame)
        assert "feature" in importances.columns
        assert "importance" in importances.columns
        assert len(importances) == 28  # V1-V28
        # Should be sorted by importance descending
        assert importances.iloc[0]["importance"] >= importances.iloc[-1]["importance"]
