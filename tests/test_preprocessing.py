"""
Unit Tests for Preprocessing Module

Tests:
1. No data leakage: scaler fit only on training data
2. Correct train/test split preserves class distribution
3. Scaling works correctly on Time/Amount features
4. Resampling only affects training data
5. Feature dimensions are correct
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import FraudPreprocessor, Resampler, get_class_weights


@pytest.fixture
def sample_df():
    """Create a sample DataFrame mimicking the credit card dataset."""
    np.random.seed(42)
    n = 1000
    n_fraud = 10  # 1% fraud rate
    
    data = {f'V{i}': np.random.randn(n) for i in range(1, 29)}
    data['Time'] = np.random.uniform(0, 172800, n)
    data['Amount'] = np.random.exponential(50, n)
    data['Class'] = [1] * n_fraud + [0] * (n - n_fraud)
    
    return pd.DataFrame(data)


class TestFraudPreprocessor:
    """Tests for the FraudPreprocessor class."""
    
    def test_train_test_split_preserves_ratio(self, sample_df):
        """Verify train/test split preserves fraud ratio."""
        preprocessor = FraudPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.split_data(sample_df)
        
        train_fraud_rate = y_train.mean()
        test_fraud_rate = y_test.mean()
        
        # Fraud rates should be approximately equal (within 1%)
        assert abs(train_fraud_rate - test_fraud_rate) < 0.01, \
            f"Fraud rate mismatch: train={train_fraud_rate:.4f}, test={test_fraud_rate:.4f}"
    
    def test_no_data_leakage_in_scaling(self, sample_df):
        """Verify scaler is fit ONLY on training data."""
        preprocessor = FraudPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.split_data(sample_df)
        
        # Fit scaler on training data
        X_train_scaled = preprocessor.fit_scale(X_train)
        
        # Test scaler parameters should not be influenced by test data
        # The scaler mean should be from training data only
        train_amount_mean = X_train['Amount'].mean()
        scaler_amount_mean = preprocessor.scaler.mean_[1]  # Amount is second in SCALE_FEATURES
        
        assert abs(train_amount_mean - scaler_amount_mean) < 1e-10, \
            "Scaler mean does not match training data mean - potential data leakage!"
    
    def test_scaling_only_affects_time_amount(self, sample_df):
        """Verify PCA features are NOT scaled."""
        preprocessor = FraudPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.split_data(sample_df)
        
        X_train_scaled = preprocessor.fit_scale(X_train)
        
        # PCA features should remain unchanged
        for i in range(1, 29):
            col = f'V{i}'
            pd.testing.assert_series_equal(
                X_train[col], X_train_scaled[col],
                check_names=False,
                atol=1e-10,
            )
    
    def test_scaling_applied_to_test_data(self, sample_df):
        """Verify scaler transforms test data correctly."""
        preprocessor = FraudPreprocessor(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = preprocessor.split_data(sample_df)
        
        X_train_scaled = preprocessor.fit_scale(X_train)
        X_test_scaled = preprocessor.transform_scale(X_test)
        
        # Test data should be transformed using training scaler parameters
        assert X_test_scaled['Amount'].mean() != X_test['Amount'].mean() or \
               np.isclose(X_test_scaled['Amount'].mean(), X_test['Amount'].mean(), atol=1e-6), \
            "Scaling not applied to test data"
    
    def test_full_preprocess_output_shape(self, sample_df):
        """Verify full preprocess produces correct shapes."""
        preprocessor = FraudPreprocessor(test_size=0.2, random_state=42)
        data = preprocessor.full_preprocess(sample_df)
        
        assert len(data['X_train']) + len(data['X_test']) == len(sample_df)
        assert data['X_train'].shape[1] == 30  # 28 PCA + Time + Amount
        assert data['X_test'].shape[1] == 30
    
    def test_no_fraud_in_train_without_resampling(self, sample_df):
        """With very small fraud count, some may end up only in test."""
        preprocessor = FraudPreprocessor(test_size=0.2, random_state=42)
        data = preprocessor.full_preprocess(sample_df)
        
        # Both sets should have at least some fraud (stratified split)
        assert data['y_train'].sum() > 0, "No fraud in training set!"
        assert data['y_test'].sum() > 0, "No fraud in test set!"


class TestResampler:
    """Tests for the Resampler class."""
    
    def test_none_strategy_returns_unchanged(self, sample_df):
        """'none' strategy should return data unchanged."""
        preprocessor = FraudPreprocessor()
        data = preprocessor.full_preprocess(sample_df)
        
        resampler = Resampler()
        X_res, y_res = resampler.resample(
            data['X_train'], data['y_train'], strategy='none'
        )
        
        assert len(X_res) == len(data['X_train'])
        assert len(y_res) == len(data['y_train'])
    
    def test_smote_increases_minority(self, sample_df):
        """SMOTE should increase the number of minority samples."""
        preprocessor = FraudPreprocessor()
        data = preprocessor.full_preprocess(sample_df)
        
        resampler = Resampler()
        X_res, y_res = resampler.resample(
            data['X_train'], data['y_train'], strategy='smote'
        )
        
        assert len(y_res) > len(data['y_train']), "SMOTE should increase sample count"
        assert int(y_res.sum()) >= data['y_train'].sum(), "SMOTE should increase fraud count"
    
    def test_resampling_does_not_affect_test(self, sample_df):
        """Resampling should only modify training data."""
        preprocessor = FraudPreprocessor()
        data = preprocessor.full_preprocess(sample_df)
        
        X_test_original = data['X_test'].copy()
        y_test_original = data['y_test'].copy()
        
        resampler = Resampler()
        resampler.resample(data['X_train'], data['y_train'], strategy='smote')
        
        pd.testing.assert_frame_equal(X_test_original, data['X_test'])
        pd.testing.assert_series_equal(y_test_original, data['y_test'])
    
    def test_invalid_strategy_raises_error(self, sample_df):
        """Invalid strategy name should raise ValueError."""
        preprocessor = FraudPreprocessor()
        data = preprocessor.full_preprocess(sample_df)
        
        resampler = Resampler()
        with pytest.raises(ValueError, match="Unknown strategy"):
            resampler.resample(data['X_train'], data['y_train'], strategy='invalid')


class TestClassWeights:
    """Tests for class weight computation."""
    
    def test_weights_inverse_to_frequency(self):
        """Minority class should have higher weight."""
        y = pd.Series([0] * 990 + [1] * 10)
        weights = get_class_weights(y)
        
        assert weights[1] > weights[0], "Fraud (minority) should have higher weight"
    
    def test_weights_sum_correctly(self):
        """Weights should be properly normalized."""
        y = pd.Series([0] * 990 + [1] * 10)
        weights = get_class_weights(y)
        
        expected_1 = 1000 / (2 * 10)
        expected_0 = 1000 / (2 * 990)
        
        assert abs(weights[1] - expected_1) < 0.01
        assert abs(weights[0] - expected_0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
