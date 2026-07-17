"""
Preprocessing Module

Handles train/test split (BEFORE resampling), scaling, and resampling strategies.
CRITICAL: All resampling is done ONLY on the training set to avoid data leakage.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class FraudPreprocessor:
    """
    Preprocessing pipeline for credit card fraud detection.
    
    Key design decisions:
    1. Train/test split FIRST, then resample ONLY training data
    2. StandardScaler fitted ONLY on training data, then applied to test
    3. Multiple resampling strategies for comparison
    """
    
    # Features to scale (Time and Amount)
    SCALE_FEATURES = ['Time', 'Amount']
    
    # PCA features (already scaled by construction)
    PCA_FEATURES = [f'V{i}' for i in range(1, 29)]
    
    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        scale_features: bool = True,
    ):
        """
        Initialize the preprocessor.
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            scale_features: Whether to scale Time/Amount features
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self._is_fitted = False
        
    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets BEFORE any resampling.
        
        Args:
            df: Full DataFrame with features and 'Class' column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        feature_cols = self.PCA_FEATURES + self.SCALE_FEATURES
        X = df[feature_cols].copy()
        y = df['Class'].copy()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,  # Preserve class distribution in both sets
        )
        
        logger.info(f"Train set: {len(X_train)} samples ({y_train.sum()} fraud)")
        logger.info(f"Test set: {len(X_test)} samples ({y_test.sum()} fraud)")
        
        return X_train, X_test, y_train, y_test
    
    def fit_scale(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit scaler on training data and transform it.
        
        Args:
            X_train: Training features
            
        Returns:
            Scaled training features
        """
        if not self.scale_features:
            return X_train
        
        X_scaled = X_train.copy()
        X_scaled[self.SCALE_FEATURES] = self.scaler.fit_transform(
            X_train[self.SCALE_FEATURES]
        )
        self._is_fitted = True
        
        logger.info(f"Scaler fitted on training data: {self.SCALE_FEATURES}")
        return X_scaled
    
    def transform_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted scaler.
        
        Args:
            X: Features to transform
            
        Returns:
            Scaled features
        """
        if not self.scale_features or not self._is_fitted:
            return X
        
        X_scaled = X.copy()
        X_scaled[self.SCALE_FEATURES] = self.scaler.transform(
            X[self.SCALE_FEATURES]
        )
        return X_scaled
    
    def full_preprocess(
        self, df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Complete preprocessing pipeline: split, scale.
        
        Args:
            df: Full DataFrame
            
        Returns:
            Dictionary with preprocessed data splits
        """
        # Step 1: Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Step 2: Scale features (fit on train only)
        X_train_scaled = self.fit_scale(X_train)
        X_test_scaled = self.transform_scale(X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
        }
    
    def save_scaler(self, path: str = "models/scaler.pkl") -> None:
        """Save the fitted scaler to disk."""
        if self.scaler is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved to {path}")
    
    def load_scaler(self, path: str = "models/scaler.pkl") -> None:
        """Load a fitted scaler from disk."""
        self.scaler = joblib.load(path)
        self._is_fitted = True
        logger.info(f"Scaler loaded from {path}")


class Resampler:
    """
    Resampling strategies for handling class imbalance.
    
    CRITICAL: Only fit_resample on training data, never on test data.
    """
    
    STRATEGIES = ['none', 'class_weight', 'random_under', 'smote', 'adasyn', 'smote_tomek']
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def resample(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        strategy: str = 'smote',
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply a resampling strategy to training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            strategy: Resampling strategy name
            
        Returns:
            Tuple of resampled (X, y)
            
        Raises:
            ValueError: If strategy is not recognized
        """
        if strategy == 'none' or strategy == 'class_weight':
            # No resampling needed; class_weight is handled during model training
            return X_train, y_train
        
        if strategy == 'random_under':
            sampler = RandomUnderSampler(
                random_state=self.random_state,
                sampling_strategy=0.5,  # Undersample majority to 2:1 ratio
            )
        elif strategy == 'smote':
            sampler = SMOTE(
                random_state=self.random_state,
                sampling_strategy=0.5,
            )
        elif strategy == 'adasyn':
            sampler = ADASYN(
                random_state=self.random_state,
                sampling_strategy=0.5,
            )
        elif strategy == 'smote_tomek':
            sampler = SMOTETomek(
                random_state=self.random_state,
                sampling_strategy=0.5,
            )
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {self.STRATEGIES}"
            )
        
        logger.info(f"Applying {strategy} resampling...")
        n_before = len(y_train)
        fraud_before = y_train.sum()
        
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        
        logger.info(
            f"  Before: {n_before} samples ({fraud_before} fraud) → "
            f"After: {len(y_res)} samples ({int(y_res.sum())} fraud)"
        )
        
        return X_res, y_res
    
    def compare_strategies(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        strategies: list = None,
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Apply multiple resampling strategies for comparison.
        
        Args:
            X_train: Training features
            y_train: Training labels
            strategies: List of strategy names (default: all non-weight strategies)
            
        Returns:
            Dictionary mapping strategy name to (X, y) tuple
        """
        if strategies is None:
            strategies = ['none', 'random_under', 'smote', 'adasyn', 'smote_tomek']
        
        results = {}
        for strat in strategies:
            results[strat] = self.resample(X_train, y_train, strat)
        
        return results


def get_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Compute class weights for imbalanced classification.
    
    Args:
        y: Target labels
        
    Returns:
        Dictionary of {class: weight}
    """
    n = len(y)
    n_classes = y.nunique()
    class_counts = y.value_counts().to_dict()
    
    weights = {}
    for cls, count in class_counts.items():
        weights[cls] = n / (n_classes * count)
    
    logger.info(f"Class weights: {weights}")
    return weights


if __name__ == "__main__":
    # Quick demo
    from data_loader import load_data
    
    df, stats = load_data()
    print(f"\nDataset stats: {stats}")
    
    preprocessor = FraudPreprocessor()
    data = preprocessor.full_preprocess(df)
    
    print(f"\nTrain: {data['X_train'].shape}, Test: {data['X_test'].shape}")
    print(f"Train fraud rate: {data['y_train'].mean():.4%}")
    print(f"Test fraud rate: {data['y_test'].mean():.4%}")
    
    # Compare resampling strategies
    resampler = Resampler()
    comparisons = resampler.compare_strategies(data['X_train'], data['y_train'])
    
    for strat, (X_r, y_r) in comparisons.items():
        print(f"\n{strat}: {len(X_r)} samples, {int(y_r.sum())} fraud ({y_r.mean():.2%})")
