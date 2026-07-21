"""
Training Module

Trains multiple fraud detection models and supports MLflow experiment tracking.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class FraudTrainer:
    """
    Train and compare multiple fraud detection models.
    
    Models:
    - Logistic Regression (baseline, interpretable)
    - Random Forest (tree ensemble)
    - XGBoost (gradient boosting)
    - LightGBM (gradient boosting, faster)
    - Isolation Forest (unsupervised anomaly detection)
    """
    
    # Default model configurations
    DEFAULT_CONFIGS = {
        'logistic_regression': {
            'model_class': LogisticRegression,
            'params': {
                'max_iter': 1000,
                'random_state': 42,
                'class_weight': 'balanced',  # Handles imbalance directly
                'solver': 'lbfgs',
            },
            'uses_class_weight': True,
        },
        'random_forest': {
            'model_class': RandomForestClassifier,
            'params': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1,
            },
            'uses_class_weight': True,
        },
        'xgboost': {
            'model_class': XGBClassifier,
            'params': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'scale_pos_weight': 1,  # Will be set dynamically
                'random_state': 42,
                'eval_metric': 'aucpr',
                'use_label_encoder': False,
                'n_jobs': -1,
            },
            'uses_class_weight': False,  # Uses scale_pos_weight instead
        },
        'lightgbm': {
            'model_class': LGBMClassifier,
            'params': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'is_unbalance': True,
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1,
            },
            'uses_class_weight': False,  # Uses is_unbalance
        },
    }
    
    def __init__(
        self,
        models_to_train: list = None,
        custom_configs: dict = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            models_to_train: List of model names to train
            custom_configs: Custom model configurations to override defaults
        """
        self.configs = {**self.DEFAULT_CONFIGS}
        if custom_configs:
            self.configs.update(custom_configs)
        
        if models_to_train is None:
            self.models_to_train = list(self.configs.keys())
        else:
            self.models_to_train = [
                m for m in models_to_train if m in self.configs
            ]
        
        self.trained_models = {}
        self.training_results = {}
    
    def _compute_scale_pos_weight(self, y: pd.Series) -> float:
        """Compute scale_pos_weight for XGBoost based on class imbalance."""
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        return n_neg / n_pos if n_pos > 0 else 1
    
    def train_model(
        self,
        name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Any:
        """
        Train a single model.
        
        Args:
            name: Model name
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        config = self.configs[name]
        params = config['params'].copy()
        
        # Dynamically set scale_pos_weight for XGBoost
        if name == 'xgboost':
            params['scale_pos_weight'] = self._compute_scale_pos_weight(y_train)
        
        logger.info(f"Training {name}...")
        start_time = time.time()
        
        model = config['model_class'](**params)
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        logger.info(f"  {name} trained in {train_time:.2f}s")
        
        self.trained_models[name] = model
        self.training_results[name] = {
            'train_time': train_time,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
        }
        
        return model
    
    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[str, Any]:
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary of trained models
        """
        logger.info(f"Training {len(self.models_to_train)} models...")
        
        for name in self.models_to_train:
            self.train_model(name, X_train, y_train)
        
        logger.info(f"All models trained: {list(self.trained_models.keys())}")
        return self.trained_models
    
    def get_model(self, name: str) -> Any:
        """Get a trained model by name."""
        return self.trained_models.get(name)
    
    def save_model(self, name: str, path: str = None) -> str:
        """
        Save a trained model to disk.
        
        Args:
            name: Model name
            path: Save path (default: models/{name}.pkl)
            
        Returns:
            Path where model was saved
        """
        if name not in self.trained_models:
            raise ValueError(f"Model '{name}' not found in trained models.")
        
        if path is None:
            path = f"models/{name}.pkl"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.trained_models[name], path)
        logger.info(f"Model '{name}' saved to {path}")
        
        return path
    
    def save_all_models(self, directory: str = "models") -> list:
        """Save all trained models to a directory."""
        saved_paths = []
        for name in self.trained_models:
            path = self.save_model(name, f"{directory}/{name}.pkl")
            saved_paths.append(path)
        return saved_paths
    
    def load_model(self, name: str, path: str = None) -> Any:
        """Load a trained model from disk."""
        if path is None:
            path = f"models/{name}.pkl"
        
        model = joblib.load(path)
        self.trained_models[name] = model
        logger.info(f"Model '{name}' loaded from {path}")
        return model
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'average_precision',
    ) -> Dict[str, Dict]:
        """
        Cross-validate all models.
        
        Args:
            X: Training features
            y: Training labels
            cv: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV results for each model
        """
        results = {}
        
        for name in self.models_to_train:
            config = self.configs[name]
            params = config['params'].copy()
            
            if name == 'xgboost':
                params['scale_pos_weight'] = self._compute_scale_pos_weight(y)
            
            model = config['model_class'](**params)
            
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
            
            results[name] = {
                'mean_score': round(float(scores.mean()), 4),
                'std_score': round(float(scores.std()), 4),
                'scores': [round(float(s), 4) for s in scores],
            }
            
            logger.info(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")
        
        return results


class IsolationForestDetector:
    """
    Unsupervised anomaly detection using Isolation Forest.
    
    Trained ONLY on legitimate transactions to learn "normal" patterns.
    Anomalies (high negative scores) are flagged as potential fraud.
    """
    
    def __init__(
        self,
        contamination: float = 0.01,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = None) -> 'IsolationForestDetector':
        """
        Fit on legitimate transactions only.
        
        Args:
            X_train: Training features
            y_train: Training labels (used to filter legitimate only)
        """
        if y_train is not None:
            X_legit = X_train[y_train == 0]
        else:
            X_legit = X_train
        
        logger.info(
            f"Training Isolation Forest on {len(X_legit)} legitimate transactions "
            f"(excluded {(y_train == 1).sum()} fraud samples)"
        )
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_legit)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies.
        
        Returns:
            Array of -1 (anomaly) and 1 (normal)
        """
        return self.model.predict(X)
    
    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous).
        
        Returns:
            Array of anomaly scores
        """
        return self.model.score_samples(X)
    
    def predict_proba_as_fraud(self, X: pd.DataFrame) -> np.ndarray:
        """
        Convert anomaly scores to fraud-like probabilities.
        
        Returns:
            Array of scores between 0 and 1 (higher = more likely fraud)
        """
        scores = self.score(X)
        # Normalize to [0, 1] where 1 = most anomalous
        min_score = scores.min()
        max_score = scores.max()
        probas = 1 - (scores - min_score) / (max_score - min_score + 1e-10)
        return probas


def get_model_descriptions() -> Dict[str, str]:
    """Get human-readable descriptions of each model."""
    return {
        'logistic_regression': (
            'Linear baseline model. Fast, interpretable, uses class_weight for imbalance. '
            'Good for establishing a performance floor.'
        ),
        'random_forest': (
            'Ensemble of decision trees. Robust, handles non-linearity well, '
            'provides feature importance. Uses balanced class weights.'
        ),
        'xgboost': (
            'Gradient boosting with sequential tree correction. Usually top performer. '
            'Uses scale_pos_weight to handle imbalance.'
        ),
        'lightgbm': (
            'Lightweight gradient boosting. Fast training, good performance. '
            'Uses is_unbalance parameter for imbalance handling.'
        ),
        'isolation_forest': (
            'Unsupervised anomaly detection. Trained only on legitimate transactions. '
            'Useful when labels are unreliable or for catching novel fraud patterns.'
        ),
    }


if __name__ == "__main__":
    # Quick demo
    from data_loader import load_data
    from preprocessing import FraudPreprocessor
    
    df, stats = load_data()
    preprocessor = FraudPreprocessor()
    data = preprocessor.full_preprocess(df)
    
    trainer = FraudTrainer()
    models = trainer.train_all(data['X_train'], data['y_train'])
    
    print(f"\nTrained models: {list(models.keys())}")
    
    # Cross-validate
    cv_results = trainer.cross_validate(data['X_train'], data['y_train'])
    for name, result in cv_results.items():
        print(f"  {name}: PR-AUC = {result['mean_score']:.4f} ± {result['std_score']:.4f}")
