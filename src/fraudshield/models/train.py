"""
Training Module

Trains multiple fraud detection models for comparison and selection.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.fraudshield.config import (
    CATBOOST_DEPTH,
    CATBOOST_ITERATIONS,
    CATBOOST_VERBOSE,
    CROSS_VALIDATION_FOLDS,
    CROSS_VALIDATION_SCORING,
    DEFAULT_MODELS,
    MODELS_DIR,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


class FraudTrainer:
    """
    Train and compare multiple fraud detection models.

    Models:
    - Logistic Regression (baseline, interpretable)
    - Random Forest (tree ensemble, robust)
    - Gradient Boosting (sklearn native, sequential trees)
    - XGBoost (optimized gradient boosting, top performer)
    - LightGBM (lightweight gradient boosting, fast)
    - CatBoost (categorical boosting, handles imbalance)
    """

    DEFAULT_CONFIGS: Dict[str, Dict] = {
        "logistic_regression": {
            "model_class": LogisticRegression,
            "params": {
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "n_jobs": -1,
            },
            "uses_class_weight": True,
        },
        "random_forest": {
            "model_class": RandomForestClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": RANDOM_STATE,
                "class_weight": "balanced",
                "n_jobs": -1,
            },
            "uses_class_weight": True,
        },
        "gradient_boosting": {
            "model_class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.1,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": RANDOM_STATE,
            },
            "uses_class_weight": False,
        },
        "xgboost": {
            "model_class": XGBClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "scale_pos_weight": 1,
                "random_state": RANDOM_STATE,
                "eval_metric": "aucpr",
                "use_label_encoder": False,
                "n_jobs": -1,
            },
            "uses_class_weight": False,
        },
        "lightgbm": {
            "model_class": lgb.LGBMClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "is_unbalance": True,
                "random_state": RANDOM_STATE,
                "verbose": -1,
                "n_jobs": -1,
            },
            "uses_class_weight": False,
        },
        "catboost": {
            "model_class": None,  # Set dynamically in __init__
            "params": {
                "iterations": CATBOOST_ITERATIONS,
                "depth": CATBOOST_DEPTH,
                "learning_rate": 0.1,
                "random_seed": RANDOM_STATE,
                "verbose": CATBOOST_VERBOSE,
                "auto_class_weights": "Balanced",
            },
            "uses_class_weight": False,
        },
    }

    def __init__(
        self,
        models_to_train: Optional[List[str]] = None,
        custom_configs: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            models_to_train: List of model names to train (default from config)
            custom_configs: Custom model configurations to override defaults
        """
        self.configs = {**self.DEFAULT_CONFIGS}
        if custom_configs:
            self.configs.update(custom_configs)

        # Try to load CatBoost — graceful fallback if not installed
        try:
            from catboost import CatBoostClassifier
            self.configs["catboost"]["model_class"] = CatBoostClassifier
        except ImportError:
            logger.info("CatBoost not installed. Removing from configs.")
            if "catboost" in self.configs:
                del self.configs["catboost"]

        self.models_to_train = [m for m in (models_to_train or DEFAULT_MODELS) if m in self.configs]
        self.trained_models: Dict[str, Any] = {}
        self.training_results: Dict[str, Dict] = {}

    def _compute_scale_pos_weight(self, y: pd.Series) -> float:
        """Compute scale_pos_weight for XGBoost based on class imbalance."""
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        return n_neg / n_pos if n_pos > 0 else 1.0

    def train_model(self, name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
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
        params = config["params"].copy()

        if name == "xgboost":
            params["scale_pos_weight"] = self._compute_scale_pos_weight(y_train)

        logger.info("Training %s...", name)
        start_time = time.time()

        model = config["model_class"](**params)
        model.fit(X_train, y_train)

        train_time = time.time() - start_time
        logger.info("  %s trained in %.2fs", name, train_time)

        self.trained_models[name] = model
        self.training_results[name] = {
            "train_time": train_time,
            "n_samples": len(X_train),
            "n_features": X_train.shape[1],
        }

        return model

    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all configured models.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary of trained models
        """
        logger.info("Training %d models...", len(self.models_to_train))
        for name in self.models_to_train:
            self.train_model(name, X_train, y_train)
        logger.info("All models trained: %s", list(self.trained_models.keys()))
        return self.trained_models

    def save_model(self, name: str, path: Optional[str] = None) -> str:
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
            path = str(MODELS_DIR / f"{name}.pkl")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.trained_models[name], path)
        logger.info("Model '%s' saved to %s", name, path)
        return path

    def save_all_models(self, directory: str = "models") -> List[str]:
        """Save all trained models to a directory."""
        return [
            self.save_model(name, f"{directory}/{name}.pkl")
            for name in self.trained_models
        ]

    def load_model(self, name: str, path: Optional[str] = None) -> Any:
        """Load a trained model from disk."""
        if path is None:
            path = str(MODELS_DIR / f"{name}.pkl")
        model = joblib.load(path)
        self.trained_models[name] = model
        logger.info("Model '%s' loaded from %s", name, path)
        return model

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = CROSS_VALIDATION_FOLDS,
        scoring: str = CROSS_VALIDATION_SCORING,
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
            params = config["params"].copy()

            if name == "xgboost":
                params["scale_pos_weight"] = self._compute_scale_pos_weight(y)

            model = config["model_class"](**params)
            skf = StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=RANDOM_STATE
            )
            scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)

            results[name] = {
                "mean_score": round(float(scores.mean()), 4),
                "std_score": round(float(scores.std()), 4),
                "scores": [round(float(s), 4) for s in scores],
            }
            logger.info("  %s: %.4f ± %.4f", name, scores.mean(), scores.std())

        return results
