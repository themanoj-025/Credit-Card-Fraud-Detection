"""
Anomaly Detection Module

Unsupervised anomaly detection models for fraud detection.
Provides an alternative signal to supervised models:
- "Known fraud pattern match" (supervised model)
- "Statistically unusual, possibly a new pattern" (anomaly detector)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.fraudlens.config import (
    IFOREST_CONTAMINATION,
    IFOREST_N_ESTIMATORS,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Unsupervised anomaly detection using Isolation Forest.

    Trained ONLY on legitimate transactions to learn "normal" patterns.
    Anomalies (high negative scores) are flagged as potential fraud,
    making this useful for catching *novel* fraud patterns the supervised
    model hasn't seen before.
    """

    def __init__(
        self,
        contamination: float = IFOREST_CONTAMINATION,
        n_estimators: int = IFOREST_N_ESTIMATORS,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """
        Args:
            contamination: Expected proportion of anomalies in the data
            n_estimators: Number of isolation trees
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None

    def fit(
        self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None
    ) -> "IsolationForestDetector":
        """
        Fit on legitimate transactions only.

        Args:
            X_train: Training features
            y_train: Training labels (used to filter legitimate only)
        """
        if y_train is not None:
            X_legit = X_train[y_train == 0]
            logger.info(
                "Training Isolation Forest on %d legitimate transactions (excluded %d fraud samples)",
                len(X_legit),
                int((y_train == 1).sum()),
            )
        else:
            X_legit = X_train
            logger.info("Training Isolation Forest on %d transactions", len(X_legit))

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
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous).

        Returns:
            Array of anomaly scores
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.score_samples(X)

    def predict_proba_as_fraud(self, X: pd.DataFrame) -> np.ndarray:
        """
        Convert anomaly scores to fraud-like probabilities (0-1).

        Higher scores = more likely to be fraud/anomalous.

        Returns:
            Array of scores between 0 and 1
        """
        scores = self.score(X)
        min_score = scores.min()
        max_score = scores.max()
        probas = 1 - (scores - min_score) / (max_score - min_score + 1e-10)
        return probas


class AutoencoderDetector:
    """
    Unsupervised anomaly detection using a simple Autoencoder.

    Reconstruction error is used as anomaly score.
    Higher reconstruction error = more anomalous.
    """

    def __init__(
        self,
        encoding_dim: int = 16,
        epochs: int = 20,
        batch_size: int = 32,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """
        Args:
            encoding_dim: Dimension of the bottleneck layer
            epochs: Number of training epochs
            batch_size: Batch size for training
            random_state: Random seed
        """
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self._fitted = False

    def _build_model(self, input_dim: int):
        """Build the autoencoder architecture."""
        try:
            from tensorflow.keras import layers, models
        except ImportError:
            logger.warning("TensorFlow/Keras not installed. Autoencoder unavailable.")
            raise ImportError(
                "TensorFlow is required for AutoencoderDetector. "
                "Install with: pip install tensorflow"
            )

        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(self.encoding_dim, activation="relu")(input_layer)
        decoded = layers.Dense(input_dim, activation="linear")(encoded)

        autoencoder = models.Model(input_layer, decoded)
        autoencoder.compile(optimizer="adam", loss="mse")
        self.model = autoencoder

    def fit(
        self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None
    ) -> "AutoencoderDetector":
        """
        Fit autoencoder on training data.

        Args:
            X_train: Training features
            y_train: Optional labels (not used in unsupervised training)
        """
        self._build_model(X_train.shape[1])
        self.model.fit(
            X_train.values,
            X_train.values,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            shuffle=True,
        )
        self._fitted = True
        logger.info(
            "Autoencoder trained on %d samples (encoding_dim=%d)",
            len(X_train),
            self.encoding_dim,
        )
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Get reconstruction error as anomaly score."""
        if not self._fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        reconstructions = self.model.predict(X.values, verbose=0)
        mse = np.mean((X.values - reconstructions) ** 2, axis=1)
        return mse

    def predict_proba_as_fraud(self, X: pd.DataFrame) -> np.ndarray:
        """Convert reconstruction error to fraud probabilities (0-1)."""
        errors = self.score(X)
        # Normalize to [0, 1]
        min_err = errors.min()
        max_err = errors.max()
        probas = (errors - min_err) / (max_err - min_err + 1e-10)
        return probas
