"""
Prediction Module

Makes predictions with SHAP-based explanations.
Provides per-transaction interpretability for fraud analysts.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import joblib
import logging
import shap

logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    Prediction pipeline with SHAP explanations.
    
    Returns not just "fraud: 92%" but "flagged mainly due to V14, V4, V12"
    — exactly what real fraud analysts need.
    """
    
    def __init__(
        self,
        model=None,
        scaler=None,
        feature_names: List[str] = None,
        threshold: float = 0.5,
        max_shap_features: int = 10,
    ):
        """
        Args:
            model: Trained model with predict_proba
            scaler: Fitted StandardScaler
            feature_names: List of feature names
            threshold: Classification threshold
            max_shap_features: Max features to include in SHAP explanation
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.threshold = threshold
        self.max_shap_features = max_shap_features
        self.explainer = None
        self._shap_initialized = False
    
    def load_model(self, model_path: str) -> 'FraudPredictor':
        """Load a model from disk."""
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return self
    
    def load_scaler(self, scaler_path: str) -> 'FraudPredictor':
        """Load a scaler from disk."""
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")
        return self
    
    def _init_shap_explainer(self, X_background: pd.DataFrame = None) -> None:
        """
        Initialize the SHAP explainer.
        
        Uses TreeExplainer for tree-based models, KernelExplainer as fallback.
        """
        if self._shap_initialized:
            return
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        model_type = type(self.model).__name__
        
        # Choose explainer based on model type
        if hasattr(self.model, 'feature_importances_') or 'XGB' in model_type or 'LGBM' in model_type or 'Forest' in model_type:
            # Tree-based model — use fast TreeExplainer
            logger.info(f"Using TreeExplainer for {model_type}")
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Fallback to KernelExplainer (slower but works with any model)
            logger.info(f"Using KernelExplainer for {model_type}")
            bg = shap.sample(X_background if X_background is not None else pd.DataFrame(
                np.zeros((100, len(self.feature_names))),
                columns=self.feature_names
            ), 100)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, bg)
        
        self._shap_initialized = True
    
    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input features (scale if scaler available)."""
        if self.scaler is not None:
            X_scaled = X.copy()
            scale_cols = ['Time', 'Amount']
            available = [c for c in scale_cols if c in X_scaled.columns]
            if available:
                X_scaled[available] = self.scaler.transform(X_scaled[available])
            return X_scaled
        return X
    
    def predict_single(
        self,
        transaction: Dict[str, Any],
        return_shap: bool = True,
        X_background: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        """
        Predict fraud for a single transaction with explanation.
        
        Args:
            transaction: Dictionary of feature values
            return_shap: Whether to compute SHAP explanations
            X_background: Background data for SHAP (if needed)
            
        Returns:
            Dictionary with prediction, probability, and explanation
        """
        # Convert to DataFrame
        if self.feature_names:
            X = pd.DataFrame([transaction])[self.feature_names]
        else:
            X = pd.DataFrame([transaction])
        
        # Preprocess
        X_processed = self.preprocess(X)
        
        # Predict probability
        fraud_proba = self.model.predict_proba(X_processed)[0][1]
        
        # Classify
        is_fraud = fraud_proba >= self.threshold
        decision = "FRAUD" if is_fraud else "LEGITIMATE"
        
        result = {
            'fraud_probability': round(float(fraud_proba), 4),
            'decision': decision,
            'threshold_used': self.threshold,
            'is_fraud': bool(is_fraud),
        }
        
        # SHAP explanation
        if return_shap:
            self._init_shap_explainer(X_background)
            
            shap_values = self.explainer.shap_values(X_processed)
            
            # For binary classifiers, shap_values might be a list [class_0, class_1]
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]  # Fraud class, first sample
            else:
                shap_vals = shap_values[0]
            
            # Get top contributing features
            feature_importance = list(zip(self.feature_names, shap_vals))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_features = feature_importance[:self.max_shap_features]
            
            # Build explanation
            explanation = []
            for feat, val in top_features:
                direction = "increases" if val > 0 else "decreases"
                explanation.append({
                    'feature': feat,
                    'value': round(float(transaction.get(feat, X.iloc[0][feat])), 4),
                    'shap_value': round(float(val), 4),
                    'impact': direction,
                })
            
            result['explanation'] = {
                'summary': self._format_explanation(top_features),
                'top_features': explanation,
                'all_shap_values': {feat: round(float(val), 6) for feat, val in feature_importance},
            }
        
        return result
    
    def predict_batch(
        self,
        X: pd.DataFrame,
        threshold: float = None,
    ) -> pd.DataFrame:
        """
        Predict fraud for a batch of transactions.
        
        Returns:
            DataFrame with original data + fraud_probability + prediction
        """
        t = threshold or self.threshold
        X_processed = self.preprocess(X)
        
        probas = self.model.predict_proba(X_processed)[:, 1]
        
        result = X.copy()
        result['fraud_probability'] = probas
        result['prediction'] = (probas >= t).astype(int)
        result['decision'] = result['prediction'].map({0: 'LEGITIMATE', 1: 'FRAUD'})
        
        return result
    
    def _format_explanation(
        self,
        top_features: List[Tuple[str, float]],
    ) -> str:
        """Format top features into a human-readable explanation."""
        increases = [(f, v) for f, v in top_features if v > 0]
        decreases = [(f, v) for f, v in top_features if v < 0]
        
        parts = []
        
        if increases:
            feats = ", ".join([f"{f}" for f, _ in increases[:3]])
            parts.append(f"Flagged mainly due to: {feats}")
        
        if decreases:
            feats = ", ".join([f"{f}" for f, _ in decreases[:3]])
            parts.append(f"Mitigated by: {feats}")
        
        return " | ".join(parts) if parts else "No strong individual feature drivers"
    
    def get_global_feature_importance(
        self,
        X_sample: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute global feature importance using SHAP.
        
        Args:
            X_sample: Sample of data to compute SHAP values over
            
        Returns:
            DataFrame with feature importances sorted by magnitude
        """
        self._init_shap_explainer(X_sample)
        X_processed = self.preprocess(X_sample)
        
        shap_values = self.explainer.shap_values(X_processed)
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # Fraud class
        else:
            shap_vals = shap_values
        
        # Mean absolute SHAP value per feature
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': np.abs(shap_vals).mean(axis=0),
            'mean_shap': shap_vals.mean(axis=0),
        }).sort_values('mean_abs_shap', ascending=False)
        
        return importance


if __name__ == "__main__":
    # Quick demo
    print("FraudPredictor module loaded successfully.")
    print("Use it with a trained model and scaler for predictions.")
