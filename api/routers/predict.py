"""
Prediction Router — /predict and /predict/batch endpoints.

Handles single and batch fraud predictions with anomaly scores.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from api.schemas import (
    BatchInput,
    BatchPredictionItem,
    BatchResponse,
    BatchSummary,
    BusinessImpact,
    Explanation,
    PredictionResponse,
    ShapFeature,
)
from src.fraudshield.config import AVG_FRAUD_LOSS, REVIEW_COST

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"])


def _get_predictor():
    """Get the global FraudPredictor instance."""
    from api.main import predictor
    return predictor


def _get_anomaly_detector():
    """Get the global anomaly detector instance."""
    from api.main import anomaly_detector
    return anomaly_detector


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(transaction: dict) -> PredictionResponse:
    """
    Predict fraud probability for a single transaction.

    Returns fraud probability, anomaly score, SHAP explanation,
    and business impact analysis.
    """
    pred = _get_predictor()
    if pred is None or pred.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = pred.predict_single(transaction, return_shap=True)

        # Get anomaly score from Isolation Forest
        anomaly_score = None
        anomaly_det = _get_anomaly_detector()
        if anomaly_det is not None and anomaly_det.model is not None:
            import pandas as pd
            X = pd.DataFrame([transaction])[pred.feature_names]
            anomaly_probas = anomaly_det.predict_proba_as_fraud(X)
            anomaly_score = round(float(anomaly_probas[0]), 4)

        # Build response
        explanation = None
        if "explanation" in result:
            explanation = Explanation(
                summary=result["explanation"]["summary"],
                top_features=[
                    ShapFeature(**f) for f in result["explanation"]["top_features"]
                ],
            )

        business = BusinessImpact(
            estimated_loss=AVG_FRAUD_LOSS,
            action="FLAG for manual review" if result["is_fraud"] else "AUTO-APPROVE",
            review_cost=REVIEW_COST,
        )

        return PredictionResponse(
            fraud_probability=result["fraud_probability"],
            decision=result["decision"],
            threshold_used=result["threshold_used"],
            is_fraud=result["is_fraud"],
            anomaly_score=anomaly_score,
            explanation=explanation,
            business_impact=business,
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(batch: BatchInput) -> BatchResponse:
    """
    Predict fraud for multiple transactions (faster, no SHAP).

    Returns a summary with counts and estimated review costs.
    """
    pred = _get_predictor()
    if pred is None or pred.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import pandas as pd

        # Convert to DataFrame
        transactions = [t.dict() for t in batch.transactions]
        X = pd.DataFrame(transactions)

        # Batch predict
        results = pred.predict_batch(X)

        predictions = []
        flagged_fraud = 0
        for _, row in results.iterrows():
            predictions.append(
                BatchPredictionItem(
                    fraud_probability=round(float(row["fraud_probability"]), 4),
                    decision=row["decision"],
                    is_fraud=bool(row["prediction"]),
                )
            )
            if row["prediction"]:
                flagged_fraud += 1

        summary = BatchSummary(
            total=len(predictions),
            flagged_fraud=flagged_fraud,
            flagged_legitimate=len(predictions) - flagged_fraud,
            estimated_review_cost=round(flagged_fraud * REVIEW_COST, 2),
        )

        return BatchResponse(predictions=predictions, summary=summary)
    except Exception as e:
        logger.error("Batch prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
