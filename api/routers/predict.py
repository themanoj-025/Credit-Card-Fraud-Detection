"""
Prediction Router — /predict and /predict/batch endpoints.

Handles single and batch fraud predictions with anomaly scores.
"""

import logging

import pandas as pd
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
    TransactionInput,
)
from api.state import get_anomaly_detector, get_predictor
from src.fraudshield.config import AVG_FRAUD_LOSS, REVIEW_COST

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(transaction: TransactionInput) -> PredictionResponse:
    """
    Predict fraud probability for a single transaction.

    Returns fraud probability, anomaly score, SHAP explanation,
    and business impact analysis.
    """
    pred = get_predictor()
    if pred is None or pred.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = pred.predict_single(transaction.dict(), return_shap=True)

        # Get anomaly score from Isolation Forest (handles both raw sklearn model and wrapper)
        anomaly_score = None
        try:
            anomaly_det = get_anomaly_detector()
            if anomaly_det is not None:
                # anomaly_det may be raw sklearn IsolationForest or IsolationForestDetector wrapper
                raw_model = anomaly_det.model if hasattr(anomaly_det, 'model') else anomaly_det
                if raw_model is not None:
                    tx_dict = transaction.dict()
                    # Ensure all expected feature columns exist
                    X = pd.DataFrame([tx_dict]).reindex(columns=pred.feature_names, fill_value=0.0)
                    scores = raw_model.score_samples(X.values)
                    min_s, max_s = scores.min(), scores.max()
                    probas = 1 - (scores - min_s) / (max_s - min_s + 1e-10)
                    anomaly_score = round(float(probas[0]), 4)
        except Exception as anomaly_err:
            logger.warning("Anomaly score computation failed: %s", anomaly_err)

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
    pred = get_predictor()
    if pred is None or pred.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        transactions = [t.dict() for t in batch.transactions]
        X = pd.DataFrame(transactions)

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
