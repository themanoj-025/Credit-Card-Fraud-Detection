"""
Explanation Router — /explain endpoint.

Returns SHAP values + LLM narrative for a transaction.
"""

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import ExplanationResponse
from src.fraudshield.llm.case_narrator import CaseNarrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["explainability"])


@router.post("/explain", response_model=ExplanationResponse)
async def explain_transaction(transaction: dict) -> ExplanationResponse:
    """
    Get SHAP values and LLM narrative for a transaction.
    """
    from api.main import predictor, case_narrator

    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predictor.predict_single(transaction, return_shap=True)

        shap_values = {}
        if "explanation" in result:
            for f in result["explanation"]["top_features"]:
                shap_values[f["feature"]] = f["shap_value"]

        # Generate LLM narrative
        narrative = None
        if case_narrator is not None:
            shap_features = result.get("explanation", {}).get("top_features", [])
            narrative = case_narrator.narrate(
                transaction=transaction,
                fraud_probability=result["fraud_probability"],
                shap_explanation=shap_features,
                is_fraud=result["is_fraud"],
            )

        return ExplanationResponse(
            fraud_probability=result["fraud_probability"],
            decision=result["decision"],
            shap_values=shap_values,
            narrative=narrative,
        )
    except Exception as e:
        logger.error("Explanation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
