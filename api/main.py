"""
FastAPI Application for Fraud Detection

Endpoints:
- POST /predict: Predict fraud probability with SHAP explanation
- GET /health: Health check endpoint
- GET /model-info: Get model metadata
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from src.predict import FraudPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App Setup ──────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Production-grade fraud detection with SHAP explainability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Model on Startup ──────────────────────────────────
predictor: Optional[FraudPredictor] = None
MODEL_LOADED = False

# PCA features that the model expects
PCA_FEATURES = [f'V{i}' for i in range(1, 29)]
REQUIRED_FEATURES = PCA_FEATURES + ['Time', 'Amount']


class TransactionInput(BaseModel):
    """Pydantic model for input validation."""
    Time: float = Field(..., description="Transaction time in seconds since first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount in dollars")
    
    # PCA features V1-V28
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    
    class Config:
        json_schema_extra = {
            "example": {
                "Time": 100000.0,
                "Amount": 150.0,
                "V1": -1.3598071336738,
                "V2": -0.0727811733098497,
                "V3": 2.53634673796914,
                "V4": 1.37815522427443,
                "V5": -0.338320769942518,
                "V6": 0.462387777762292,
                "V7": 0.239598554061358,
                "V8": 0.0986979011518104,
                "V9": 0.363786969611213,
                "V10": 0.0907941719789316,
                "V11": -0.551599533260813,
                "V12": -0.617800855762336,
                "V13": -0.991389847235408,
                "V14": -0.311169353699879,
                "V15": 1.46817697209427,
                "V16": -0.470400525087464,
                "V17": 0.207971241929242,
                "V18": 0.0257905801985591,
                "V19": 0.403992960255733,
                "V20": 0.251412098239705,
                "V21": -0.018306777944153,
                "V22": 0.277837575558899,
                "V23": -0.110473910188767,
                "V24": 0.0669280749146731,
                "V25": 0.128539358273528,
                "V26": -0.189114877844649,
                "V27": 0.133558376740387,
                "V28": 0.0211094540172886,
            }
        }


class PredictionResponse(BaseModel):
    """Response model for fraud prediction."""
    fraud_probability: float
    decision: str
    threshold_used: float
    is_fraud: bool
    explanation: Optional[Dict[str, Any]] = None
    business_impact: Optional[Dict[str, Any]] = None


class BatchInput(BaseModel):
    """Batch prediction input."""
    transactions: List[TransactionInput]


# ─── Model Loading ──────────────────────────────────────────
def load_models():
    """Load the trained model, scaler, and threshold."""
    global predictor, MODEL_LOADED
    
    model_path = Path("models/xgboost.pkl")
    scaler_path = Path("models/scaler.pkl")
    threshold_path = Path("models/threshold.txt")
    
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}. API will return mock predictions.")
        return
    
    predictor = FraudPredictor(
        feature_names=REQUIRED_FEATURES,
        threshold=0.5,
    )
    predictor.load_model(str(model_path))
    
    if scaler_path.exists():
        predictor.load_scaler(str(scaler_path))
    
    if threshold_path.exists():
        predictor.threshold = float(threshold_path.read_text().strip())
        logger.info(f"Loaded optimal threshold: {predictor.threshold}")
    
    MODEL_LOADED = True
    logger.info("All models loaded successfully.")


@app.on_event("startup")
async def startup():
    load_models()


# ─── Endpoints ──────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "version": "1.0.0",
    }


@app.get("/model-info")
async def model_info():
    """Return model metadata."""
    if not MODEL_LOADED:
        return {"status": "no model loaded"}
    
    return {
        "model_type": type(predictor.model).__name__,
        "threshold": predictor.threshold,
        "n_features": len(REQUIRED_FEATURES),
        "features": REQUIRED_FEATURES,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionInput):
    """
    Predict fraud probability for a single transaction.
    
    Returns:
    - fraud_probability: 0 to 1 probability of fraud
    - decision: 'FRAUD' or 'LEGITIMATE'
    - explanation: SHAP-based feature importance breakdown
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first and place it in models/",
        )
    
    # Convert to dict
    transaction_dict = transaction.model_dump()
    
    try:
        result = predictor.predict_single(
            transaction_dict,
            return_shap=True,
        )
        
        # Add business impact
        if result['is_fraud']:
            result['business_impact'] = {
                'estimated_loss': 150.0,
                'action': 'FLAG for manual review',
                'review_cost': 5.0,
            }
        else:
            result['business_impact'] = {
                'estimated_loss': 0.0,
                'action': 'APPROVE',
                'review_cost': 0.0,
            }
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(batch: BatchInput):
    """
    Predict fraud for multiple transactions.
    
    Returns a list of predictions.
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )
    
    results = []
    for transaction in batch.transactions:
        try:
            result = predictor.predict_single(
                transaction.model_dump(),
                return_shap=False,  # Skip SHAP for batch speed
            )
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "fraud_probability": 0.0,
                "decision": "ERROR",
            })
    
    # Summary
    n_fraud = sum(1 for r in results if r.get('is_fraud', False))
    total_review_cost = n_fraud * 5.0
    
    return {
        "predictions": results,
        "summary": {
            "total": len(results),
            "flagged_fraud": n_fraud,
            "flagged_legitimate": len(results) - n_fraud,
            "estimated_review_cost": total_review_cost,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
