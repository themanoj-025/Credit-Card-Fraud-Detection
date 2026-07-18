"""
FraudLens API — FastAPI Application

Production-grade REST API for credit card fraud detection with:
- Single & batch prediction
- SHAP explainability
- LLM case narration
- RAG similar-case retrieval
- Analyst copilot chat
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routers import chat, explain, predict, similar_cases
from api.state import (
    get_predictor,
    set_anomaly_detector,
    set_case_narrator,
    set_case_retriever,
    set_copilot_client,
    set_predictor,
)
from src.fraudshield.config import AVG_FRAUD_LOSS, MODELS_DIR, REVIEW_COST
from src.fraudshield.explainability.shap_utils import FraudPredictor
from src.fraudshield.llm.case_narrator import CaseNarrator
from src.fraudshield.llm.rag_similar_cases import SimilarCaseRetriever
from src.fraudshield.models.anomaly import IsolationForestDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ─── Attempt to load optional dependencies ───────────────────────────────

def _try_load_copilot():
    """Try to load Anthropic client for copilot features."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            from anthropic import Anthropic
            set_copilot_client(Anthropic(api_key=api_key))
            logger.info("Analyst Copilot initialized")
        except ImportError:
            logger.warning("anthropic package not installed. Copilot unavailable.")
    else:
        logger.info("ANTHROPIC_API_KEY not set. Copilot unavailable.")


def _try_load_case_narrator():
    """Try to load the case narrator."""
    try:
        set_case_narrator(CaseNarrator())
        logger.info("Case Narrator initialized")
    except Exception as e:
        logger.warning("Case Narrator unavailable: %s", e)


def _try_load_rag_retriever():
    """Try to load the RAG-based similar case retriever."""
    index_path = MODELS_DIR / "rag_index"
    if index_path.exists():
        try:
            retriever = SimilarCaseRetriever()
            retriever.load(str(index_path))
            set_case_retriever(retriever)
            logger.info("RAG retriever loaded from %s", index_path)
        except Exception as e:
            logger.warning("RAG retriever unavailable: %s", e)
    else:
        logger.info("No RAG index found at %s", index_path)


# ─── Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and dependencies on startup."""
    logger.info("Starting FraudLens API v2.0.0...")

    # Load supervised model
    try:
        predictor = FraudPredictor()
        predictor.load_from_config()
        threshold_path = MODELS_DIR / "threshold.txt"
        if threshold_path.exists():
            with open(threshold_path) as f:
                predictor.threshold = float(f.read().strip())
            logger.info("Threshold loaded: %.4f", predictor.threshold)
        set_predictor(predictor)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning("Failed to load model: %s", e)

    # Load anomaly detector
    try:
        anomaly_path = MODELS_DIR / "anomaly_detector.pkl"
        if anomaly_path.exists():
            import joblib
            detector = joblib.load(anomaly_path)
            set_anomaly_detector(detector)
            logger.info("Anomaly detector loaded")
    except Exception as e:
        logger.warning("Failed to load anomaly detector: %s", e)

    # Load optional features
    _try_load_case_narrator()
    _try_load_rag_retriever()
    _try_load_copilot()

    logger.info("FraudLens API ready")
    yield


# ─── App Creation ────────────────────────────────────────────────────────

app = FastAPI(
    title="FraudLens API",
    description="Production-grade credit card fraud detection with SHAP explainability, "
                "LLM case narration, and RAG-based similar case retrieval.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ──────────────────────────────────────────────────────────────

app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(similar_cases.router)
app.include_router(chat.router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    pred = get_predictor()
    return {
        "status": "healthy",
        "model_loaded": pred is not None and pred.model is not None,
        "version": "2.0.0",
    }


@app.get("/model-info")
async def model_info():
    """Get model metadata and configuration."""
    pred = get_predictor()
    if pred is None:
        return {"status": "model not loaded"}

    return {
        "model_type": type(pred.model).__name__ if pred.model else None,
        "threshold": pred.threshold,
        "n_features": len(pred.feature_names or []),
        "features": pred.feature_names,
        "avg_fraud_loss": AVG_FRAUD_LOSS,
        "review_cost": REVIEW_COST,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
