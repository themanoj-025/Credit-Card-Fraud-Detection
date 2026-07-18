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
from src.fraudshield.config import AVG_FRAUD_LOSS, MODELS_DIR, REVIEW_COST
from src.fraudshield.explainability.shap_utils import FraudPredictor
from src.fraudshield.llm.case_narrator import CaseNarrator
from src.fraudshield.llm.rag_similar_cases import SimilarCaseRetriever
from src.fraudshield.models.anomaly import IsolationForestDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Global State ────────────────────────────────────────────────────────
predictor: Optional[FraudPredictor] = None
anomaly_detector: Optional[IsolationForestDetector] = None
case_narrator: Optional[CaseNarrator] = None
case_retriever: Optional[SimilarCaseRetriever] = None
copilot_client = None

# ─── Attempt to load optional dependencies ───────────────────────────────

def _try_load_copilot():
    """Try to load Anthropic client for copilot features."""
    global copilot_client
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            from anthropic import Anthropic
            copilot_client = Anthropic(api_key=api_key)
            logger.info("Analyst Copilot initialized")
        except ImportError:
            logger.warning("anthropic package not installed. Copilot unavailable.")
    else:
        logger.info("ANTHROPIC_API_KEY not set. Copilot unavailable.")


def _try_load_case_narrator():
    """Try to load the case narrator."""
    global case_narrator
    try:
        case_narrator = CaseNarrator()
        logger.info("Case Narrator initialized")
    except Exception as e:
        logger.warning("Case Narrator unavailable: %s", e)


def _try_load_rag_retriever():
    """Try to load the RAG-based similar case retriever."""
    global case_retriever
    index_path = MODELS_DIR / "rag_index"
    if index_path.exists():
        try:
            case_retriever = SimilarCaseRetriever()
            case_retriever.load(str(index_path))
            logger.info("RAG retriever loaded from %s", index_path)
        except Exception as e:
            logger.warning("RAG retriever unavailable: %s", e)
    else:
        logger.info("No RAG index found at %s. Run build_rag_index.py first.", index_path)


# ─── Lifecycle ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and dependencies on startup."""
    global predictor, anomaly_detector

    logger.info("Starting FraudLens API v2.0.0...")

    # Load supervised model
    try:
        predictor = FraudPredictor()
        predictor.load_from_config()
        # Load threshold
        threshold_path = MODELS_DIR / "threshold.txt"
        if threshold_path.exists():
            with open(threshold_path) as f:
                predictor.threshold = float(f.read().strip())
            logger.info("Threshold loaded: %.4f", predictor.threshold)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning("Failed to load model: %s", e)

    # Load anomaly detector
    try:
        anomaly_path = MODELS_DIR / "anomaly_detector.pkl"
        if anomaly_path.exists():
            import joblib
            anomaly_detector = joblib.load(anomaly_path)
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
    return {
        "status": "healthy",
        "model_loaded": predictor is not None and predictor.model is not None,
        "version": "2.0.0",
    }


@app.get("/model-info")
async def model_info():
    """Get model metadata and configuration."""
    if predictor is None:
        return {"status": "model not loaded"}

    return {
        "model_type": type(predictor.model).__name__ if predictor.model else None,
        "threshold": predictor.threshold,
        "n_features": len(predictor.feature_names or []),
        "features": predictor.feature_names,
        "avg_fraud_loss": AVG_FRAUD_LOSS,
        "review_cost": REVIEW_COST,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
