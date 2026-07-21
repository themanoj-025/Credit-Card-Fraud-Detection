"""
Centralized configuration for FraudLens.

All configurable constants live here — business costs, paths, model params,
and feature definitions. Import from here rather than hardcoding elsewhere.
"""

from pathlib import Path
from typing import Dict, List

# ─── Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Dataset path
DEFAULT_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"

# ─── Business Cost Assumptions ──────────────────────────────────────────
AVG_FRAUD_LOSS: float = 150.0       # Average dollar loss per missed fraud
REVIEW_COST: float = 5.0            # Cost to manually review a flagged transaction

# ─── Data Split ──────────────────────────────────────────────────────────
TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.1
RANDOM_STATE: int = 42

# ─── Features ────────────────────────────────────────────────────────────
PCA_FEATURES: List[str] = [f"V{i}" for i in range(1, 29)]
SCALE_FEATURES: List[str] = ["Time", "Amount"]
ALL_FEATURES: List[str] = PCA_FEATURES + SCALE_FEATURES
TARGET_COLUMN: str = "Class"

# ─── Resampling ──────────────────────────────────────────────────────────
RESAMPLING_STRATEGIES: List[str] = [
    "none",
    "class_weight",
    "random_under",
    "smote",
    "adasyn",
    "smote_tomek",
]
DEFAULT_RESAMPLING: str = "smote"

# ─── Model Training ──────────────────────────────────────────────────────
DEFAULT_MODELS: List[str] = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "catboost",
]

# Auto-selection rule: pick model with highest PR-AUC
MODEL_SELECTION_METRIC: str = "pr_auc"
SELECTION_RULE_DESCRIPTION: str = (
    "Model with highest PR-AUC is selected as best. "
    "PR-AUC is preferred over ROC-AUC for imbalanced fraud data."
)

# Isolation Forest
IFOREST_CONTAMINATION: float = 0.01
IFOREST_N_ESTIMATORS: int = 200

# Autoencoder
AUTOENCODER_ENCODING_DIM: int = 16
AUTOENCODER_EPOCHS: int = 20
AUTOENCODER_BATCH_SIZE: int = 32

# CatBoost
CATBOOST_VERBOSE: bool = False
CATBOOST_ITERATIONS: int = 200
CATBOOST_DEPTH: int = 6

# ─── Threshold Tuning ────────────────────────────────────────────────────
N_THRESHOLDS: int = 100

# ─── SHAP ────────────────────────────────────────────────────────────────
MAX_SHAP_FEATURES: int = 10
N_SHAP_BACKGROUND_SAMPLES: int = 100

# ─── LLM / Copilot ───────────────────────────────────────────────────────
LLM_MODEL: str = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS: int = 1000
LLM_TEMPERATURE: float = 0.3

# RAG
RAG_TOP_K: int = 3
EMBEDDING_DIM: int = 30  # Same as feature count (V1-V28 + Time + Amount)

# ─── Drift Detection ─────────────────────────────────────────────────────
DRIFT_THRESHOLD: float = 0.05  # p-value threshold for KS test
DRIFT_ALERT_WINDOW: int = 1000  # Check drift every N transactions

# ─── Dashboard ───────────────────────────────────────────────────────────
DASHBOARD_REFRESH_MS: int = 500
MAX_TRANSACTION_HISTORY: int = 500
SIMULATION_FRAUD_RATE: float = 0.02
SIMULATION_BATCH_SIZE: int = 10

# ─── API ────────────────────────────────────────────────────────────────────
API_URL: str = "http://localhost:8000"              # Base URL for FastAPI
API_PORT: int = 8000
DASHBOARD_PORT: int = 8501

# ─── Evaluation ──────────────────────────────────────────────────────────
CROSS_VALIDATION_FOLDS: int = 5
CROSS_VALIDATION_SCORING: str = "average_precision"

# ─── MLflow Experiment Tracking ──────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME: str = "fraudlens_model_comparison"
MLFLOW_TRACKING_URI: str = "http://localhost:5000"  # MLflow server URL
MLFLOW_ARTIFACT_DIR: str = "mlruns"                 # Local artifact storage

# ─── Hyperparameter Optimization (Optuna) ─────────────────────────────────
HPO_ENABLED: bool = True                # Enable HPO when running full pipeline
HPO_N_TRIALS: int = 30                  # Number of Optuna trials per model
HPO_CV_FOLDS: int = 3                  # CV folds for HPO evaluation
HPO_MODELS: List[str] = ["xgboost", "lightgbm"]  # Models to tune
