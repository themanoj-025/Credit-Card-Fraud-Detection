# 🏗️ Architecture — FraudLens

## System Architecture Overview

FraudLens uses a **three-layer hybrid architecture** combining traditional ML detection with LLM-powered explanation and interaction.

### Layer 1: Detection (Traditional ML)

The core detection engine uses **XGBoost** (PR-AUC: 0.8810) trained on PCA-transformed credit card transaction features, with an **Isolation Forest** for unsupervised anomaly detection alongside it.

### Layer 2: Explanation (SHAP + LLM)

Every prediction includes SHAP-based feature importance values. When an LLM (Anthropic Claude) is configured, the **CaseNarrator** translates SHAP values into plain-English narratives for fraud analysts.

### Layer 3: Interaction (RAG + Copilot)

A **FAISS-based SimilarCaseRetriever** finds the most similar historical transactions for every flagged case. The **Analyst Copilot** provides a natural-language chat interface with tool-use access to simulation data.

## Container Architecture (Docker Compose)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DOCKER NETWORK (fraudlens-net)                  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  FastAPI Server (:8000)                                          │   │
│  │  ┌────────────┐ ┌───────────┐ ┌──────────┐ ┌───────────────┐   │   │
│  │  │ Auth       │ │ Rate      │ │ RFC 7807 │ │ Security      │   │   │
│  │  │ (API Key)  │ │ Limiter   │ │ Errors   │ │ Headers       │   │   │
│  │  └────────────┘ └───────────┘ └──────────┘ └───────────────┘   │   │
│  │                                                                 │   │
│  │  Routers:                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │ /v1/predict  /v1/predict/batch  /v1/explain             │   │   │
│  │  │ /v1/similar-cases  /v1/chat  /v1/auth/keys  /v1/health  │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  │                                                                 │   │
│  │  DI Providers (app.state + FastAPI Depends()):                  │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │ get_predictor  get_anomaly_detector  get_case_narrator   │   │   │
│  │  │ get_case_retriever  get_copilot_client  get_db_session   │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│  ┌──────────────────────────┴──────────────┐                            │
│  │          FraudPredictor                  │                            │
│  │  ┌─────────────┐  ┌───────────────┐     │                            │
│  │  │ ModelLoader │  │ ShapExplainer │     │                            │
│  │  │ (checksum)  │  │ (async SHAP)  │     │                            │
│  │  └─────────────┘  └───────────────┘     │                            │
│  │  ┌────────────────────────┐             │                            │
│  │  │ PredictionCache (LRU)  │             │                            │
│  │  └────────────────────────┘             │                            │
│  └─────────────────────────────────────────┘                            │
│                                                                         │
│  ┌─────────────────────┐  ┌───────────────────┐  ┌─────────────────┐   │
│  │ PostgreSQL (5432)   │  │ Redis (6379)       │  │ Streamlit (:8501)│  │
│  │ fraudlens_db        │  │ Cache + RateLimit  │  │ 4-page dashboard│   │
│  │ Alembic migrations  │  │                   │  │ Shared API       │   │
│  └─────────────────────┘  └───────────────────┘  └─────────────────┘   │
│                                                                         │
│  ┌──────────────────────┐ (opt-in --profile training)                  │
│  │ MLflow Tracking (:5000)                                              │
│  │ Postgres backend store                                               │
│  └──────────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Kubernetes Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      KUBERNETES CLUSTER                              │
│                                                                      │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────┐               │
│  │ Ingress      │──▶│ Service      │──▶│ Deployment │               │
│  │ (nginx)     │   │ (ClusterIP)  │   │ fraudlens- │               │
│  │ TLS: *.fraud│   │ :8000 / :8501│   │ api (2-10) │               │
│  └─────────────┘   └──────────────┘   └──────┬─────┘               │
│                                              │                       │
│  ┌───────────────────┐   ┌──────────────────┐│                      │
│  │ ConfigMap         │   │ Secret           ││                      │
│  │ (non-sensitive)   │   │ (API keys, DB)   ││                      │
│  └───────────────────┘   └──────────────────┘│                      │
│                                              ▼                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ HorizontalPodAutoscaler (CPU 70% / Memory 80%)              │   │
│  │ minReplicas: 2  maxReplicas: 10                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Probes: Liveness (:/health :30s) + Readiness (:/health :15s)       │
│  Strategy: RollingUpdate (maxUnavailable: 1, maxSurge: 1)           │
└──────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### 1. Core Library (`src/fraudlens/`)

| Package | Module | Class/Function | Responsibility |
|---------|--------|----------------|----------------|
| **config** | `config.py` | — | Centralized constants (paths, costs, features) |
| **data** | `loaders.py` | `DataLoader` | Load CSV, compute statistics |
| **data** | `preprocessing.py` | `FraudPreprocessor` | Train/test split, scaling (no leakage) |
| **features** | `engineering.py` | `FeatureEngineer` | Log transforms, interactions, binning |
| **models** | `train.py` | `FraudTrainer` | Train LR, RF, XGBoost, LightGBM + Optuna tuning |
| **models** | `anomaly.py` | `IsolationForestDetector` | Unsupervised anomaly detection |
| **models** | `model_selection.py` | `ModelSelector` | Auto-select best model by PR-AUC |
| **prediction** | `model_loader.py` | `ModelLoader` | Load/verify model artifacts (checksum) |
| **prediction** | | `FraudPredictor` | Vectorized predictions + caching |
| **explainability** | `shap_explainer.py` | `ShapExplainer` | SHAP computation (async for high-risk) |
| **evaluation** | `metrics.py` | `FraudEvaluator` | PR-AUC, F1, confusion matrix |
| **evaluation** | `business_cost.py` | `BusinessCostCalculator` | Fraud caught/missed, net benefit |
| **llm** | `case_narrator.py` | `CaseNarrator` | LLM narrative from SHAP values |
| **llm** | `rag_similar_cases.py` | `SimilarCaseRetriever` | FAISS-based RAG retrieval |
| **monitoring** | `drift.py` | `DriftDetector` | KS-test for data drift |
| **persistence** | | SQLAlchemy models | Predictions, feedback, API keys, drift events |

### 2. API Layer (`api/`)

| Module | Responsibility |
|--------|---------------|
| `main.py` | FastAPI app creation, CORS, security headers, lifespan |
| `schemas.py` | All Pydantic request/response models (RFC 7807) |
| `providers.py` | FastAPI `Depends()` providers (DI), `FraudPredictor`, `PredictionCache` |
| `auth.py` | API key authentication (SHA-256 hashed) |
| `rate_limit.py` | SlowAPI rate limiter (Redis-backed) |
| `errors.py` | RFC 7807 error handlers |
| `routers/predict.py` | `POST /v1/predict`, `POST /v1/predict/batch` |
| `routers/explain.py` | `POST /v1/explain` |
| `routers/similar_cases.py` | `POST /v1/similar-cases` (cursor pagination) |
| `routers/chat.py` | `POST /v1/chat` (Anthropic Claude) |
| `routers/admin.py` | `GET /v1/auth/keys`, `POST /v1/auth/keys` |

### 3. Dashboard Layer (`app/`)

| Page | File | Purpose |
|------|------|---------|
| Live Monitor | `pages/live_monitor.py` | Real-time transaction feed, drift alerts |
| Case Investigator | `pages/case_investigator.py` | SHAP deep-dive, LLM narrative, RAG similar cases |
| Model Performance | `pages/model_performance.py` | Model comparison charts |
| Analyst Copilot | `pages/analyst_copilot.py` | Natural-language chat with tool-use |
| API Client | `api_client.py` | Shared HTTP client (retries, timeouts, spinners) |

## Dependency Injection (Providers)

All shared services are initialized in `api/main.py`'s lifespan and accessed via `api/providers.py`:

```python
# Example: FastAPI Depends() injection
@router.post("/v1/predict")
async def predict_single(
    request: Request,
    transaction: TransactionInput,
    predictor: FraudPredictor = Depends(get_predictor),
    api_key: str = Depends(require_api_key),
) -> PredictionResponse:
    ...
```

No module-level mutable globals remain — all state is managed through `app.state` + `Depends()`.

## Data Flow (End-to-End)

```
1. INGEST
   creditcard.csv → DataLoader.load() → DataFrame (284,807 × 31)

2. PREPROCESS (No Data Leakage)
   DataFrame → FraudPreprocessor.split_data() → Train (80%) | Test (20%)
            → StandardScaler.fit(Time, Amount on TRAIN ONLY)
            → X_train_scaled, X_test_scaled

3. TRAIN + TUNE
   X_train_scaled → FraudTrainer.train_all() + Optuna tuning
                 → models/*.pkl (XGBoost, RF, LR, LGBM)
                 → MLflow tracking (params, metrics, artifacts)

4. EVALUATE
   X_test + models → FraudEvaluator.evaluate_model()
                  → PR-AUC, Business Cost ($)
                  → Optimal threshold → threshold.txt

5. SERVE
   FastAPI lifespan → ModelLoader (checksum verify) → app.state.predictor
   
   POST /v1/predict:
     TransactionInput → vectorize (numpy) → scale → predict_proba
                     → [optional] SHAP explain → [optional] LLM narrate
                     → [optional] cache result → PredictionResponse
   
   POST /v1/predict/batch:
     BatchInput → DataFrame (reordered columns) → predict_batch
               → BatchResponse (no SHAP for speed)

6. PERSIST
   Prediction → PostgreSQL (predictions table)
   Analyst feedback → PostgreSQL (feedback table)
   Drift events → PostgreSQL (drift_events table)

7. MONITOR
   DriftDetector(reference=training_data) → KS-test on new data
                                         → Drift report
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM features | No |
| `FRAUDLENS_API_KEYS` | Semicolon-delimited `hash=role` pairs | No |
| `DATABASE_URL` | PostgreSQL connection string | No (SQLite fallback) |
| `REDIS_URL` | Redis connection string | No |
| `LOG_LEVEL` | Logging level (default: INFO) | No |

