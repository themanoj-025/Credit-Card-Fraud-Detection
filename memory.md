# 🧠 Memory — FraudLens Project

> **Permanent brain of the project.** A completely new engineer should be able to understand everything about this system from this document alone.

---

## 1. Project Overview

### What It Does

FraudLens is a **production-grade credit card fraud detection system** that:

- Trains **XGBoost, LightGBM, Random Forest, Logistic Regression** models
- Serves predictions via a **FastAPI REST API** with SHAP-based explainability
- Provides **LLM-powered case narratives** (via Anthropic Claude)
- Retrieves **similar historical cases** using FAISS-based RAG
- Features an **AI analyst copilot** with tool-use chat
- Displays results in a **real-time Streamlit dashboard**
- Monitors for **data drift** over time
- Persists predictions, feedback, and API keys to **PostgreSQL**
- Deploys via **Docker Compose** or **Kubernetes**

### Why It Exists

**Business Problem:** Credit card fraud costs banks billions. This system aims to:
- **Maximize fraud caught** (each missed fraud = ~$150 loss)
- **Minimize false positives** (each flagged transaction = ~$5 manual review cost)
- Provide **explainable decisions** so fraud analysts trust the model

### Key Design Philosophy

- **PR-AUC over ROC-AUC** — ROC-AUC is misleading on 99.8%-imbalanced data
- **Train/test split BEFORE resampling** — SMOTE on full data leaks test information
- **Business cost function** — thresholds optimized by net dollars saved, not default 0.5
- **SHAP explainability** — every prediction comes with "why" this was flagged
- **Dependency Injection** — no module-level globals, all services via `Depends()`
- **API versioning** — all endpoints under `/v1/`
- **RFC 7807 errors** — consistent error format everywhere

---

## 2. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | All code |
| **ML** | XGBoost, LightGBM, scikit-learn, imbalanced-learn | Model training & prediction |
| **Tuning** | Optuna | Hyperparameter optimization |
| **Explainability** | SHAP (TreeExplainer) | Per-prediction explanations |
| **LLM** | Anthropic Claude (via SDK) | Case narration + copilot chat |
| **RAG** | FAISS (cosine similarity) | Similar case retrieval |
| **API** | FastAPI + Pydantic v1 | REST API with validation |
| **Auth** | API key (SHA-256) + Admin key | Two-tier access control |
| **Rate Limiting** | slowapi (Redis-backed) | Request throttling |
| **Database** | PostgreSQL + SQLAlchemy (async) + Alembic | System of record |
| **Cache** | In-memory LRU (PredictionCache) | Prediction dedup |
| **Dashboard** | Streamlit + Plotly | Interactive monitoring |
| **Drift Detection** | SciPy (KS-test) | Data drift monitoring |
| **Observability** | structlog, Prometheus, Jaeger | Logging, metrics, tracing |
| **Container** | Docker + docker-compose | Multi-service orchestration |
| **Orchestration** | Kubernetes (manifests) | Production deployment |
| **CI/CD** | GitHub Actions (lint → test → scan → deploy) | Automation |
| **Testing** | pytest + pytest-cov + respx + locust | Unit, integration, load, contract |
| **Serialization** | joblib | Model persistence |

---

## 3. Repository Structure

```
Credit Card Fraud Detection/
├── src/fraudlens/               # Core ML library
│   ├── api/                     # FastAPI app, routers, DI providers
│   ├── config/                  # pydantic-settings config
│   ├── data/                    # Data loading, preprocessing
│   ├── features/                # Feature engineering
│   ├── models/                  # Training, anomaly, model selection
│   ├── prediction/              # ModelLoader, FraudPredictor
│   ├── explainability/          # ShapExplainer
│   ├── llm/                     # CaseNarrator, RAG retriever
│   ├── monitoring/              # Drift detection
│   ├── persistence/             # SQLAlchemy models, migrations, repos
│   └── common/                  # Logging, exceptions, enums
├── api/                         # FastAPI routes, schemas, providers
│   ├── main.py                  # App creation, lifespan, middleware
│   ├── providers.py             # DI providers + FraudPredictor class
│   ├── schemas.py               # Pydantic models (v1)
│   ├── auth.py                  # API key authentication
│   ├── rate_limit.py            # slowapi limiter
│   ├── errors.py                # RFC 7807 error handlers
│   └── routers/                 # Endpoint definitions
│       ├── predict.py           # POST /v1/predict, /v1/predict/batch
│       ├── explain.py           # POST /v1/explain
│       ├── similar_cases.py     # POST /v1/similar-cases (cursor pagination)
│       ├── chat.py              # POST /v1/chat
│       └── admin.py             # GET/POST /v1/auth/keys
├── app/                         # Streamlit dashboard
│   ├── streamlit_app.py         # Multi-page entry point
│   ├── api_client.py            # Shared HTTP client (retries, spinners)
│   ├── components/              # Metric cards, status chips
│   ├── pages/                   # Live Monitor, Investigator, Performance, Copilot
│   └── assets/                  # CSS theme
├── infra/
│   ├── k8s/                     # K8s: deployment, service, ingress, hpa, configmap, secret
│   └── docker/                  # Docker configs
├── tests/                       # Test suite (248+ tests)
│   ├── conftest.py              # Shared fixtures (mock_anthropic, sample_transaction)
│   ├── test_api.py              # API endpoint tests
│   ├── test_integration.py      # End-to-end training→prediction
│   ├── test_edge_cases.py       # NaN/Inf/boundary/empty batch
│   ├── test_contract.py         # OpenAPI schema contract tests
│   ├── test_preprocessing.py    # Preprocessing pipeline
│   └── ...                      # 14 total test modules
├── docs/
│   ├── adr/                     # Architecture Decision Records
│   ├── MODEL_CARD.md            # Model card
│   └── SYSTEM_DESIGN.md         # System design writeup
├── .github/workflows/           # CI/CD: lint → test → build → scan → deploy
├── Dockerfile                   # Multi-stage: serve (~400MB) + train
├── docker-compose.yml           # API + Dashboard + Postgres + Redis + MLflow
├── Makefile                     # Dev commands
└── requirements.txt             # Python dependencies
```

---

## 4. Data Flow (End-to-End)

```
1. INGEST
   creditcard.csv → DataLoader.load() → DataFrame (284,807 × 31)

2. PREPROCESS (No Data Leakage)
   DataFrame → FraudPreprocessor.split_data() [stratified, 80/20]
            → StandardScaler.fit(Time, Amount on TRAIN ONLY)
            → X_train_scaled, X_test_scaled

3. TRAIN + TUNE
   X_train_scaled → FraudTrainer.train_all()
                  → Optuna tuning (30 trials, 3-fold CV)
                  → models/*.pkl → MLflow tracking

4. EVALUATE
   X_test + models → FraudEvaluator.evaluate_model()
                  → PR-AUC, F1, Business Cost
                  → Optimal threshold via cost minimization → threshold.txt

5. SERVE (FastAPI)
   app.state.predictor = FraudPredictor(ModelLoader, ShapExplainer)
   
   POST /v1/predict:
     TransactionInput → vectorize (numpy, ~5µs) → scale → predict_proba
                     → [optional] SHAP (if ?explain=true or high-risk background task)
                     → [optional] cache result → PredictionResponse

   POST /v1/predict/batch:
     BatchInput → DataFrame (column-reordered) → predict_batch → BatchResponse

6. EXPLAIN (LLM)
   SHAP values → CaseNarrator.narrate()
              → Plain-English: "Flagged due to V14 (-5.23, +0.34 increase)..."

7. RAG RETRIEVE
   Transaction features → FAISS index → Top-20 similar cases → SimilarCasesResponse

8. PERSIST
   Prediction → PostgreSQL (predictions table)
   Feedback → PostgreSQL (feedback table)
   Drift → PostgreSQL (drift_events table)

9. MONITOR
   DriftDetector(reference=training_data) → KS-test on new data → drift events
```

---

## 5. Architecture

### Three-Layer Design

| Layer | Component | Tech |
|-------|-----------|------|
| **Detection** | XGBoost + Isolation Forest | Traditional ML |
| **Explanation** | SHAP + LLM Narrator | Structured + Generative AI |
| **Interaction** | RAG Retrieval + Copilot Chat | Vector Search + LLM |

### Dependency Injection

All services are managed via `app.state` + FastAPI `Depends()`:

```python
# Providers in api/providers.py:
get_predictor()        → FraudPredictor instance
get_anomaly_detector() → IsolationForestDetector
get_case_narrator()    → CaseNarrator (LLM)
get_case_retriever()   → SimilarCaseRetriever (FAISS)
get_copilot_client()   → Anthropic client
get_db_session()       → Async SQLAlchemy session
```

No module-level `_predictor = None` patterns exist.

### Key Design Decisions (ADRs)

1. **DI over globals** — `api/state.py` deleted, replaced with `Depends()` providers
2. **Async SHAP** — SHAP off hot path, computed via BackgroundTasks for high-risk transactions
3. **Vectorized prediction** — numpy array path for single predictions (~10x faster than DataFrame)
4. **Column reordering** — batch predictions reorder columns to match model's expected order (V1-V28 first)
5. **RFC 7807 errors** — consistent problem-details format on all error responses
6. **Cursor pagination** — `/v1/similar-cases` uses cursor-based pagination instead of offset/limit

---

## 6. Completed Phases

| Phase | Focus | Key Changes |
|-------|-------|-------------|
| **0** | Baseline & Safety Net | Rename `fraudshield` → `fraudlens`, pre-commit, Makefile, baseline snapshot |
| **1** | Security | API key auth (SHA-256), slowapi rate limiting, Pydantic validation, security headers, CORS locked, SECURITY.md |
| **2** | Architecture & Code Quality | Delete globals → DI providers, split FraudPredictor (ModelLoader + Predictor + ShapExplainer), enums, SRP, mypy |
| **3** | Data Layer | PostgreSQL + Alembic migrations, SQLAlchemy models, repository pattern, async sessions |
| **4** | Performance | Async SHAP via BackgroundTasks, LRU PredictionCache, vectorized numpy path, locust load tests |
| **5** | ML Quality | FeatureEngineer wired into train.py, Optuna tuning, LLM eval harness, docs/MODEL_CARD.md |
| **6** | API Design | `/v1/` versioning, RFC 7807 errors, cursor pagination, per-dependency health check, OpenAPI spec |
| **7** | Frontend | Shared api_client.py (retries, timeouts), case_investigator uses real API, !important CSS fixed, loading spinners |
| **8** | Testing | conftest.py fixtures, integration tests, NaN/Inf edge cases, OpenAPI contract tests, 85% coverage target |
| **9** | DevOps | Multi-stage Docker (~400MB serve image), CI/CD (lint→test→build→scan→deploy), K8s manifests, Trivy scan |
| **10** | Observability | 🔜 |
| **11** | Configuration | 🔜 |
| **12** | Error Handling | 🔜 |
| **13** | Docs & Polish | 🔜 |

---

## 7. Authentication & Authorization

| Feature | Implementation |
|---------|---------------|
| API Key Auth | `X-API-Key` header, SHA-256 hashed, compared against `FRAUDLENS_API_KEYS` env var |
| Key Format | `fl_` + 48 hex chars (cryptographically random via `secrets.token_hex`) |
| Key Tiers | `admin` (can manage keys) + `readonly` (predictions only) |
| Admin Endpoints | `POST /v1/auth/keys` (generate), `GET /v1/auth/keys` (list) |
| Rate Limits | Per-endpoint limits via slowapi |

### Auth Flow

1. Client sends `X-API-Key: fl_abc123...` header
2. Server SHA-256 hashes the key
3. Compares hash against `FRAUDLENS_API_KEYS` (semicolon-delimited `hash=role` pairs)
4. Valid → route handler executes
5. Invalid → 401 Unauthorized (RFC 7807 format)

---

## 8. Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM features | — | No |
| `FRAUDLENS_API_KEYS` | Semicolon-delimited `sha256hash=role` pairs | — | No |
| `DATABASE_URL` | PostgreSQL connection string | SQLite fallback | No |
| `REDIS_URL` | Redis connection string | In-memory fallback | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |

---

## 9. Critical Files

| File | Why Critical |
|------|-------------|
| `src/fraudlens/data/preprocessing.py` | Data leakage prevention — split/scale logic |
| `src/fraudlens/prediction/model_loader.py` | Model artifact loading + checksum verification |
| `src/fraudlens/prediction/` | FraudPredictor — prediction pipeline |
| `api/providers.py` | DI providers — FraudPredictor, PredictionCache |
| `api/main.py` | App creation, lifespan middleware |
| `api/schemas.py` | Pydantic models — external API contract |
| `api/routers/predict.py` | Prediction endpoints |
| `tests/conftest.py` | Test fixtures — mock_anthropic, sample data |

---

## 10. Known Technical Debt

1. **No automated data download** — user must manually download `creditcard.csv` from Kaggle
2. **Feature engineering underused** — `FeatureEngineer` exists but isn't integrated into the main pipeline
3. **Autoencoder not trained** — IsolationForestDetector is the only anomaly detector; AutoencoderDetector is untrained
4. **No database-backed rate limiting** — slowapi uses in-memory by default (Redis for production)
5. **No structured logging** — still using `logging` module instead of `structlog`
6. **No Prometheus metrics** — no request/response metrics emitted
7. **No OpenTelemetry tracing** — no distributed tracing
8. **Config still hardcoded** — `config.py` constants not yet migrated to `pydantic-settings`
9. **No retry/backoff on LLM calls** — Anthropic calls can fail without recovery
10. **No circuit breaker** — LLM outage affects all `/explain` requests

---

## 11. Development Workflow

```bash
# Setup
make install
make install-dev

# Train
make train

# Run
make api     # FastAPI on :8000
make dashboard  # Streamlit on :8501

# Test
make test         # All tests
make test-cov     # With coverage (85% target)
make test-integration
make test-contract

# Lint (blocking)
make lint

# Docker
make docker-up      # Full stack
make docker-up --profile training  # + MLflow

# K8s (preview)
make k8s-dry-run
make k8s-apply
```
