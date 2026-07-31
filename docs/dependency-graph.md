# 🔗 Dependency Graph — FraudLens

## Module Dependency Map

```
api/main.py (FastAPI app, lifespan, middleware)
├── api/auth.py           → SHA-256 key validation
├── api/rate_limit.py     → slowapi limiter
├── api/errors.py         → RFC 7807 error handlers
├── api/providers.py      → DI: FraudPredictor, anomaly, LLM, DB
│   ├── src/fraudlens/prediction/model_loader.py
│   │   ├── joblib (.pkl loading)
│   │   ├── hashlib (checksum)
│   │   └── sklearn.preprocessing (StandardScaler)
│   ├── src/fraudlens/explainability/shap_explainer.py
│   │   └── shap (TreeExplainer)
│   ├── src/fraudlens/llm/case_narrator.py
│   │   └── anthropic (Anthropic SDK)
│   ├── src/fraudlens/llm/rag_similar_cases.py
│   │   └── faiss (RAG retrieval)
│   └── src/fraudlens/models/anomaly.py
│       └── sklearn.ensemble (IsolationForest)
│
├── api/routers/predict.py
│   └── api/providers (get_predictor, get_anomaly_detector)
├── api/routers/explain.py
│   └── api/providers (get_predictor, get_case_narrator)
├── api/routers/similar_cases.py
│   └── api/providers (get_case_retriever)
├── api/routers/chat.py
│   └── api/providers (get_copilot_client)
├── api/routers/admin.py
│   ├── api/auth (require_admin_key)
│   └── src.fraudlens.llm.cost_tracker (LLM usage endpoint)
├── api/metrics.py
│   └── fraudlens_llm_cost_usd_total, fraudlens_llm_tokens_total, fraudlens_llm_calls_total

│
└── src/fraudlens/persistence/ (DB initialization)
    ├── sqlalchemy.ext.asyncio (AsyncSession)
    └── alembic (migrations)

src/fraudlens/models/train.py (FraudTrainer)
├── xgboost
├── lightgbm
├── sklearn.ensemble (RandomForest)
├── sklearn.linear_model (LogisticRegression)
├── optuna (hyperparameter tuning)
└── mlflow (experiment tracking)

src/fraudlens/data/preprocessing.py
├── sklearn.model_selection (train_test_split)
├── sklearn.preprocessing (StandardScaler)
└── imblearn.over_sampling (SMOTE)

src/fraudlens/features/engineering.py
├── FeatureEngineer
│   ├── pandas (DataFrame operations)
│   └── numpy (vectorized ops)
└── sklearn.preprocessing (KBinsDiscretizer)

src/fraudlens/evaluation/metrics.py
└── sklearn.metrics (precision_recall_curve, f1, etc.)

src/fraudlens/evaluation/business_cost.py
└── BusinessCostCalculator (pure numpy)

src/fraudlens/monitoring/drift.py
└── scipy.stats (ks_2samp)

tests/conftest.py
├── pytest (fixtures)
├── fastapi.testclient (TestClient)
└── anthropic (mock Anhtropic)

tests/test_contract.py
└── fastapi (app.openapi)

app/api_client.py
├── httpx (async HTTP)
├── manual retry with backoff (built-in)
└── streamlit (st.spinner, st.cache_resource)
```

## Critical Dependency Chains

```
1. Prediction Hot Path (single):
   FastAPI → get_predictor() → FraudPredictor.predict_single()
     → ModelLoader.preprocess_numpy() [~5µs]
     → Model.predict_proba() [~500µs]
     → [optional] ShapExplainer.explain() [~100ms]
     → [optional] PredictionCache.set() [~1µs]
   Total: ~500µs (fast) / ~101ms (with SHAP)

2. Prediction Hot Path (batch):
   FastAPI → FraudPredictor.predict_batch()
     → Column reorder (prevent train/serve skew)
     → ModelLoader.preprocess() [DataFrame path]
     → Model.predict_proba()
   Total: ~5ms for 100 transactions

3. LLM Narrative:
   FastAPI → get_case_narrator() → CaseNarrator.narrate()
     → Anthropic messages.create() [~500ms–2s]
   Note: With tenacity retries + circuit breaker

4. RAG Retrieval:
   FastAPI → get_case_retriever() → SimilarCaseRetriever.retrieve()
     → FAISS index.search() [~1ms for top-20]
   Total: ~1ms

5. Auth:
   FastAPI → require_api_key() → hashlib.sha256()
     → Compare against FRAUDLENS_API_KEYS
   Total: ~10µs

6. Rate Limiting:
   slowapi middleware → check Redis/in-memory counter
   Total: ~1ms
```

## Layer Dependencies

```
┌──────────────────────────────────────────────────────────────┐
│                         INFRASTRUCTURE                       │
│  Docker · Docker Compose · Kubernetes · GitHub Actions       │
├──────────────────────────────────────────────────────────────┤
│                         DEPLOYMENT                           │
│  Dockerfile · docker-compose.yml · infra/k8s/ · Makefile     │
├──────────────────────────────────────────────────────────────┤
│                         API LAYER                            │
│  api/main.py · api/routers/ · api/providers.py · api/schemas │
├──────────────────────────────────────────────────────────────┤
│                   APPLICATION LAYER                          │
│  app/streamlit_app.py · app/pages/ · app/api_client.py       │
├──────────────────────────────────────────────────────────────┤
│                    PREDICTION LAYER                          │
│  FraudPredictor · ModelLoader · ShapExplainer · PredictionCache│
├──────────────────────────────────────────────────────────────┤
│                    ML PIPELINE                               │
│  FraudTrainer · FraudPreprocessor · FraudEvaluator · Optuna  │
├──────────────────────────────────────────────────────────────┤
│                    DATA LAYER                                │
│  PostgreSQL · SQLAlchemy · Alembic · Redis · FAISS           │
├──────────────────────────────────────────────────────────────┤
│                    CORE LIBRARY                              │
│  src/fraudlens/ (data, features, models, monitoring, common) │
└──────────────────────────────────────────────────────────────┘
```

## Package Dependencies (requirements.txt)

```
### Core ML
pandas, numpy, scikit-learn, xgboost, lightgbm
imbalanced-learn, shap, joblib, optuna

### Deep Learning (removed in Phase 14)
~~tensorflow, keras~~ — removed (AutoencoderDetector eliminated, ADR-0001)

catboost  (optional, not in default training)

### API
fastapi, uvicorn, pydantic, httpx, slowapi

### LLM
anthropic

### Database
sqlalchemy[asyncio], alembic, asyncpg, aiosqlite, psycopg2-binary

### Dashboard
streamlit, plotly, streamlit-option-menu

### Visualization (training/EDA)
matplotlib, seaborn, plotly, umap-learn, scipy

### Monitoring
evidently

### Experiment Tracking
mlflow

### Utilities
python-dotenv, tqdm, faiss-cpu

### Testing
pytest, pytest-cov, pytest-xdist, respx, locust

### Development
black, isort, ruff, mypy, pre-commit
```
