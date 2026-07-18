# 🏗️ Architecture — Credit Card Fraud Detection

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      BROWSER / CLIENT                                   │
│  ┌─────────────────────────────┐      ┌───────────────────────┐        │
│  │  Streamlit Dashboard         │      │   API Consumer        │        │
│  │  app/streamlit_app.py        │      │   (curl/Postman)      │        │
│  │  (Port 8501)                 │      │                       │        │
│  │  4 pages:                    │      └───────────┬───────────┘        │
│  │  • Live Monitor              │                  │                    │
│  │  • Case Investigator         │                  │                    │
│  │  • Model Performance         │                  │                    │
│  │  • Analyst Copilot           │                  │                    │
│  └─────────────┬───────────────┘                  │                    │
│                │ HTTP (localhost:8000)             │ HTTP               │
└────────────────┼──────────────────────────────────┼────────────────────┘
                 │                                  │
┌────────────────▼──────────────────────────────────▼────────────────────┐
│                          DOCKER NETWORK (fraud-net)                    │
│                                                                        │
│  ┌──────────────────────────────────┐  ┌───────────────────────────┐   │
│  │   FastAPI Server                 │  │   Streamlit Server        │   │
│  │   api/main.py                    │  │   Port: 8501              │   │
│  │   Port: 8000                     │  │                           │   │
│  │                                  │  └───────────────────────────┘   │
│  │   Routers:                       │                                  │
│  │   ┌─────────────────────────┐   │                                  │
│  │   │ predict.py              │   │                                  │
│  │   │  • POST /predict        │   │                                  │
│  │   │  • POST /predict/batch  │   │                                  │
│  │   ├─────────────────────────┤   │                                  │
│  │   │ explain.py              │   │                                  │
│  │   │  • POST /explain        │   │                                  │
│  │   ├─────────────────────────┤   │                                  │
│  │   │ similar_cases.py        │   │                                  │
│  │   │  • POST /similar-cases  │   │                                  │
│  │   ├─────────────────────────┤   │                                  │
│  │   │ chat.py                 │   │                                  │
│  │   │  • POST /chat           │   │                                  │
│  │   └─────────────────────────┘   │                                  │
│  └──────────────┬───────────────────┘                                  │
│                 │                                                      │
│                 │                                                      │
│      ┌──────────▼──────────┐                                           │
│      │   FraudPredictor    │                                           │
│      │   shap_utils.py     │                                           │
│      │                     │                                           │
│      │   • SHAP Explainer  │                                           │
│      │   • Preprocessing   │                                           │
│      │   • Threshold logic │                                           │
│      └──────────┬──────────┘                                           │
│                 │                                                      │
│      ┌──────────▼──────────┐                                           │
│      │   XGBoost Model     │                                           │
│      │   best_fraud_model  │                                           │
│      │   anomaly_detector  │                                           │
│      │   scaler.pkl        │                                           │
│      └─────────────────────┘                                           │
│                                                                        │
│  ┌──────────────────────┐    (Optional profile: full)                  │
│  │   MLflow Server      │                                              │
│  │   Port: 5000         │                                              │
│  └──────────────────────┘                                              │
└────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### 1. Core Library (`src/fraudshield/`)

The library is organized as a Python package with subpackages:

| Package | Module | Class | Responsibility |
|---------|--------|-------|----------------|
| **config** | `config.py` | — | Centralized constants: paths, costs, model params, features |
| **data** | `loaders.py` | `DataLoader` | Load CSV, compute statistics, extract samples |
| **data** | `preprocessing.py` | `FraudPreprocessor` | Train/test split, scaling (fit on train only) |
| **data** | `preprocessing.py` | `Resampler` | SMOTE, ADASYN, undersampling, SMOTE+Tomek |
| **features** | `engineering.py` | `FeatureEngineer` | Log transforms, interactions, PCA statistics |
| **models** | `train.py` | `FraudTrainer` | Train LR, RF, XGBoost, LightGBM, CatBoost |
| **models** | `anomaly.py` | `IsolationForestDetector` | Unsupervised anomaly detection |
| **models** | `anomaly.py` | `AutoencoderDetector` | Neural-network based anomaly detection |
| **models** | `model_selection.py` | `ModelSelector` | Auto-select best model by PR-AUC |
| **evaluation** | `metrics.py` | `FraudEvaluator` | PR-AUC, ROC, F1, confusion matrix |
| **evaluation** | `business_cost.py` | `BusinessCostCalculator` | Fraud caught/missed, review cost, net benefit |
| **explainability** | `shap_utils.py` | `FraudPredictor` | Prediction with SHAP explanations |
| **analysis** | `eda.py` | — | Exploratory data analysis utilities |
| **llm** | `case_narrator.py` | `CaseNarrator` | LLM-generated plain-English case narratives |
| **llm** | `rag_similar_cases.py` | `SimilarCaseRetriever` | FAISS-based RAG for similar case retrieval |
| **monitoring** | `drift.py` | `DriftDetector` | KS-test for data drift per feature |

### 2. API Layer (`api/`)

| Router | File | Endpoints | 
|--------|------|-----------|
| Predictions | `routers/predict.py` | `POST /predict`, `POST /predict/batch` |
| Explainability | `routers/explain.py` | `POST /explain` |
| Similar Cases | `routers/similar_cases.py` | `POST /similar-cases` |
| Copilot Chat | `routers/chat.py` | `POST /chat` |

| Endpoint | Method | Input | Output | Notes |
|----------|--------|-------|--------|-------|
| `/health` | GET | — | `{status, model_loaded, version}` | Health check |
| `/model-info` | GET | — | `{model_type, threshold, features}` | Model metadata |
| `/predict` | POST | `TransactionInput` | `PredictionResponse` | + SHAP explanation |
| `/predict/batch` | POST | `BatchInput` | `{predictions, summary}` | Faster, no SHAP |
| `/explain` | POST | `TransactionInput` | `{shap_values, narrative}` | + LLM narrative |
| `/similar-cases` | POST | transaction features | `{similar_cases[]}` | RAG retrieval |
| `/chat` | POST | `{message, history}` | `{response, tool_calls}` | Analyst copilot |

Supporting modules:
- `schemas.py` — All Pydantic request/response models
- `state.py` — Global state management (model, detector, narrator, retriever, copilot)
- `main.py` — App creation, CORS, lifespan (model loading)

### 3. Dashboard Layer (`app/`)

The dashboard is a multi-page Streamlit application:

| Page | File | Purpose |
|------|------|---------|
| Live Monitor | `pages/live_monitor.py` | Real-time transaction simulation, fraud flagging, drift alerts |
| Case Investigator | `pages/case_investigator.py` | Deep-dive into flagged transactions (SHAP + LLM narrative) |
| Model Performance | `pages/model_performance.py` | Model comparison charts, PR-AUC, business impact |
| Analyst Copilot | `pages/analyst_copilot.py` | Natural-language Q&A about simulation state |

| Component | File | Purpose |
|-----------|------|---------|
| Metric Cards | `components/metric_cards.py` | Reusable styled metric and status-chip components |
| Theme | `assets/theme.css` | Dark theme styling for the whole app |
| Entry Point | `streamlit_app.py` | Multi-page navigation hub, loads page config |

### 4. Monitoring (`src/fraudshield/monitoring/`)

| Module | Class/Function | Purpose |
|--------|----------------|---------|
| `drift.py` | `DriftDetector` | KS-test for data drift per feature, alert levels (OK/WARNING/CRITICAL) |
| `drift.py` | `simulate_drift` | Generate drifted data for testing |

## Deployment Architecture

### Docker Compose Services

```yaml
services:
  api:
    ports: 8000
    healthcheck: curl http://localhost:8000/health
    volumes: ./models, ./data
    
  dashboard:
    ports: 8501
    depends_on: api (healthy)
    volumes: ./models, ./data
    
  mlflow:  # optional
    ports: 5000
    profile: full
```

### Network

All services communicate over `fraud-net` (bridge driver).

### Volume Mounts

- `./models` → `/app/models` — Shared model artifacts
- `./data` → `/app/data` — Shared data directory
