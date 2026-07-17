# 🏗️ Architecture — Credit Card Fraud Detection

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      BROWSER / CLIENT                               │
│  ┌─────────────────────┐          ┌─────────────────────┐          │
│  │  Streamlit Dashboard │          │   API Consumer      │          │
│  │    (Port 8501)       │          │   (curl/Postman)    │          │
│  └──────────┬──────────┘          └──────────┬──────────┘          │
│             │                                │                      │
└─────────────┼────────────────────────────────┼──────────────────────┘
              │ HTTP                           │ HTTP
              │                                │
┌─────────────▼────────────────────────────────▼──────────────────────┐
│                     DOCKER NETWORK (fraud-net)                      │
│                                                                     │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│  │   FastAPI Server      │    │   Streamlit Server               │  │
│  │   api/main.py         │    │   app/dashboard.py               │  │
│  │   Port: 8000          │    │   Port: 8501                     │  │
│  │                       │    │                                   │  │
│  │   Endpoints:          │    │   Features:                       │  │
│  │   • GET /health       │    │   • Live transaction simulation   │  │
│  │   • GET /model-info   │    │   • Fraud flagging                │  │
│  │   • POST /predict     │    │   • Business impact metrics       │  │
│  │   • POST /predict/batch│    │   • Probability distribution     │  │
│  └──────────┬───────────┘    │   • Cumulative impact charts      │  │
│             │                 └──────────┬───────────────────────┘  │
│             │                            │                          │
│             └────────────┬───────────────┘                          │
│                          │                                          │
│               ┌──────────▼──────────┐                               │
│               │   FraudPredictor    │                               │
│               │   src/predict.py    │                               │
│               │                     │                               │
│               │   • SHAP Explainer  │                               │
│               │   • Preprocessing   │                               │
│               │   • Threshold logic │                               │
│               └──────────┬──────────┘                               │
│                          │                                          │
│               ┌──────────▼──────────┐                               │
│               │   XGBoost Model     │                               │
│               │   models/xgboost.pkl│                               │
│               │   models/scaler.pkl │                               │
│               └─────────────────────┘                               │
│                                                                     │
│  ┌──────────────────────┐    (Optional profile: full)              │
│  │   MLflow Server      │                                          │
│  │   Port: 5000         │                                          │
│  └──────────────────────┘                                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### 1. Data Layer (`src/`)

| Module | Class | Responsibility |
|--------|-------|----------------|
| `data_loader.py` | `DataLoader` | Load CSV, compute statistics, extract samples |
| `preprocessing.py` | `FraudPreprocessor` | Train/test split, scaling (fit on train only) |
| `preprocessing.py` | `Resampler` | SMOTE, ADASYN, undersampling, SMOTE+Tomek |
| `features.py` | `FeatureEngineer` | Log transforms, interactions, PCA statistics |
| `train.py` | `FraudTrainer` | Train LR, RF, XGBoost, LightGBM |
| `train.py` | `IsolationForestDetector` | Unsupervised anomaly detection |
| `evaluate.py` | `FraudEvaluator` | PR-AUC, business cost, threshold optimization |
| `predict.py` | `FraudPredictor` | Prediction with SHAP explanations |

### 2. API Layer (`api/`)

| Endpoint | Method | Input | Output | Auth |
|----------|--------|-------|--------|------|
| `/health` | GET | — | `{status, model_loaded, version}` | None |
| `/model-info` | GET | — | `{model_type, threshold, features}` | None |
| `/predict` | POST | `TransactionInput` | `PredictionResponse` | None |
| `/predict/batch` | POST | `BatchInput` | `{predictions, summary}` | None |

### 3. Dashboard Layer (`app/`)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Transaction feed | Streamlit expander | Show recent transactions with fraud/legit status |
| Probability chart | Plotly histogram | Distribution of fraud probabilities |
| Impact chart | Plotly area chart | Cumulative $ saved vs $ lost vs $ review cost |
| Business metrics | Streamlit metric | Real-time KPIs (caught, missed, cost, net benefit) |
| Sidebar controls | Streamlit widgets | Simulation speed, fraud rate, batch size |

### 4. Monitoring Layer (`monitoring/`)

| Module | Class | Purpose |
|--------|-------|---------|
| `drift_detection.py` | `DriftDetector` | KS-test for data drift per feature |
| `drift_detection.py` | `simulate_drift` | Generate drifted data for testing |

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
