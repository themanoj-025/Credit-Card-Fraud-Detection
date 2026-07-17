# 🧠 Memory — Credit Card Fraud Detection Project

> **Permanent brain of the project.** A completely new engineer should be able to understand everything about this system from this document alone.

---

## 1. Project Overview

### What It Does
A **production-grade credit card fraud detection system** that:
- Trains multiple ML models (XGBoost, LightGBM, Random Forest, Logistic Regression, Isolation Forest)
- Serves predictions via a REST API with SHAP-based explainability
- Displays results in a real-time Streamlit dashboard
- Monitors for data drift over time

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

---

## 2. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10 | All code |
| **ML** | XGBoost, LightGBM, scikit-learn, imbalanced-learn | Model training & prediction |
| **Explainability** | SHAP (TreeExplainer) | Per-prediction explanations |
| **API** | FastAPI + Pydantic | REST API with validation |
| **Dashboard** | Streamlit + Plotly | Interactive monitoring |
| **Visualization** | Matplotlib, Seaborn, Plotly | EDA and result charts |
| **Drift Detection** | SciPy (KS-test) | Data drift monitoring |
| **Experiment Tracking** | MLflow (optional) | Model versioning |
| **Containerization** | Docker + docker-compose | Deployment |
| **Testing** | pytest + pytest-cov | Unit and smoke tests |
| **Serialization** | joblib | Model and scaler persistence |

---

## 3. Repository Structure

```
Credit Card Fraud Detection/
├── Dataset/
│   └── Dataset/
│       └── creditcard.csv          # Source data (284,807 rows, 31 columns)
├── src/                             # Core Python modules
│   ├── __init__.py                  # Package init (version 1.0.0)
│   ├── data_loader.py               # DataLoader class, load_data() convenience fn
│   ├── preprocessing.py             # FraudPreprocessor, Resampler, get_class_weights()
│   ├── features.py                  # FeatureEngineer class
│   ├── train.py                     # FraudTrainer, IsolationForestDetector
│   ├── evaluate.py                  # FraudEvaluator, print_evaluation_summary()
│   └── predict.py                   # FraudPredictor with SHAP explanations
├── api/
│   ├── __init__.py
│   └── main.py                      # FastAPI app with /predict, /health, /model-info
├── app/
│   ├── __init__.py
│   └── dashboard.py                 # Streamlit dashboard with live simulation
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py        # 11 tests for preprocessing pipeline
│   └── test_api.py                  # 7 tests for API endpoints
├── monitoring/
│   └── drift_detection.py           # DriftDetector class with KS-test
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb       # Split, scale, resampling comparison
│   ├── 03_modeling.ipynb            # Model training & evaluation
│   └── 04_explainability.ipynb      # SHAP global & per-prediction explanations
├── data/
│   ├── raw/                         # (gitignored) Raw CSV
│   └── processed/                   # Charts, saved splits, results CSVs
├── models/                          # Saved .pkl model artifacts (gitignored)
├── Dockerfile                       # Multi-stage build
├── docker-compose.yml               # API + Dashboard + MLflow services
├── requirements.txt                 # Pinned Python dependencies
├── .gitignore                       # Ignores data/raw, models/*.pkl, etc.
└── README.md                        # Case study format documentation
```

---

## 4. Data Flow (End-to-End)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. INGEST                                                         │
│     creditcard.csv → DataLoader.load() → DataFrame                │
│                                                                     │
│  2. PREPROCESS                                                     │
│     DataFrame → FraudPreprocessor.split_data()                     │
│       → X_train (80%), X_test (20%) [stratified]                   │
│       → FraudPreprocessor.fit_scale() [Scaler fit on train ONLY]  │
│       → X_train_scaled, X_test_scaled                              │
│                                                                     │
│  3. RESAMPLE (train only)                                          │
│     X_train_scaled → Resampler.resample(strategy='smote')          │
│       → X_train_resampled (increased minority)                     │
│                                                                     │
│  4. TRAIN                                                          │
│     X_train_resampled → FraudTrainer.train_all()                   │
│       → models/*.pkl (XGBoost, LightGBM, RF, LR)                  │
│       → models/scaler.pkl                                          │
│                                                                     │
│  5. EVALUATE                                                       │
│     X_test_scaled + models → FraudEvaluator.evaluate_model()       │
│       → PR-AUC, F1, Business Cost ($ saved/lost)                  │
│       → Optimal threshold from cost function                       │
│       → models/threshold.txt                                       │
│                                                                     │
│  6. EXPLAIN                                                        │
│     X_test + model → FraudPredictor.predict_single()               │
│       → SHAP TreeExplainer.shap_values()                           │
│       → "Flagged due to V14, V4, V12"                             │
│                                                                     │
│  7. SERVE                                                          │
│     models/*.pkl → FastAPI loads at startup                        │
│       → POST /predict → {fraud_probability, explanation}          │
│       → Streamlit dashboard (live simulation)                      │
│                                                                     │
│  8. MONITOR                                                        │
│     Training data → DriftDetector (reference)                     │
│       → New transactions → KS-test → Drift Report                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                        ┌─────────────┐                              │
│                        │   Browser   │                              │
│                        └──────┬──────┘                              │
│                               │                                      │
│                    ┌──────────┼──────────┐                          │
│                    │          │          │                           │
│              ┌─────▼────┐  ┌─▼────────┐ │                          │
│              │ Streamlit │  │  FastAPI  │ │                          │
│              │ Dashboard │  │   :8000   │ │                          │
│              │   :8501   │  └────┬─────┘ │                          │
│              └─────┬────┘       │       │                           │
│                    │      ┌─────▼──────┐ │                          │
│                    │      │ FraudPredictor│                          │
│                    │      │   (SHAP)    │ │                          │
│                    │      └─────┬──────┘ │                          │
│                    │            │         │                          │
│                    │      ┌─────▼──────┐ │                          │
│                    │      │ XGBoost    │ │                          │
│                    │      │ Model.pkl  │ │                          │
│                    │      └─────┬──────┘ │                          │
│                    │            │         │                          │
│                    │      ┌─────▼──────┐ │                          │
│                    │      │  Scaler    │ │                          │
│                    │      │  .pkl      │ │                          │
│                    │      └─────┬──────┘ │                          │
│                    │            │         │                          │
│              ┌─────▼────────────▼─────────▼──────┐                  │
│              │         creditcard.csv             │                  │
│              │      (284,807 transactions)       │                  │
│              └──────────────────────────────────┘                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  Docker Compose                                         │       │
│  │  ├── api (FastAPI :8000)                                │       │
│  │  ├── dashboard (Streamlit :8501)                        │       │
│  │  └── mlflow (optional :5000)                            │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Authentication & Authorization

**Current:** No authentication implemented. The API is open.

**Future considerations:**
- FastAPI API key authentication
- Streamlit password protection
- Role-based access (analyst vs admin)

---

## 7. Environment Variables & Configuration

No `.env` file is used. Configuration is hardcoded:

| Variable | Location | Default | Purpose |
|----------|----------|---------|---------|
| `avg_fraud_loss` | `data_loader.py`, `evaluate.py` | $150 | Cost per missed fraud |
| `review_cost` | `data_loader.py`, `evaluate.py` | $5 | Cost per manual review |
| `test_size` | `preprocessing.py` | 0.2 | Train/test split ratio |
| `random_state` | All modules | 42 | Reproducibility seed |
| `contamination` | `train.py` | 0.01 | Isolation Forest anomaly ratio |
| `threshold` | `predict.py` | 0.5 (overridden by optimal) | Classification threshold |

**External Services:**
- MLflow tracking server (optional, via docker-compose profile `full`)
- Kaggle dataset download (manual, not automated)

---

## 8. Known Technical Debt

1. **No automated data download** — user must manually place `creditcard.csv`
2. **No MLflow integration in training** — `requirements.txt` includes it but code doesn't use it yet
3. **No authentication** on API or dashboard
4. **Streamlit session state** — transaction history limited to 500 (fixed), but business metrics accumulate without reset
5. **Batch prediction** — skips SHAP for speed, no explanation in batch responses
6. **Feature engineering** — `FeatureEngineer` module exists but isn't used in the main pipeline (notebooks use it manually)

---

## 9. Critical Files (Don't Modify Lightly)

| File | Why Critical |
|------|-------------|
| `src/preprocessing.py` | **Data leakage prevention** — train/test split and scaler logic must stay correct |
| `src/evaluate.py` | **Business cost function** — threshold optimization logic |
| `src/predict.py` | **SHAP integration** — explainer initialization and prediction pipeline |
| `api/main.py` | **API contract** — Pydantic models define the external API |
| `tests/test_preprocessing.py` | **Regression tests** — catches data leakage bugs |

---

## 10. Development Workflow

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Run EDA
jupyter notebook notebooks/01_eda.ipynb

# 3. Preprocess (generates data/processed/*.pkl)
jupyter notebook notebooks/02_preprocessing.ipynb

# 4. Train models (generates models/*.pkl)
jupyter notebook notebooks/03_modeling.ipynb

# 5. Explain (generates SHAP plots)
jupyter notebook notebooks/04_explainability.ipynb

# 6. Test
pytest tests/ -v

# 7. Serve
uvicorn api.main:app --reload
streamlit run app/dashboard.py

# 8. Docker
docker-compose up --build
```

---

## 11. Future Recommendations

1. **Add MLflow tracking** to `train.py` — log params, metrics, artifacts per run
2. **Add GitHub Actions CI** — run pytest on every push
3. **Add autoencoder** — unsupervised baseline (listed in requirements but not implemented)
4. **Graph-based detection** — model transactions as graph for fraud ring detection
5. **Streaming simulation** — Kafka for real-time transaction scoring
6. **Customer fairness audit** — false-positive rates across demographics
7. **Deploy to cloud** — Render/Railway for API, Streamlit Community Cloud for dashboard
