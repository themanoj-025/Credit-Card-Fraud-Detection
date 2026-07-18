# 🔗 Dependency Graph — Credit Card Fraud Detection

## Module Dependency Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DEPENDENCY GRAPH                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ENTRY POINTS                                                               │
│  ┌────────────────────────┐                                                 │
│  │ train_and_compare.py   │──── src/fraudshield/data/loaders.py            │
│  │ (Full training pipeline)│──── src/fraudshield/data/preprocessing.py     │
│  │                        │──── src/fraudshield/models/train.py            │
│  │                        │──── src/fraudshield/models/anomaly.py          │
│  │                        │──── src/fraudshield/models/model_selection.py  │
│  │                        │──── src/fraudshield/evaluation/metrics.py      │
│  │                        │──── src/fraudshield/evaluation/business_cost.py│
│  └────────────────────────┘                                                 │
│  ┌────────────────────────┐                                                 │
│  │ run_pipeline.py        │──── (same modules + AutoencoderDetector)       │
│  │ (Extended pipeline)    │──── src/fraudshield/monitoring/drift.py        │
│  └────────────────────────┘                                                 │
│  ┌────────────────────────┐                                                 │
│  │ api/main.py (FastAPI)  │──── src/fraudshield/explainability/shap_utils  │
│  │                        │──── src/fraudshield/llm/case_narrator.py       │
│  │                        │──── src/fraudshield/llm/rag_similar_cases.py   │
│  │                        │──── src/fraudshield/models/anomaly.py          │
│  │                        │──── src/fraudshield/config.py                  │
│  │                        │──── api/routers/(predict,explain,chat,cases)   │
│  └────────────────────────┘                                                 │
│  ┌────────────────────────┐                                                 │
│  │ app/streamlit_app.py   │──── api.main (via HTTP requests)                │
│  │ (Dashboard)            │──── app/pages/(live_monitor, case, perf, chat) │
│  │                        │──── app/components/metric_cards.py             │
│  └────────────────────────┘                                                 │
│                                                                             │
│  CORE PACKAGES (src/fraudshield/)                                           │
│                                                                             │
│  ┌────────────────────────┐                                                 │
│  │ config.py              │──── (no internal imports — all constants)       │
│  └────────────────────────┘                                                 │
│                                                                             │
│  ┌────────────────────────┐  ┌────────────────────────┐                     │
│  │ data/loaders.py        │  │ data/preprocessing.py  │                     │
│  │ pandas, numpy, joblib  │  │ sklearn (split, scale)  │                     │
│  └─────────┬──────────────┘  │ imblearn (SMOTE, ...)  │                     │
│            │                 │ joblib                  │                     │
│            │                 └──────────┬──────────────┘                     │
│            └─────────────────────────┬──┘                                    │
│                                      │                                       │
│  ┌────────────────────────┐          │                                       │
│  │ features/engineering.py│          │  ┌────────────────────────┐           │
│  │ pandas, numpy          │          │  │ models/train.py        │           │
│  └────────────────────────┘          │  │ sklearn (LR, RF, GB)   │           │
│                                      │  │ xgboost, lightgbm      │           │
│  ┌────────────────────────┐          │  │ catboost               │           │
│  │ models/anomaly.py      │          │  └──────────┬─────────────┘           │
│  │ sklearn (IsolationForest)│         │             │                          │
│  │ tensorflow (Autoencoder)│         │             │                          │
│  └────────────────────────┘          │             │                          │
│                                      │             │                          │
│  ┌────────────────────────┐          │             │                          │
│  │ models/model_selection.py│◄────────┼─────────────┘                          │
│  │ pandas                   │        │                                        │
│  └──────────────────────────┘        │                                        │
│                                      │                                        │
│  ┌──────────────────────────┐        │                                        │
│  │ evaluation/metrics.py     │◄───────┘                                        │
│  │ sklearn.metrics, matplotlib│                                              │
│  └─────────────┬────────────┘                                                │
│                │                                                              │
│  ┌─────────────▼──────────────┐                                               │
│  │ evaluation/business_cost.py│                                               │
│  │ numpy, logging             │                                               │
│  └────────────────────────────┘                                               │
│                                                                               │
│  ┌─────────────────────────────┐                                              │
│  │ explainability/shap_utils.py│                                              │
│  │ shap, joblib, pandas, numpy │                                              │
│  └──────────────┬──────────────┘                                              │
│                 │                                                             │
│  ┌──────────────▼──────────────┐                                              │
│  │ llm/case_narrator.py        │                                              │
│  │ anthropic SDK                │                                              │
│  └──────────────┬──────────────┘                                              │
│                 │                                                             │
│  ┌──────────────▼──────────────┐                                              │
│  │ llm/rag_similar_cases.py    │                                              │
│  │ faiss, numpy                 │                                              │
│  └─────────────────────────────┘                                              │
│                                                                               │
│  ┌──────────────────────────────┐                                             │
│  │ monitoring/drift.py          │                                             │
│  │ scipy.stats, pandas, numpy   │                                             │
│  └──────────────────────────────┘                                             │
│                                                                               │
│  API ROUTERS (api/routers/)                                                   │
│  ┌────────────────────────┐  ┌────────────────────────┐                       │
│  │ predict.py             │  │ explain.py             │                       │
│  │ api.schemas            │  │ api.schemas             │                       │
│  │ api.state              │  │ api.state               │                       │
│  │ FraudPredictor         │  │ FraudPredictor          │                       │
│  └────────────────────────┘  │ CaseNarrator            │                       │
│                               └────────────────────────┘                       │
│  ┌────────────────────────┐  ┌────────────────────────┐                       │
│  │ similar_cases.py       │  │ chat.py                │                       │
│  │ api.state              │  │ api.state              │                       │
│  │ RAG retriever          │  │ Anthropic client       │                       │
│  └────────────────────────┘  └────────────────────────┘                       │
│                                                                               │
│  APP PAGES (app/pages/)                                                       │
│  ┌────────────────────────┐  ┌────────────────────────┐                       │
│  │ live_monitor.py        │  │ case_investigator.py   │                       │
│  │ httpx (→ API)          │  │ httpx (→ API)          │                       │
│  │ plotly, pandas         │  │ plotly                 │                       │
│  └────────────────────────┘  └────────────────────────┘                       │
│  ┌────────────────────────┐  ┌────────────────────────┐                       │
│  │ model_performance.py   │  │ analyst_copilot.py     │                       │
│  │ pandas, plotly         │  │ httpx (→ API /chat)    │                       │
│  └────────────────────────┘  └────────────────────────┘                       │
│                                                                               │
│  TESTS (tests/)                                                               │
│  ┌────────────────────────┐  ┌────────────────────────┐                       │
│  │ test_api.py            │  │ test_preprocessing.py  │                       │
│  │ api/main.py            │  │ preprocessing.py       │                       │
│  │ fastapi.testclient      │  │ pytest, pandas         │                       │
│  └────────────────────────┘  └────────────────────────┘                       │
│  ┌────────────────────────┐  ┌────────────────────────┐                       │
│  │ test_anomaly.py        │  │ test_business_cost.py  │                       │
│  │ anomaly.py             │  │ business_cost.py       │                       │
│  └────────────────────────┘  └────────────────────────┘                       │
│  ┌────────────────────────┐  ┌────────────────────────┐                       │
│  │ test_shap_utils.py     │  │ test_rag_similar_cases │                       │
│  │ shap_utils.py          │  │ rag_similar_cases.py   │                       │
│  └────────────────────────┘  └────────────────────────┘                       │
│                                                                               │
│  NOTEBOOKS (notebooks/)                                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐            │
│  │ 01_eda.ipynb     │  │ 02_preprocessing │  │ 03_modeling.ipynb│            │
│  │ data/loaders     │  │ .ipynb           │  │ models/train     │            │
│  └──────────────────┘  │ data/loaders     │  │ evaluation/*     │            │
│                        │ data/preprocessing│ └──────────────────┘            │
│                        └──────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Import Graph (Internal)

| Source File | Internal Imports | External Imports |
|------------|-----------------|-----------------|
| `src/fraudshield/config.py` | — | pathlib |
| `src/fraudshield/data/loaders.py` | `config` | pandas, numpy, joblib, logging |
| `src/fraudshield/data/preprocessing.py` | `config` | sklearn, imblearn, joblib, logging |
| `src/fraudshield/features/engineering.py` | — | pandas, numpy, logging |
| `src/fraudshield/models/train.py` | `config` | sklearn, xgboost, lightgbm, catboost, joblib, logging |
| `src/fraudshield/models/anomaly.py` | `config` | sklearn, numpy, pandas, logging, (optional) tensorflow |
| `src/fraudshield/models/model_selection.py` | — | pandas, numpy, logging, joblib |
| `src/fraudshield/evaluation/metrics.py` | `config` | sklearn.metrics, matplotlib, seaborn, numpy, pandas |
| `src/fraudshield/evaluation/business_cost.py` | — | numpy, logging |
| `src/fraudshield/explainability/shap_utils.py` | `config` | shap, joblib, pandas, numpy, logging |
| `src/fraudshield/llm/case_narrator.py` | — | (optional) anthropic SDK |
| `src/fraudshield/llm/rag_similar_cases.py` | — | faiss, numpy, logging |
| `src/fraudshield/monitoring/drift.py` | `config` | scipy.stats, pandas, numpy, json, logging |
| `src/fraudshield/analysis/eda.py` | `config`, `data/loaders` | pandas, numpy, matplotlib, seaborn, plotly |
| `api/main.py` | `state`, `routers/*`, `config`, `shap_utils`, `llm/*`, `anomaly` | fastapi, pydantic, joblib, logging |
| `api/routers/predict.py` | `schemas`, `state`, `config` | fastapi, pandas, logging |
| `api/routers/explain.py` | `schemas`, `state` | fastapi, logging |
| `api/routers/similar_cases.py` | `schemas`, `state`, `config` | fastapi, logging |
| `api/routers/chat.py` | `state` | fastapi, pydantic, logging |
| `api/schemas.py` | — | pydantic, typing |
| `api/state.py` | — | logging, typing |
| `app/streamlit_app.py` | — | streamlit |
| `app/pages/live_monitor.py` | `config` | streamlit, httpx, pandas, plotly |
| `app/pages/case_investigator.py` | `config` | streamlit, httpx, pandas, plotly |
| `app/pages/model_performance.py` | `config` | streamlit, pandas, plotly |
| `app/pages/analyst_copilot.py` | `config` | streamlit, httpx |
| `app/components/metric_cards.py` | — | streamlit |

## Critical Files (High Impact)

| File | Impact | Why |
|------|--------|-----|
| `src/fraudshield/data/preprocessing.py` | 🔴 Critical | Data leakage prevention — any change affects all downstream |
| `src/fraudshield/explainability/shap_utils.py` | 🔴 Critical | `FraudPredictor` — SHAP integration used by API, dashboard, and notebooks |
| `src/fraudshield/evaluation/business_cost.py` | 🟡 High | Business cost logic — affects threshold optimization and ROI metrics |
| `src/fraudshield/models/train.py` | 🟡 High | Model training — affects all model artifacts |
| `src/fraudshield/models/anomaly.py` | 🟡 High | `IsolationForestDetector` — anomaly scores used in API predictions |
| `api/main.py` | 🟡 High | App lifecycle, model loading, router registration |
| `api/schemas.py` | 🟡 High | Pydantic models define the external API contract |
| `src/fraudshield/config.py` | 🟡 Medium | Central config — many modules depend on it |
| `app/pages/*.py` | 🟢 Medium | UI pages — changes don't affect core ML pipeline |
| `src/fraudshield/data/loaders.py` | 🟢 Medium | Entry point — used only by notebooks and training pipelines |
| `src/fraudshield/features/engineering.py` | 🟢 Low | Not currently used in the main prediction pipeline |
| `src/fraudshield/monitoring/drift.py` | 🟢 Low | Standalone — no downstream dependencies |
| `src/fraudshield/llm/*.py` | 🟢 Low | Optional LLM features (require API keys / external services) |

## External Dependency Map

```
Project
├── Data Science
│   ├── pandas (DataFrame handling)
│   ├── numpy (Numerical operations)
│   └── scipy (Statistical tests for drift)
│
├── Machine Learning
│   ├── scikit-learn (Splitting, scaling, metrics, RF, LR, GB, IForest)
│   ├── xgboost (Primary supervised model)
│   ├── lightgbm (Alternative gradient boosting)
│   ├── catboost (Alternative gradient boosting)
│   └── imbalanced-learn (SMOTE, ADASYN, undersampling)
│
├── Deep Learning (optional)
│   └── tensorflow/keras (AutoencoderDetector)
│
├── Explainability
│   └── shap (TreeExplainer, KernelExplainer)
│
├── LLM / RAG (optional)
│   ├── anthropic (Analyst Copilot / Case Narrator)
│   └── faiss (Similar-case vector search)
│
├── Visualization
│   ├── matplotlib (Static plots in reports)
│   ├── seaborn (Statistical plots)
│   └── plotly (Interactive dashboard charts)
│
├── API
│   ├── fastapi (REST API framework)
│   ├── pydantic (Request/response validation)
│   └── uvicorn (ASGI server)
│
├── Dashboard
│   └── streamlit (Interactive web app)
│
├── Serialization
│   └── joblib (Model persistence)
│
├── Testing
│   └── pytest (Test framework)
│
├── Experiment Tracking (optional)
│   └── mlflow (Training metrics tracking)
│
└── Dev Tools
    ├── black (Formatter)
    ├── isort (Import sorter)
    ├── ruff (Linter)
    └── pre-commit (Git hooks)
```

## File Modification Risk Matrix

| Risk Level | Files | Notes |
|-----------|-------|-------|
| 🔴 Do Not Modify Lightly | `src/fraudshield/data/preprocessing.py`, `src/fraudshield/explainability/shap_utils.py` | Core ML logic, data leakage risk, SHAP integration |
| 🟡 Modify Carefully | `src/fraudshield/evaluation/*`, `src/fraudshield/models/*`, `api/main.py`, `api/schemas.py` | Business logic, training pipeline, API contract |
| 🟢 Safe to Modify | `app/pages/*`, `src/fraudshield/monitoring/*`, `src/fraudshield/llm/*`, `tests/*` | UI, optional features, tests |
| ⚪ Never Modify | `models/*.pkl`, `models/*.joblib`, `data/raw/*`, `data/processed/*` | Generated artifacts, source data (reproducible) |
