# 🔗 Dependency Graph — Credit Card Fraud Detection

## Module Dependency Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEPENDENCY GRAPH                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  NOTEBOOKS (Entry Points)                                           │
│  ┌──────────────────┐                                               │
│  │ 01_eda.ipynb     │──── src/data_loader.py                       │
│  └──────────────────┘                                               │
│  ┌──────────────────┐                                               │
│  │ 02_preprocessing │──── src/data_loader.py                       │
│  │    .ipynb        │──── src/preprocessing.py                     │
│  └──────────────────┘                                               │
│  ┌──────────────────┐                                               │
│  │ 03_modeling.ipynb│──── src/train.py                             │
│  │                  │──── src/evaluate.py                          │
│  └──────────────────┘                                               │
│  ┌──────────────────┐                                               │
│  │ 04_explainability│──── src/predict.py                           │
│  │    .ipynb        │──── shap                                     │
│  └──────────────────┘                                               │
│                                                                     │
│  APPLICATION LAYER                                                  │
│  ┌──────────────────┐                                               │
│  │ api/main.py      │──── src/predict.py                           │
│  └──────────────────┘     └──→ joblib (load model)                 │
│                                                                     │
│  ┌──────────────────┐                                               │
│  │ app/dashboard.py │──── src/predict.py                           │
│  │                  │──── src/evaluate.py                          │
│  │                  │──── streamlit                                │
│  │                  │──── plotly                                   │
│  └──────────────────┘                                               │
│                                                                     │
│  CORE MODULES (src/)                                                │
│  ┌──────────────────┐                                               │
│  │ data_loader.py   │──── pandas, numpy                            │
│  └──────────────────┘                                               │
│  ┌──────────────────┐                                               │
│  │ preprocessing.py │──── sklearn.model_selection                  │
│  │                  │──── sklearn.preprocessing                    │
│  │                  │──── imblearn (SMOTE, ADASYN, etc.)          │
│  └──────────────────┘                                               │
│  ┌──────────────────┐                                               │
│  │ features.py      │──── pandas, numpy                            │
│  └──────────────────┘                                               │
│  ┌──────────────────┐                                               │
│  │ train.py         │──── sklearn.linear_model                     │
│  │                  │──── sklearn.ensemble                         │
│  │                  │──── xgboost                                  │
│  │                  │──── lightgbm                                 │
│  └──────────────────┘                                               │
│  ┌──────────────────┐                                               │
│  │ evaluate.py      │──── sklearn.metrics                          │
│  │                  │──── matplotlib, seaborn                      │
│  └──────────────────┘                                               │
│  ┌──────────────────┐                                               │
│  │ predict.py       │──── shap                                     │
│  │                  │──── joblib                                   │
│  └──────────────────┘                                               │
│                                                                     │
│  MONITORING                                                         │
│  ┌──────────────────┐                                               │
│  │ drift_detection  │──── scipy.stats                              │
│  │    .py           │──── pandas, numpy                            │
│  └──────────────────┘                                               │
│                                                                     │
│  TESTS                                                              │
│  ┌──────────────────┐                                               │
│  │ test_preprocessing│──── src/preprocessing.py                    │
│  └──────────────────┘                                               │
│  ┌──────────────────┐                                               │
│  │ test_api.py      │──── api/main.py                              │
│  │                  │──── fastapi.testclient                       │
│  └──────────────────┘                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Import Graph (Internal)

| Source File | Imports From (Internal) | Imports From (External) |
|------------|------------------------|------------------------|
| `src/data_loader.py` | — | pandas, numpy, joblib, logging |
| `src/preprocessing.py` | — | sklearn, imblearn, joblib, logging |
| `src/features.py` | — | pandas, numpy, logging |
| `src/train.py` | — | sklearn, xgboost, lightgbm, joblib, logging |
| `src/evaluate.py` | — | sklearn.metrics, matplotlib, seaborn, logging |
| `src/predict.py` | — | shap, joblib, pandas, numpy, logging |
| `api/main.py` | `src/predict.py` | fastapi, pydantic, joblib, pandas |
| `app/dashboard.py` | `src/predict.py`, `src/evaluate.py` | streamlit, plotly, joblib |
| `monitoring/drift_detection.py` | — | scipy.stats, pandas, numpy, json |
| `tests/test_preprocessing.py` | `src/preprocessing.py` | pytest, pandas, numpy |
| `tests/test_api.py` | `api/main.py` | pytest, numpy, fastapi.testclient |

## Critical Files (High Impact)

| File | Impact | Why |
|------|--------|-----|
| `src/preprocessing.py` | 🔴 Critical | Data leakage prevention — any change affects all downstream |
| `src/predict.py` | 🔴 Critical | SHAP integration — used by both API and dashboard |
| `src/evaluate.py` | 🟡 High | Business cost logic — affects threshold optimization |
| `src/train.py` | 🟡 High | Model training — affects all model artifacts |
| `api/main.py` | 🟡 High | API contract — Pydantic models define external interface |
| `app/dashboard.py` | 🟢 Medium | UI only — changes don't affect core ML pipeline |
| `src/data_loader.py` | 🟢 Medium | Entry point — used only by notebooks |
| `src/features.py` | 🟢 Low | Not currently used in main pipeline |
| `monitoring/drift_detection.py` | 🟢 Low | Standalone monitoring |

## External Dependency Map

```
Project
├── Data Science
│   ├── pandas (DataFrame handling)
│   ├── numpy (Numerical operations)
│   └── scipy (Statistical tests for drift)
│
├── Machine Learning
│   ├── scikit-learn (Splitting, scaling, metrics, RF, LR)
│   ├── xgboost (Primary model)
│   ├── lightgbm (Alternative gradient boosting)
│   └── imbalanced-learn (SMOTE, ADASYN, undersampling)
│
├── Explainability
│   └── shap (TreeExplainer, KernelExplainer)
│
├── Visualization
│   ├── matplotlib (Static plots)
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
└── Testing
    └── pytest (Test framework)
```

## File Modification Risk Matrix

| Risk Level | Files | Notes |
|-----------|-------|-------|
| 🔴 Do Not Modify | `src/preprocessing.py`, `src/predict.py` | Core ML logic, data leakage risk |
| 🟡 Modify Carefully | `src/evaluate.py`, `src/train.py`, `api/main.py` | Business logic, API contract |
| 🟢 Safe to Modify | `app/dashboard.py`, `tests/*.py`, `monitoring/*.py` | UI, tests, monitoring |
| ⚪ Never Modify | `models/*.pkl`, `Dataset/*.csv` | Generated artifacts, source data |
