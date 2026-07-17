# 🗄️ Database Map — Credit Card Fraud Detection

## Overview

This project does **not use a database**. All data storage is file-based:

| Storage Type | Location | Purpose | Format |
|-------------|----------|---------|--------|
| Raw Data | `Dataset/Dataset/creditcard.csv` | Source dataset | CSV |
| Processed Data | `data/processed/*.pkl` | Train/test splits | Pickle |
| Model Artifacts | `models/*.pkl` | Trained models | Pickle |
| Threshold | `models/threshold.txt` | Optimal classification threshold | Plain text |
| Drift Reports | `monitoring/drift_report.json` | Drift detection results | JSON |
| MLflow (optional) | `mlruns/` | Experiment tracking | SQLite + artifacts |

## Data Schema

### creditcard.csv (Source Data)

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| Time | float | Seconds since first transaction | 0 – 172,800 |
| V1-V28 | float | PCA-anonymized features | ~-30 to +30 |
| Amount | float | Transaction amount ($) | 0 – 25,691 |
| Class | int | 0 = legitimate, 1 = fraud | 0 or 1 |

**Shape:** 284,807 rows × 31 columns
**Class Distribution:** 284,315 legitimate (99.828%), 492 fraud (0.172%)

### Processed Data Files

| File | Contents | Created By |
|------|----------|-----------|
| `X_train.pkl` | Training features (scaled) | `02_preprocessing.ipynb` |
| `X_test.pkl` | Test features (scaled) | `02_preprocessing.ipynb` |
| `y_train.pkl` | Training labels | `02_preprocessing.ipynb` |
| `y_test.pkl` | Test labels | `02_preprocessing.ipynb` |
| `X_train_smote.pkl` | SMOTE-resampled training features | `02_preprocessing.ipynb` |
| `y_train_smote.pkl` | SMOTE-resampled training labels | `02_preprocessing.ipynb` |

### Model Artifacts

| File | Contents | Size (approx) |
|------|----------|---------------|
| `xgboost.pkl` | Trained XGBoost model | ~500KB |
| `scaler.pkl` | Fitted StandardScaler | ~2KB |
| `threshold.txt` | Optimal threshold (e.g., 0.35) | ~5 bytes |

## Entity Relationships

```
creditcard.csv
    │
    ├──► FraudPreprocessor.split_data()
    │       │
    │       ├──► X_train.pkl ──► Resampler.resample() ──► X_train_smote.pkl
    │       │                       │
    │       │                       └──► FraudTrainer.train_all()
    │       │                               │
    │       │                               ├──► models/xgboost.pkl
    │       │                               ├──► models/lightgbm.pkl
    │       │                               ├──► models/random_forest.pkl
    │       │                               └──► models/logistic_regression.pkl
    │       │
    │       ├──► X_test.pkl ──► FraudEvaluator.evaluate_model()
    │       │                       │
    │       │                       └──► models/threshold.txt
    │       │
    │       ├──► y_train.pkl
    │       └──► y_test.pkl
    │
    └──► FraudPredictor.predict_single()
            │
            ├──► models/xgboost.pkl (loaded)
            ├──► models/scaler.pkl (loaded)
            └──► models/threshold.txt (loaded)
```

## Data Volume

| Dataset | Rows | Columns | Size |
|---------|------|---------|------|
| Raw CSV | 284,807 | 31 | ~150MB |
| Training set | ~227,845 | 30 | ~50MB |
| Test set | ~56,962 | 30 | ~12MB |
| SMOTE training | ~341,593 | 30 | ~75MB |

## File Lifecycle

1. **Input:** `creditcard.csv` (manual download from Kaggle)
2. **Processing:** Notebooks create `data/processed/*.pkl`
3. **Training:** Notebooks create `models/*.pkl`
4. **Serving:** FastAPI loads `models/*.pkl` at startup
5. **Monitoring:** Drift detector creates `monitoring/drift_report.json`
6. **Cleanup:** All artifacts are gitignored; regenerate from notebooks
