# 🗺️ Routes Map — Credit Card Fraud Detection

## FastAPI Routes

| Method | Route | Purpose | Request Body | Response | Auth |
|--------|-------|---------|-------------|----------|------|
| `GET` | `/health` | Health check | — | `{status, model_loaded, version}` | No |
| `GET` | `/model-info` | Model metadata | — | `{model_type, threshold, n_features, features}` | No |
| `POST` | `/predict` | Single prediction | `TransactionInput` | `PredictionResponse` | No |
| `POST` | `/predict/batch` | Batch prediction | `BatchInput` | `{predictions[], summary}` | No |

### Route Details

#### `GET /health`
- **Purpose:** Check if API and model are loaded
- **Response:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "version": "1.0.0"
  }
  ```

#### `GET /model-info`
- **Purpose:** Get model type and configuration
- **Response:**
  ```json
  {
    "model_type": "XGBClassifier",
    "threshold": 0.35,
    "n_features": 30,
    "features": ["V1", "V2", ..., "V28", "Time", "Amount"]
  }
  ```

#### `POST /predict`
- **Purpose:** Predict fraud probability for a single transaction
- **Request Body (`TransactionInput`):**
  ```json
  {
    "Time": 100000.0,
    "Amount": 150.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": 0.02
  }
  ```
- **Response (`PredictionResponse`):**
  ```json
  {
    "fraud_probability": 0.9234,
    "decision": "FRAUD",
    "threshold_used": 0.35,
    "is_fraud": true,
    "explanation": {
      "summary": "Flagged mainly due to: V14, V4, V12",
      "top_features": [
        {"feature": "V14", "value": -5.23, "shap_value": 0.34, "impact": "increases"},
        {"feature": "V4", "value": 4.12, "shap_value": 0.22, "impact": "increases"},
        {"feature": "V12", "value": -3.89, "shap_value": 0.18, "impact": "increases"}
      ]
    },
    "business_impact": {
      "estimated_loss": 150.0,
      "action": "FLAG for manual review",
      "review_cost": 5.0
    }
  }
  ```

#### `POST /predict/batch`
- **Purpose:** Predict fraud for multiple transactions (faster, no SHAP)
- **Request Body:**
  ```json
  {
    "transactions": [
      {"Time": 100000.0, "Amount": 150.0, "V1": -1.36, ...},
      {"Time": 200000.0, "Amount": 5000.0, "V1": -4.50, ...}
    ]
  }
  ```
- **Response:**
  ```json
  {
    "predictions": [
      {"fraud_probability": 0.12, "decision": "LEGITIMATE", "is_fraud": false},
      {"fraud_probability": 0.87, "decision": "FRAUD", "is_fraud": true}
    ],
    "summary": {
      "total": 2,
      "flagged_fraud": 1,
      "flagged_legitimate": 1,
      "estimated_review_cost": 5.0
    }
  }
  ```

## Streamlit Dashboard Routes

Streamlit uses a single-page app model (no URL routing). The dashboard is accessed at:

| URL | Page | Description |
|-----|------|-------------|
| `http://localhost:8501` | Main Dashboard | Transaction feed, analytics, business impact |

### Dashboard Sections

1. **Sidebar Controls** — Simulation speed, fraud rate, batch size, reset button
2. **Top Metrics Row** — Total transactions, actual fraud, flagged, saved, review costs
3. **Transaction Feed** — Expandable cards for each transaction (last 20)
4. **Analytics Charts** — Probability distribution, cumulative business impact
5. **Business Impact Summary** — Fraud caught, missed, reviews, net benefit
