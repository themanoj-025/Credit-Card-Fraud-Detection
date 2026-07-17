# 📡 API Map — Credit Card Fraud Detection

## API Overview

- **Framework:** FastAPI v0.95+
- **Base URL:** `http://localhost:8000`
- **Documentation:** `http://localhost:8000/docs` (Swagger UI)
- **OpenAPI Schema:** `http://localhost:8000/openapi.json`
- **CORS:** Enabled (all origins)

## API Inventory

| Method | Route | Purpose | Input Model | Response Model | Used By |
|--------|-------|---------|-------------|----------------|---------|
| GET | `/health` | Health check | — | Dict | Docker healthcheck, dashboard |
| GET | `/model-info` | Model metadata | — | Dict | Dashboard sidebar |
| POST | `/predict` | Single fraud prediction | `TransactionInput` | `PredictionResponse` | Dashboard, external clients |
| POST | `/predict/batch` | Batch predictions | `BatchInput` | BatchResponse | Bulk processing |

## Pydantic Models

### TransactionInput (Request)
```
Fields:
  Time: float          # Transaction time in seconds
  Amount: float        # Transaction amount (≥ 0)
  V1-V28: float        # PCA features (default 0.0)
```

### PredictionResponse (Response)
```
Fields:
  fraud_probability: float     # 0.0 to 1.0
  decision: str                # "FRAUD" or "LEGITIMATE"
  threshold_used: float        # Classification threshold
  is_fraud: bool               # Boolean flag
  explanation: Optional[Dict]  # SHAP breakdown
  business_impact: Optional[Dict]  # Cost analysis
```

### BatchInput (Request)
```
Fields:
  transactions: List[TransactionInput]
```

## Request/Response Flow

### Single Prediction Flow

```
Client                    FastAPI                    FraudPredictor          XGBoost
  │                          │                            │                    │
  │  POST /predict           │                            │                    │
  │  {TransactionInput}      │                            │                    │
  │ ─────────────────────►   │                            │                    │
  │                          │  predict_single()          │                    │
  │                          │ ──────────────────────►    │                    │
  │                          │                            │  predict_proba()   │
  │                          │                            │ ───────────────►   │
  │                          │                            │ ◄───────────────   │
  │                          │                            │  shap_values()    │
  │                          │                            │ ───────────────►   │
  │                          │                            │ ◄───────────────   │
  │                          │ ◄──────────────────────    │                    │
  │  PredictionResponse      │                            │                    │
  │ ◄─────────────────────   │                            │                    │
```

### Error Responses

| Status Code | Meaning | Example |
|-------------|---------|---------|
| 200 | Success | Prediction returned |
| 422 | Validation Error | Negative Amount, missing fields |
| 500 | Server Error | Model prediction failed |
| 503 | Service Unavailable | Model not loaded |

## Rate Limiting

**None currently implemented.** Future consideration for production deployment.

## Authentication

**None currently implemented.** API is open for local development.

## Batch vs Single Performance

| Metric | Single | Batch |
|--------|--------|-------|
| SHAP Explanation | ✅ Yes | ❌ No (skipped for speed) |
| Per-request overhead | Higher | Amortized |
| Use case | Real-time | Bulk processing |
