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
| POST | `/explain` | SHAP + LLM narrative | `TransactionInput` | `ExplanationResponse` | Dashboard detail panel |
| POST | `/similar-cases` | RAG similar-case retrieval | Transaction dict | `SimilarCasesResponse` | Case investigator page |
| POST | `/chat` | Analyst copilot chat | `ChatRequest` | ChatResponse | Analyst copilot page |

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
  anomaly_score: Optional[float]  # Isolation Forest score
  explanation: Optional[Dict]  # SHAP breakdown
  business_impact: Optional[Dict]  # Cost analysis
```

### BatchInput (Request)
```
Fields:
  transactions: List[TransactionInput]  # 1–1000 transactions
```

### ExplanationResponse (Response)
```
Fields:
  fraud_probability: float        # 0.0 to 1.0
  decision: str                   # "FRAUD" or "LEGITIMATE"
  shap_values: Dict[str, float]   # Feature → SHAP contribution
  narrative: Optional[str]        # LLM-generated plain-English explanation
```

### SimilarCasesResponse (Response)
```
Fields:
  transaction_id: str                 # Hashed transaction ID
  similar_cases: List[SimilarCase]    # Top-k similar cases
```

### SimilarCase (Nested)
```
Fields:
  similarity_score: float         # 0.0 to 1.0 (cosine similarity)
  actual_outcome: str             # "FRAUD" or "LEGITIMATE"
  features: Dict[str, float]      # Transaction features
```

### ChatRequest (Request)
```
Fields:
  message: str                           # Natural-language question
  conversation_history: Optional[List]    # Previous messages [{"role", "content"}]
```

### ChatResponse (Response)
```
Fields:
  response: str                     # AI-generated answer
  tool_calls: Optional[List[Dict]]  # Tool invocations (future use)
```

### Error Responses

| Status Code | Meaning | Common Causes |
|-------------|---------|---------------|
| 200 | Success | |
| 422 | Validation Error | Negative Amount, missing fields, empty batch |
| 500 | Server Error | Model prediction failed, RAG retrieval failed |
| 503 | Service Unavailable | Model not loaded, Copilot API key missing, RAG index absent |

## Error Flow

```
Client                    FastAPI
  │                          │
  │  POST /explain (invalid) │
  │  {"Amount": -100}        │
  │ ─────────────────────►   │
  │                          │  Pydantic: Amount ≥ 0?
  │                          │  ── 422 Validation Error
  │ ◄─────────────────────   │
  │                          │
  │  POST /explain (valid)   │
  │ ─────────────────────►   │
  │                          │  Model loaded?
  │                          │  ── 503 Service Unavailable
  │ ◄─────────────────────   │
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
