# 🗺️ Routes Map — FraudLens

## FastAPI Routes

| Method | Route | Purpose | Auth | Rate Limit | Request | Response |
|--------|-------|---------|------|------------|---------|----------|
| `GET` | `/health` | Health check (legacy) | No | — | — | `{status, version, dependencies}` |
| `GET` | `/v1/health` | Per-dependency health | No | — | — | `{status, version, auth_enabled, dependencies}` |
| `GET` | `/model-info` | Model metadata | No | — | — | `{model_type, threshold, n_features, features}` |
| `POST` | `/v1/predict` | Single prediction | API Key | 100/min | `TransactionInput` | `PredictionResponse` |
| `POST` | `/v1/predict/batch` | Batch prediction | API Key | 30/min | `BatchInput` | `BatchResponse` |
| `POST` | `/v1/explain` | SHAP + LLM narrative | API Key | 60/min | `TransactionInput` | `ExplanationResponse` |
| `POST` | `/v1/similar-cases` | RAG similar cases | API Key | 60/min | `TransactionInput` + query params | `SimilarCasesResponse` |
| `POST` | `/v1/chat` | Analyst copilot | API Key | 20/min | `ChatRequest` | `{response, tool_calls}` |
| `GET` | `/v1/auth/keys` | List API keys | Admin Key | 30/min | — | `{keys[], count}` |
| `POST` | `/v1/auth/keys` | Generate API key | Admin Key | 10/hour | `{role, description}` | `{api_key, role, sha256_hash}` |
| `GET` | `/v1/admin/llm-usage` | LLM cost & usage (DB+memory merge) | Admin Key | 30/min | `?period=today\|month\|total` | `{date, total_cost_usd, total_calls, by_model, by_endpoint}` |
| `GET` | `/v1/admin/models/candidates` | List model candidates | Admin Key | 30/min | `?status_filter=candidate\|promoted\|rejected&limit=50` | `{candidates[], total, pending, promoted, rejected}` |
| `GET` | `/v1/admin/models/candidates/{version}` | Get candidate details | Admin Key | 30/min | — | `{model_version, trigger, pr_auc, f1_score, ...}` |
| `POST` | `/v1/admin/models/candidates/{version}/promote` | Promote candidate | Admin Key | 10/hour | — | `{success, model_version, message, candidate}` |
| `POST` | `/v1/admin/models/candidates/{version}/reject` | Reject candidate | Admin Key | 10/hour | — | `{success, model_version, message, candidate}` |
| `GET` | `/v1/admin/models/candidates/{version}/compare` | Compare vs production | Admin Key | 30/min | — | `{current_production, candidate, metrics_delta}` |

## Route Details

### `GET /health` and `GET /v1/health`

**Purpose:** Check service health with per-dependency breakdown.

```
GET /v1/health
```

**Response (200):**
```json
{
  "status": "degraded",
  "version": "2.0.0",
  "auth_enabled": false,
  "dependencies": {
    "model": {"status": "degraded", "detail": "not loaded"},
    "database": {"status": "ok", "detail": "connected"},
    "anomaly_detector": {"status": "degraded", "detail": "not loaded"},
    "llm": {"status": "degraded", "detail": "API key not set"},
    "case_narrator": {"status": "degraded", "detail": "not loaded"},
    "rag_retriever": {"status": "degraded", "detail": "no index found"}
  }
}
```

The health endpoint never fails — it always returns 200 with a `degraded` status when dependencies are missing, making it safe for orchestration probes.

### `GET /model-info`

**Purpose:** Get model type, threshold, and feature configuration.

```json
{
  "model_type": "XGBClassifier",
  "threshold": 0.0298,
  "n_features": 30,
  "features": ["V1", "V2", ..., "V28", "Time", "Amount"],
  "avg_fraud_loss": 150.0,
  "review_cost": 5.0
}
```

### `POST /v1/predict`

**Purpose:** Predict fraud probability for a single transaction.  
**Auth:** `X-API-Key` header required if `FRAUDLENS_API_KEYS` is set.  
**Query params:** `?explain=true` enables SHAP explanation (default: off for performance).

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: fl_your_key_here" \
  -d '{
    "Time": 100000.0, "Amount": 150.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": 0.02
  }'
```

**Response (200):**
```json
{
  "fraud_probability": 0.9234,
  "decision": "FRAUD",
  "threshold_used": 0.0298,
  "is_fraud": true,
  "anomaly_score": 0.84,
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

### `POST /v1/predict/batch`

**Purpose:** Predict fraud for multiple transactions (faster, no SHAP).  
**Auth:** `X-API-Key` header required if `FRAUDLENS_API_KEYS` is set.  
**Limits:** 1–1000 transactions per batch.

**Response (200):**
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

### `POST /v1/explain`

**Purpose:** Get SHAP values and optional LLM narrative explaining why a transaction was flagged.  
**Auth:** `X-API-Key` header required if `FRAUDLENS_API_KEYS` is set.

**Response (200):**
```json
{
  "fraud_probability": 0.9234,
  "decision": "FRAUD",
  "shap_values": {"V14": 0.34, "V4": 0.22, "V12": 0.18},
  "narrative": "Transaction flagged as fraud (92.3% probability). Key drivers: V14 (high negative value) strongly increases fraud risk..."
}
```

### `POST /v1/similar-cases`

**Purpose:** Retrieve similar historical flagged transactions using FAISS-based RAG retrieval.  
**Auth:** `X-API-Key` header required if `FRAUDLENS_API_KEYS` is set.  
**Query Params:** `top_k` (1–20), `cursor` (pagination), `limit` (1–50).

**Response (200):**
```json
{
  "transaction_id": "tx_abc123",
  "similar_cases": [
    {
      "similarity_score": 0.94,
      "actual_outcome": "FRAUD",
      "features": {"V1": -1.36, "V4": 2.54, ...}
    }
  ],
  "pagination": {
    "next_cursor": "10",
    "has_more": true,
    "limit": 5,
    "total": 50
  }
}
```

### `POST /v1/chat`

**Purpose:** Analyst copilot chat — ask natural-language questions (requires `ANTHROPIC_API_KEY`).  
**Auth:** `X-API-Key` header required if `FRAUDLENS_API_KEYS` is set.

**Request Body:**
```json
{
  "message": "Why was transaction tx_789 flagged?",
  "conversation_history": [
    {"role": "user", "content": "What's the current fraud rate?"},
    {"role": "assistant", "content": "Current fraud rate is 2.3%..."}
  ]
}
```

**Response (200):**
```json
{
  "response": "Transaction tx_789 was flagged because...",
  "tool_calls": []
}
```

## Error Status Codes

| Code | Title | Scenario |
|------|-------|----------|
| 200 | — | Successful prediction/response |
| 401 | Unauthorized | Missing or invalid API key |
| 422 | Validation Error | Negative Amount, NaN/Inf, empty batch, missing features |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Model prediction failure |
| 409 | Conflict | Candidate already promoted or rejected (idempotent protect) |
| 503 | Service Unavailable | Model not loaded, LLM unavailable, RAG index missing, DB unavailable |
