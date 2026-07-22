# 📡 API Map — FraudLens

## API Overview

- **Framework:** FastAPI v0.95+
- **Base URL:** `http://localhost:8000`
- **Documentation:** `http://localhost:8000/docs` (Swagger UI)
- **OpenAPI Schema:** `http://localhost:8000/openapi.json`
- **Auth:** `X-API-Key` header (optional — set `FRAUDLENS_API_KEYS` env var)
- **Rate Limiting:** slowapi (Redis-backed in production, in-memory for dev)
- **Error Format:** RFC 7807 Problem Details

## API Inventory (v1)

| Method | Route | Purpose | Auth | Rate Limit |
|--------|-------|---------|------|------------|
| `GET` | `/health` | Legacy health check | No | — |
| `GET` | `/v1/health` | Per-dependency health check | No | — |
| `GET` | `/model-info` | Model metadata | No | — |
| `POST` | `/v1/predict` | Single fraud prediction | API Key | 100/min |
| `POST` | `/v1/predict/batch` | Batch predictions (no SHAP) | API Key | 30/min |
| `POST` | `/v1/explain` | SHAP values + LLM narrative | API Key | 60/min |
| `POST` | `/v1/similar-cases` | RAG similar-case retrieval (cursor pagination) | API Key | 60/min |
| `POST` | `/v1/chat` | Analyst copilot chat (requires Anthropic key) | API Key | 20/min |
| `GET` | `/v1/auth/keys` | List configured API keys (admin only) | Admin Key | 30/min |
| `POST` | `/v1/auth/keys` | Generate new API key (admin only) | Admin Key | 10/hour |
| `GET` | `/v1/admin/llm-usage` | LLM cost & usage summary (admin only) — merges DB + in-memory | Admin Key | 30/min |
| `GET` | `/v1/admin/models/candidates` | List model candidates with optional status filter | Admin Key | 30/min |
| `GET` | `/v1/admin/models/candidates/{version}` | Get candidate details | Admin Key | 30/min |
| `POST` | `/v1/admin/models/candidates/{version}/promote` | Promote candidate to production | Admin Key | 10/hour |
| `POST` | `/v1/admin/models/candidates/{version}/reject` | Reject candidate | Admin Key | 10/hour |
| `GET` | `/v1/admin/models/candidates/{version}/compare` | Compare candidate vs current production | Admin Key | 30/min |

## Pydantic Models

### TransactionInput (Request)

```python
Fields:
  Time: float          # Transaction time in seconds (0–172,800)
  Amount: float        # Transaction amount (≥ 0, finite)
  V1..V28: float       # PCA components (default 0.0)

Validation:
  - Amount ≥ 0 (Field constraint)
  - Amount must be finite (NaN/Inf → 422)
```

### PredictionResponse (Response)

```python
Fields:
  fraud_probability: float        # 0.0 to 1.0
  decision: str                   # "FRAUD" or "LEGITIMATE"
  threshold_used: float           # Classification threshold
  is_fraud: bool                  # Boolean flag
  anomaly_score: Optional[float]  # Isolation Forest score (0–1)
  explanation: Optional[Dict]     # SHAP breakdown (only if ?explain=true)
  business_impact: Optional[Dict] # Cost analysis
```

### BatchInput (Request)

```python
Fields:
  transactions: List[TransactionInput]  # 1–1000 transactions
```

### BatchResponse (Response)

```python
Fields:
  predictions: List[BatchPredictionItem]
  summary: BatchSummary  # total, flagged_fraud, flagged_legitimate, estimated_review_cost
```

### SimilarCasesResponse (Response)

```python
Fields:
  transaction_id: str                 # Hashed transaction ID
  similar_cases: List[SimilarCase]    # Top-k similar cases
  pagination: CursorPagination        # next_cursor, has_more, limit, total
```

### CursorPagination (Nested)

```python
Fields:
  next_cursor: Optional[str]  # Cursor for next page (null if no more)
  has_more: bool              # Whether more results exist
  limit: int                  # Max results per page
  total: Optional[int]        # Total results (if known)
```

### Query Parameters for /v1/similar-cases

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `top_k` | int | `RAG_TOP_K` | 1–20 | Number of similar cases |
| `cursor` | str | null | — | Pagination cursor from previous response |
| `limit` | int | `RAG_TOP_K` | 1–50 | Max results per page |

## Error Responses (RFC 7807)

All error responses use the RFC 7807 Problem Details format:

```json
{
  "type": "https://httpstatuses.io/422",
  "title": "Validation Error",
  "status": 422,
  "detail": "Input validation failed",
  "errors": [
    {
      "field": "Amount",
      "message": "Amount must be non-negative",
      "type": "value_error"
    }
  ]
}
```

| Status Code | RFC 7807 `title` | Common Causes |
|-------------|------------------|---------------|
| **422** | Validation Error | Negative Amount, NaN/Inf, missing fields, empty batch, missing features |
| **401** | Unauthorized | Missing or invalid `X-API-Key` header |
| **403** | Forbidden | Valid key but insufficient permissions |
| **429** | Too Many Requests | Rate limit exceeded |
| **500** | Internal Server Error | Model prediction failed, RAG retrieval failed |
| **503** | Service Unavailable | Model not loaded, Copilot API key missing, RAG index absent |

## Health Check

`GET /health` and `GET /v1/health` return per-dependency status:

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

Each dependency reports its own `status`: `ok`, `degraded`, or `error`.
The overall `status` is `healthy` only if all dependencies are `ok`.

## Rate Limiting

| Endpoint | Limit | Reason |
|----------|-------|--------|
| `/v1/predict` | 100/minute | Production prediction |
| `/v1/predict/batch` | 30/minute | Batch processing |
| `/v1/explain` | 60/minute | SHAP + LLM (compute heavy) |
| `/v1/chat` | 20/minute | LLM cost protection |
| `/v1/similar-cases` | 60/minute | RAG retrieval |
| `/v1/auth/keys` POST | 10/hour | Admin security |
| `/v1/auth/keys` GET | 30/minute | Admin security |

## Auth Flow

```
Client                              FastAPI
  │                                    │
  │  POST /v1/predict                  │
  │  X-API-Key: fl_abc123...           │
  │ ───────────────────────────────►   │
  │                                    │ 1. Extract X-API-Key header
  │                                    │ 2. SHA-256 hash the key
  │                                    │ 3. Compare against FRAUDLENS_API_KEYS
  │                                    │ 4. If valid → proceed
  │                                    │ 5. If invalid → 401
  │ ◄───────────────────────────────   │
```

### Admin vs Readonly Keys

| Role | Permissions |
|------|-------------|
| `admin` | Can generate/list API keys, all prediction endpoints |
| `readonly` | All prediction endpoints (cannot manage keys) |

## Batch vs Single Performance

| Metric | Single (/v1/predict) | Batch (/v1/predict/batch) |
|--------|---------------------|---------------------------|
| SHAP Explanation | ✅ Optional (`?explain=true`) | ❌ Skipped for speed |
| Prediction Cache | ✅ LRU cache | ❌ Not cached |
| Vectorized Path | ✅ Numpy (no DataFrame) | ✅ DataFrame (amortized) |
| Anomaly Score | ✅ Isolation Forest | ❌ Not computed |
| Per-request overhead | ~5µs vectorization | Amortized across batch |
| Use Case | Real-time per transaction | Bulk processing |

## Request/Response Flow

```
Client                    FastAPI                    FraudPredictor          XGBoost
  │                          │                            │                    │
  │  POST /v1/predict        │                            │                    │
  │  {TransactionInput}      │                            │                    │
  │ ─────────────────────►   │                            │                    │
  │                          │  Auth check (X-API-Key)    │                    │
  │                          │  Rate limit check          │                    │
  │                          │  Pydantic validation       │                    │
  │                          │                            │                    │
  │                          │  predict_single()          │                    │
  │                          │ ──────────────────────►    │                    │
  │                          │                            │  Check cache       │
  │                          │                            │  vectorize (numpy) │
  │                          │                            │  preprocess (scale)│
  │                          │                            │  predict_proba()   │
  │                          │                            │ ───────────────►   │
  │                          │                            │ ◄───────────────   │
  │                          │                            │                    │
  │                          │  [if ?explain=true]        │                    │
  │                          │  shap_values()             │                    │
  │                          │ ──────────────────────►    │                    │
  │                          │ ◄──────────────────────    │                    │
  │                          │                            │                    │
  │  PredictionResponse      │                            │                    │
  │ ◄─────────────────────   │                            │                    │
```
