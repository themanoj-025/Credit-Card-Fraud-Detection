# 🗄️ Database Map — FraudLens

## Overview

FraudLens uses **PostgreSQL** as the system of record for:

| Schema | Purpose |
|--------|---------|
| `predictions` | All prediction results (single + batch) |
| `feedback` | Analyst-confirmed labels (fraud/legitimate) |
| `api_keys` | Stored as SHA-256 hashes |
| `drift_events` | Data drift monitoring events |

SQLite is used as a dev-only fallback. PostgreSQL is the production target.

## Connection

| Environment | Database URL |
|-------------|-------------|
| **Development (Docker)** | `postgresql+asyncpg://fraudlens:fraudlens_dev@localhost:5432/fraudlens` |
| **Local (SQLite)** | `sqlite+aiosqlite:///./fraudlens.db` |
| **Test** | `sqlite+aiosqlite:///./test_fraudlens.db` |

## Schema

### Table: `predictions`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PK, default gen_random_uuid() | Unique prediction ID |
| `transaction_snapshot` | JSONB | NOT NULL | Full input transaction features |
| `fraud_probability` | FLOAT | NOT NULL | Model output (0.0–1.0) |
| `decision` | VARCHAR(16) | NOT NULL | `FRAUD` or `LEGITIMATE` |
| `threshold_used` | FLOAT | NOT NULL | Decision threshold at time of prediction |
| `model_version` | VARCHAR(64) | NOT NULL | Model version identifier |
| `is_fraud` | BOOLEAN | NOT NULL | Whether decision was fraud |
| `anomaly_score` | FLOAT | NULLABLE | Isolation Forest score (0–1) |
| `latency_ms` | FLOAT | NOT NULL | Prediction latency in milliseconds |
| `explanation` | JSONB | NULLABLE | SHAP explanation (if computed) |
| `created_at` | TIMESTAMPTZ | NOT NULL, default NOW() | Timestamp of prediction |

**Indexes:**
- `idx_predictions_created_at` ON `created_at`
- `idx_predictions_decision` ON `decision`
- `idx_predictions_is_fraud` ON `is_fraud`

### Table: `feedback`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PK, default gen_random_uuid() | Unique feedback ID |
| `prediction_id` | UUID | FK → predictions(id), NOT NULL | Which prediction this feedback is for |
| `confirmed_fraud` | BOOLEAN | NOT NULL | Analyst's confirmed label |
| `analyst_notes` | TEXT | NULLABLE | Free-text notes from analyst |
| `reviewed_by` | VARCHAR(128) | NULLABLE | Analyst identifier |
| `created_at` | TIMESTAMPTZ | NOT NULL, default NOW() | When feedback was submitted |

**Indexes:**
- `idx_feedback_prediction_id` ON `prediction_id`
- `idx_feedback_confirmed_fraud` ON `confirmed_fraud`

### Table: `api_keys`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PK, default gen_random_uuid() | Unique key ID |
| `key_hash` | VARCHAR(64) | UNIQUE, NOT NULL | SHA-256 hash of the API key |
| `role` | VARCHAR(16) | NOT NULL | `admin` or `readonly` |
| `description` | VARCHAR(255) | NULLABLE | Human-readable description |
| `is_active` | BOOLEAN | NOT NULL, default TRUE | Whether key is active |
| `created_at` | TIMESTAMPTZ | NOT NULL, default NOW() | When key was created |
| `last_used_at` | TIMESTAMPTZ | NULLABLE | Last authentication timestamp |

**Indexes:**
- `idx_api_keys_key_hash` ON `key_hash` (UNIQUE)

### Table: `drift_events`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | UUID | PK, default gen_random_uuid() | Unique event ID |
| `feature_name` | VARCHAR(64) | NOT NULL | Feature that drifted |
| `ks_statistic` | FLOAT | NOT NULL | KS-test statistic |
| `p_value` | FLOAT | NOT NULL | KS-test p-value |
| `alert_level` | VARCHAR(16) | NOT NULL | `OK`, `WARNING`, or `CRITICAL` |
| `reference_mean` | FLOAT | NOT NULL | Reference distribution mean |
| `current_mean` | FLOAT | NOT NULL | Current distribution mean |
| `created_at` | TIMESTAMPTZ | NOT NULL, default NOW() | When drift was detected |

## Entity Relationships

```mermaid
erDiagram
    predictions ||--o{ feedback : has
    predictions {
        uuid id PK
        jsonb transaction_snapshot
        float fraud_probability
        varchar decision
        float threshold_used
        varchar model_version
        boolean is_fraud
        float anomaly_score
        float latency_ms
        jsonb explanation
        timestamptz created_at
    }
    feedback {
        uuid id PK
        uuid prediction_id FK
        boolean confirmed_fraud
        text analyst_notes
        varchar reviewed_by
        timestamptz created_at
    }
    api_keys {
        uuid id PK
        varchar key_hash UK
        varchar role
        varchar description
        boolean is_active
        timestamptz created_at
        timestamptz last_used_at
    }
    drift_events {
        uuid id PK
        varchar feature_name
        float ks_statistic
        float p_value
        varchar alert_level
        float reference_mean
        float current_mean
        timestamptz created_at
    }
```

## Repository Pattern

All database access goes through repository classes in `src/fraudlens/persistence/repositories/`:

| Repository | Methods |
|------------|---------|
| `PredictionRepository` | `create()`, `get_by_id()`, `list_recent()`, `get_statistics()` |
| `FeedbackRepository` | `create()`, `get_by_prediction_id()`, `get_statistics()` |
| `ApiKeyRepository` | `create()`, `get_by_hash()`, `list_all()`, `deactivate()` |
| `DriftEventRepository` | `create()`, `list_recent()` |

Routers never touch SQLAlchemy sessions directly — they go through repositories.

## Migrations (Alembic)

```bash
# Create a new migration
alembic revision --autogenerate -m "description"

# Apply all pending migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1

# View history
alembic history
```

## RAG Data

Similar cases are stored in a **FAISS index** built from historical transactions. The source-of-truth data lives in Postgres; the FAISS index is rebuilt on startup or on a schedule.

| Component | Storage | Purpose |
|-----------|---------|---------|
| Source cases | PostgreSQL | System of record for historical transactions |
| FAISS index | Filesystem (`models/rag_index/`) | Fast cosine-similarity nearest-neighbor search |
| Embedding | PCA-reduced feature vectors | Dimensionality reduction for similarity search |

## Data Volume Estimates

| Table | Rows (est.) | Size (est.) | Growth Rate |
|-------|-------------|-------------|-------------|
| `predictions` | 1,000/day | ~100KB/day | Linear |
| `feedback` | 10/day | ~2KB/day | Linear (rare) |
| `api_keys` | < 100 | ~10KB | Static |
| `drift_events` | 30/batch | ~5KB/batch | Per monitoring run |

## File-Based Storage (Legacy)

The following file-based storage remains for training/evaluation (not serving):

| Storage | Location | Format |
|---------|----------|--------|
| Raw Dataset | `Dataset/Dataset/creditcard.csv` | CSV |
| Processed Data | `data/processed/*.pkl` | Pickle |
| Model Artifacts | `models/*.pkl` | Pickle |
| Threshold | `models/threshold.txt` | Plain text |
| MLflow Artifacts | `mlruns/` | Filesystem |
