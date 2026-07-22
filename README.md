# FraudLens — Credit Card Fraud Detection

[![CI](https://github.com/your-org/fraudlens/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/fraudlens/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-78%25-yellowgreen)](https://github.com/your-org/fraudlens)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Production-grade credit card fraud detection** with SHAP explainability, LLM-powered case narratives, and RAG-based similar case retrieval.

---

## 🚀 Quickstart

### Option A: Docker (Recommended)

```bash
# Clone and start everything — zero setup required
git clone https://github.com/your-org/fraudlens.git
cd fraudlens
docker compose up
```

**First run automatically generates a synthetic dataset** (5,000 transactions matching the real schema) so the demo works immediately. If you have Kaggle credentials, the real dataset downloads automatically.

### Option B: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download or generate dataset (Kaggle API or synthetic fallback)
make setup-data

# Train the model
make train

# Start the API
make api  # FastAPI on :8000

# Start the dashboard (separate terminal)
make dashboard  # Streamlit on :8501
```

### Option C: Real Dataset from Kaggle

If you want the real 284K-transaction dataset:

```bash
# Set Kaggle credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Download real data
make setup-data

# Or let it auto-download on first docker compose up
docker compose up
```

**Get Kaggle credentials:** [Kaggle API Token Setup](https://github.com/Kaggle/kaggle-api#api-credentials)

---

## 📊 Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Training** | XGBoost, LightGBM, Random Forest, Logistic Regression, Isolation Forest |
| **SHAP Explainability** | Every prediction comes with feature importance explanations |
| **LLM Case Narration** | Plain-English analyst summaries via Anthropic Claude |
| **LLM Cost Tracking** | Per-call cost logging, Prometheus counters, persisted to DB |
| **RAG Similar Cases** | FAISS-based retrieval of historical fraud precedents |
| **Real-time Dashboard** | Streamlit UI with 5 pages: Live Monitor, Case Investigator, Model Performance, Model Governance, Analyst Copilot |
| **Model Governance** | Review, compare, and promote/reject retrained model candidates from the UI |
| **Automated Retraining** | Drift + feedback volume triggers, K8s CronJob, MLflow candidate tracking, human-gated promotion |
| **Production API** | FastAPI with auth, rate limiting, CORS, RFC 7807 errors |
| **Observability** | Structured logging, Prometheus metrics, Jaeger tracing, Grafana dashboards |
| **Kubernetes Ready** | Docker multi-stage build (~400MB), K8s manifests (deployment, HPA, CronJob, Ingress) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Dashboard                      │
│                    (Live Monitor, Investigator, ...)             │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP
┌─────────────────────────▼───────────────────────────────────────┐
│                      FastAPI Server (:8000)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ /predict │  │ /explain │  │  /chat   │  │ /similar │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │              │              │              │              │
│  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐        │
│  │ XGBoost  │  │   SHAP   │  │  Claude  │  │   FAISS  │        │
│  │ Model    │  │ Explainer│  │   LLM    │  │   RAG    │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     PostgreSQL + Redis                           │
│              (predictions, feedback, rate limiting)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage (target 85%)
make test-cov

# Run integration tests
make test-integration

# Run OpenAPI contract tests
make test-contract
```

---

## 📚 Documentation

- [Architecture Decision Records](docs/adr/)
- [Model Card](docs/MODEL_CARD.md)
- [Resilience & Error Handling](docs/RESILIENCE.md)
- [CHANGELOG](CHANGELOG.md)
- [Contributing Guidelines](CONTRIBUTING.md)

---

## 🔧 Configuration

All settings are configurable via environment variables. See [`.env.example`](.env.example) for the full list.

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM features | — |
| `FRAUDLENS_API_KEYS` | API key auth (semicolon-delimited) | — |
| `FRAUDLENS_DASHBOARD_API_KEY` | Admin API key for dashboard Model Governance page | — |
| `DATABASE_URL` | PostgreSQL connection string | SQLite fallback |
| `REDIS_URL` | Redis connection string | In-memory fallback |
| `KAGGLE_USERNAME` | Kaggle username for data download | — |
| `KAGGLE_KEY` | Kaggle API key | — |
| `RETRAINING_FEEDBACK_THRESHOLD` | Min feedback labels to trigger retraining | `100` |
| `RETRAINING_DRIFT_CRITICAL_THRESHOLD` | Min CRITICAL drift events to trigger | `3` |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
