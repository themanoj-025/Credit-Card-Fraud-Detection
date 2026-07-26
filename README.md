# FraudLens — Credit Card Fraud Detection

[![CI](https://github.com/your-org/fraudlens/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/fraudlens/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-78%25-yellowgreen)](https://github.com/your-org/fraudlens)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-grade credit card fraud detection with SHAP explainability, LLM-powered case narratives, and RAG-based similar case retrieval.

---

## Quick Start

### Option A: Docker (Recommended)

```bash
git clone https://github.com/your-org/fraudlens.git
cd fraudlens
docker compose up
```

**First run automatically generates a synthetic dataset** (5,000 transactions matching the real schema) so the demo works immediately. If you have Kaggle credentials, the real dataset downloads automatically.

### Option B: Local Development

```bash
pip install -r requirements.txt
make setup-data     # Download or generate dataset
make train          # Train the model
make api            # FastAPI on :8000
make dashboard      # Streamlit on :8501
```

---

## Features

| Feature | Description |
|---------|-------------|
| Multi-Model Training | XGBoost, LightGBM, Random Forest, Logistic Regression, Isolation Forest |
| SHAP Explainability | Feature importance explanations per prediction |
| LLM Case Narration | Plain-English analyst summaries via Anthropic Claude |
| RAG Similar Cases | FAISS-based retrieval of historical fraud precedents |
| Real-time Dashboard | Streamlit UI with 5 pages: Live Monitor, Case Investigator, Model Performance, Model Governance, Analyst Copilot |
| Model Governance | Review, compare, and promote/reject retrained model candidates |
| Automated Retraining | Drift + feedback volume triggers, MLflow candidate tracking, human-gated promotion |
| Production API | FastAPI with auth, rate limiting, CORS |
| Observability | Structured logging, Prometheus metrics, Jaeger tracing, Grafana dashboards |
| Kubernetes Ready | Docker multi-stage build, K8s manifests |

---

## Architecture

```
Streamlit Dashboard ← HTTP → FastAPI Server
                              ├── /predict   → XGBoost Model
                              ├── /explain   → SHAP Explainer
                              ├── /chat      → Claude LLM
                              └── /similar   → FAISS RAG
                                   ↓
                         PostgreSQL + Redis
```

---

## Testing

```bash
make test             # Run all tests
make test-cov         # Run with coverage
make test-integration # Integration tests
```

---

## Configuration

All settings via environment variables. See [`.env.example`](.env.example) for full list.

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key for LLM features | — |
| `DATABASE_URL` | PostgreSQL connection string | SQLite fallback |
| `REDIS_URL` | Redis connection string | In-memory fallback |
| `KAGGLE_USERNAME` | Kaggle username for data download | — |
| `KAGGLE_KEY` | Kaggle API key | — |

---

## License

MIT
