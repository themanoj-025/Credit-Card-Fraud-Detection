# Changelog

All notable changes to FraudLens will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] — 2026-07-21

### Added

#### Observability (Phase 10)
- Structured JSON logging via `structlog` with correlation ID middleware
- Prometheus metrics via `prometheus-fastapi-instrumentator`:
  - `fraudlens_prediction_total{outcome="fraud|legitimate"}`
  - `fraudlens_prediction_latency_ms`, `fraudlens_shap_latency_ms`, `fraudlens_llm_latency_ms`
  - `fraudlens_cache_hit_total`, `fraudlens_model_loaded`, `fraudlens_llm_available`
  - Auto-instrumentation for request count/latency by endpoint
- OpenTelemetry tracing with OTLP exporter to Jaeger
  - FastAPI auto-instrumentation (graceful fallback if packages not installed)
  - Console span exporter for local debugging
- Grafana dashboard (`infra/grafana/dashboard.json`) with 13 panels
- Prometheus scrape config (`infra/prometheus/prometheus.yml`)

#### Configuration (Phase 11)
- `pydantic-settings` `BaseSettings` with env-driven configuration
- Feature flags: `FEATURE_LLM_NARRATOR`, `FEATURE_ANOMALY_SCORE`, `FEATURE_SHAP_EXPLANATION`, `FEATURE_CACHE_PREDICTIONS`, `FEATURE_RAG_RETRIEVAL`
- `.env.example` with all documented environment variables
- Backward-compatible module-level constants

#### Error Handling & Resilience (Phase 12)
- Typed exception classes: `ModelNotLoadedError` (503), `LLMServiceUnavailable` (503), `PredictionError` (500), `RetrieverUnavailable` (503), `InvalidInputError` (422)
- `LLMCircuitBreaker` with CLOSED/OPEN/HALF_OPEN states, configurable thresholds, cooldown multiplier
- Tenacity retry wrapper around Anthropic API (3 attempts, exponential backoff 2s–8s)
- Honest fallback narratives with `[Automated summary — narrative generation unavailable]` prefix
- `docs/RESILIENCE.md` documenting all graceful degradation scenarios

#### Documentation & Polish (Phase 13)
- `CHANGELOG.md` (this file)
- `CONTRIBUTING.md` (developer guide)
- `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)
- GitHub issue templates (bug report + feature request)
- GitHub pull request template

### Changed

- **Package rename:** `fraudshield` → `fraudlens` across all imports, Docker, CI, and docs
- **Configured pre-commit** with `black`, `ruff`, `isort`, `mypy`, `nbstripout`
- **Security:** API key auth via `X-API-Key` header with two tiers (admin/readonly), SHA-256 hashed keys, slowapi rate limiting on all endpoints, Pydantic validation with NaN/Inf rejection, CORS locked to explicit origins, security headers (HSTS, CSP, X-Frame-Options, etc.)
- **Architecture:** Removed global mutable state, injected dependencies via FastAPI `Depends()`, split `FraudPredictor` into `ModelLoader` + `FraudPredictor` + `ShapExplainer` + `ExplanationFormatter`
- **Data layer:** PostgreSQL via docker-compose, Alembic migrations, repository pattern, prediction/feedback/API keys tables
- **Performance:** Async SHAP via FastAPI `BackgroundTasks`, Redis LRU prediction cache, vectorized numpy prediction path, Locust load tests
- **ML quality:** `FeatureEngineer` wired into both training and inference, Optuna hyperparameter tuning (30 trials, 3-fold CV), model card (`docs/MODEL_CARD.md`)
- **API design:** All endpoints under `/v1/`, RFC 7807 error format, cursor-based pagination on `/v1/similar-cases`, per-dependency health check with degraded status
- **Frontend:** Shared API client (`app/api_client.py`), no hardcoded mock data in production pages, loading spinners with retries/timeouts, live synthetic transaction streaming
- **Testing:** 85% coverage target, integration tests with real model fixture, mocked external services (respx), edge-case tests (NaN/Inf/empty batch/boundary), OpenAPI contract tests
- **DevOps:** Multi-stage Docker (slim serve image ~400MB), Trivy/Grype image scanning in CI, blocking lint, parallelized tests (`pytest -n auto`), CD stage to GHCR, K8s manifests (deployment/service/hpa/ingress/configmap/secret)
- **Updated `.gitignore`** with comprehensive patterns for all generated/temp files

### Fixed

- Health endpoint now returns per-dependency status (`{"model": "ok", "llm": "down", ...}`) instead of a single boolean
- Empty batch requests return 422 with clear validation message
- NaN/Inf inputs return 422 instead of causing model errors
- Fallback narrative prefix clearly distinguishes template output from LLM-generated narratives
- Model checksum verification prevents loading tampered artifacts

## [1.0.0] — 2026-07-01

### Added

- Initial project scaffolding — `fraudshield` package with basic ML pipeline
- FastAPI server with prediction endpoints
- Streamlit dashboard with 4 pages: Live Monitor, Case Investigator, Model Performance, Analyst Copilot
- XGBoost, Random Forest, Logistic Regression, LightGBM, Isolation Forest model training
- SHAP explainability integration
- LLM Case Narrator (Anthropic Claude)
- FAISS-based RAG similar-case retrieval
- Docker Compose setup
- Basic CI pipeline (GitHub Actions)
- Unit tests for core modules
- EDA notebooks
- Baseline audit and architecture decision records
