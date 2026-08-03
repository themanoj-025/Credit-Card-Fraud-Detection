# FraudLens (Credit Card Fraud Detection) — Docker Guide

## Quick start

```bash
cp .env.example .env
docker compose up -d
```

Starts Postgres, API (`:8000`), Streamlit dashboard (`:8501`), and the
observability stack (Prometheus `:9090`, Grafana `:3000`, Jaeger `:16686`).

## Image targets

```bash
docker build --target serve -t fraudlens:serve .   # slim API image
docker build --target train -t fraudlens:train .   # full training image
```

## Services & profiles

| Service | Purpose | Port | Profile |
|---------|---------|------|---------|
| `postgres` | Relational DB | `5432` | — |
| `redis` | Rate-limit backend | `6379` | `prod` |
| `jaeger` | Distributed tracing | `16686`, `4317`, `4318` | — |
| `prometheus` | Metrics | `9090` | — |
| `grafana` | Dashboards | `3000` | — |
| `api` | FastAPI predictions | `8000` | — |
| `dashboard` | Streamlit UI | `8501` | — |
| `mlflow` | Experiment tracking | `5000` | `training` |

Optional services use compose profiles: `docker compose --profile training up`.

## Environment

`ANTHROPIC_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`, `DATABASE_URL`,
`REDIS_URL` (see `.env.example`).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| API unhealthy at startup | `start_period: 40s` — model/data loading takes time; check `docker compose logs api` |
| No models | `docker compose exec api python -m src.fraudlens.data.download` (ensure_data_ready runs at boot) |
| Port conflicts | Adjust `ports` per service |
