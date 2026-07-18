# ════════════════════════════════════════════════════════════════
# FraudLens — Makefile
# Common development commands
# ════════════════════════════════════════════════════════════════

.PHONY: help install train api dashboard test lint clean docker-up docker-build mlflow-ui

help:
	@echo "FraudLens — Development Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install      Install Python dependencies"
	@echo "  make train        Run full training pipeline"
	@echo "  make api          Start FastAPI server"
	@echo "  make dashboard    Start Streamlit dashboard"
	@echo "  make test         Run all tests"
	@echo "  make lint         Run linters (black, isort, ruff)"
	@echo "  make clean        Remove cache and artifacts"
	@echo "  make docker-up    Start Docker services"
	@echo "  make docker-build Build Docker images"
	@echo "  make mlflow-ui    Start MLflow tracking UI"

install:
	pip install -r requirements.txt
	pre-commit install || echo "pre-commit not installed, skipping"

train:
	python run_pipeline.py

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	streamlit run app/streamlit_app.py --server.port 8501

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src/fraudshield --cov-report=term-missing --tb=short

lint:
	black --check src/ api/ app/ tests/ || echo "Formatting issues found (non-blocking)"
	isort --check-only src/ api/ app/ tests/ || echo "Import sorting issues found (non-blocking)"
	ruff check src/ api/ app/ tests/ || echo "Linting issues found (non-blocking)"

format:
	black src/ api/ app/ tests/
	isort src/ api/ app/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache
	rm -rf build/ dist/ *.egg-info

mlflow-ui:
	@echo "Starting MLflow UI on port 5000..."
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlruns

docker-build:
	docker-compose build

docker-up:
	docker-compose up

docker-down:
	docker-compose down
