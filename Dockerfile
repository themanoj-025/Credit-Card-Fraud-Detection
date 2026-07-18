# ─── FraudLens — Multi-stage Docker Build ─────────────────────
# Build: docker build -t fraudlens .
# Run:   docker run -p 8000:8000 fraudlens

# Stage 1: Base Python Environment
FROM python:3.10-slim as base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as app

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY tests/ ./tests/

# Copy trained models (if available)
COPY models/ ./models/

# Create directories
RUN mkdir -p data/raw data/processed reports/figures logs

# Expose ports: FastAPI on 8000, Streamlit on 8501
EXPOSE 8000 8501

# Default: run the API
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
