# Multi-stage Docker build for Credit Card Fraud Detection

# ─── Stage 1: Base Python Environment ──────────────────────
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ─── Stage 2: Dependencies ─────────────────────────────────
FROM base as dependencies

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Stage 3: Application ──────────────────────────────────
FROM dependencies as app

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY monitoring/ ./monitoring/
COPY tests/ ./tests/

# Copy trained models (if available)
COPY models/ ./models/

# Copy configuration files
COPY .gitignore .

# Create directories for data and logs
RUN mkdir -p data/raw data/processed logs

# Expose ports
# FastAPI on 8000, Streamlit on 8501
EXPOSE 8000 8501

# Default command: run the API
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
