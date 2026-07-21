"""
FraudLens — Repository Pattern

Repository classes abstract database access behind clean interfaces,
so routers and services never touch SQLAlchemy sessions directly.

Usage:
    from src.fraudlens.persistence.repositories import PredictionRepository

    repo = PredictionRepository(session)
    pred = await repo.create(fraud_probability=0.95, decision="FRAUD", ...)
"""

from .api_keys import ApiKeyRepository
from .drift_events import DriftEventRepository
from .feedback import FeedbackRepository
from .predictions import PredictionRepository

__all__ = [
    "ApiKeyRepository",
    "DriftEventRepository",
    "FeedbackRepository",
    "PredictionRepository",
]
