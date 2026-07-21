"""
FraudLens — SQLAlchemy ORM Models

System-of-record tables for the fraud detection platform:
- predictions: every prediction made by the model
- api_keys: hashed API keys for auth
- feedback: analyst-confirmed fraud/legit labels (enables feedback loop)
- drift_events: detected data drift events

Usage:
    from src.fraudlens.persistence.models import PredictionModel

    # Query
    preds = await session.execute(
        select(PredictionModel).where(PredictionModel.decision == "FRAUD")
    )
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .database import Base

# Use appropriate types for SQLite vs PostgreSQL compatibility


def _uuid_column():
    """Return a UUID column compatible with both SQLite and PostgreSQL."""
    from sqlalchemy import String as SAString
    from sqlalchemy import TypeDecorator

    class _UUID(TypeDecorator):
        """Platform-independent UUID type."""

        impl = SAString(36)
        cache_ok = True

        def process_bind_param(self, value, dialect):
            if value is None:
                return None
            return str(value)

        def process_result_value(self, value, dialect):
            if value is None:
                return None
            return uuid.UUID(value) if isinstance(value, str) else value

    return Column(_UUID(36), primary_key=True, default=uuid.uuid4)


def _json_column():
    """Return a JSON column compatible with both SQLite and PostgreSQL."""
    from sqlalchemy import JSON

    return Column(JSON, nullable=True)


class PredictionModel(Base):
    """Record of every prediction made by the model."""

    __tablename__ = "predictions"

    id = _uuid_column()
    transaction_id = Column(String(64), nullable=True, index=True)
    fraud_probability = Column(Float, nullable=False)
    decision = Column(String(32), nullable=False)  # FRAUD or LEGITIMATE
    threshold_used = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    model_version = Column(String(64), nullable=True)
    latency_ms = Column(Float, nullable=True)
    features = _json_column()
    shap_values = _json_column()
    anomaly_score = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Relationships
    feedback = relationship("FeedbackModel", back_populates="prediction", uselist=False)

    def __repr__(self) -> str:
        return f"<Prediction {self.id[:8]} decision={self.decision} prob={self.fraud_probability:.4f}>"


class ApiKeyModel(Base):
    """Stored API keys with hashed values and role information."""

    __tablename__ = "api_keys"

    id = _uuid_column()
    key_hash = Column(String(64), nullable=False, unique=True, index=True)
    role = Column(String(32), nullable=False, default="readonly")  # admin or readonly
    description = Column(String(255), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<ApiKey {self.id[:8]} role={self.role}>"


class FeedbackModel(Base):
    """Analyst feedback on predictions — enables the model feedback loop.

    Links back to a prediction via prediction_id.
    """

    __tablename__ = "feedback"

    id = _uuid_column()
    prediction_id = Column(
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    confirmed_fraud = Column(Boolean, nullable=False)  # True if analyst confirms fraud
    analyst_notes = Column(Text, nullable=True)
    reviewed_by = Column(String(128), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    prediction = relationship("PredictionModel", back_populates="feedback")

    def __repr__(self) -> str:
        return f"<Feedback {self.id[:8]} confirmed_fraud={self.confirmed_fraud}>"


class DriftEventModel(Base):
    """Recorded data drift events for monitoring."""

    __tablename__ = "drift_events"

    id = _uuid_column()
    feature_name = Column(String(64), nullable=False, index=True)
    drift_score = Column(Float, nullable=False)
    p_value = Column(Float, nullable=True)
    alert_type = Column(
        String(32), nullable=False, default="drift"
    )  # drift, warning, alert
    window_size = Column(Integer, nullable=True)
    details = _json_column()
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    def __repr__(self) -> str:
        return f"<DriftEvent {self.feature_name} score={self.drift_score:.4f}>"
