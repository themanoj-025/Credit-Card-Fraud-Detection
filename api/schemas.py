"""
FraudLens API Schemas

Pydantic models for request validation and response serialization.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Single transaction to predict."""

    Time: float = Field(..., description="Transaction time in seconds")
    Amount: float = Field(..., ge=0, description="Transaction amount ($)")
    V1: float = Field(default=0.0, description="PCA component V1")
    V2: float = Field(default=0.0, description="PCA component V2")
    V3: float = Field(default=0.0, description="PCA component V3")
    V4: float = Field(default=0.0, description="PCA component V4")
    V5: float = Field(default=0.0, description="PCA component V5")
    V6: float = Field(default=0.0, description="PCA component V6")
    V7: float = Field(default=0.0, description="PCA component V7")
    V8: float = Field(default=0.0, description="PCA component V8")
    V9: float = Field(default=0.0, description="PCA component V9")
    V10: float = Field(default=0.0, description="PCA component V10")
    V11: float = Field(default=0.0, description="PCA component V11")
    V12: float = Field(default=0.0, description="PCA component V12")
    V13: float = Field(default=0.0, description="PCA component V13")
    V14: float = Field(default=0.0, description="PCA component V14")
    V15: float = Field(default=0.0, description="PCA component V15")
    V16: float = Field(default=0.0, description="PCA component V16")
    V17: float = Field(default=0.0, description="PCA component V17")
    V18: float = Field(default=0.0, description="PCA component V18")
    V19: float = Field(default=0.0, description="PCA component V19")
    V20: float = Field(default=0.0, description="PCA component V20")
    V21: float = Field(default=0.0, description="PCA component V21")
    V22: float = Field(default=0.0, description="PCA component V22")
    V23: float = Field(default=0.0, description="PCA component V23")
    V24: float = Field(default=0.0, description="PCA component V24")
    V25: float = Field(default=0.0, description="PCA component V25")
    V26: float = Field(default=0.0, description="PCA component V26")
    V27: float = Field(default=0.0, description="PCA component V27")
    V28: float = Field(default=0.0, description="PCA component V28")


class BatchInput(BaseModel):
    """Batch of transactions to predict."""

    transactions: List[TransactionInput] = Field(
        ..., min_length=1, max_length=1000, description="List of transactions"
    )


class ShapFeature(BaseModel):
    """SHAP feature contribution for a prediction."""

    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature value")
    shap_value: float = Field(..., description="SHAP contribution value")
    impact: str = Field(..., description="Direction of impact (increases/decreases)")


class Explanation(BaseModel):
    """SHAP explanation for a prediction."""

    summary: str = Field(..., description="Plain-English summary of why flagged")
    top_features: List[ShapFeature] = Field(
        ..., description="Top contributing features"
    )


class BusinessImpact(BaseModel):
    """Business impact analysis for a prediction."""

    estimated_loss: float = Field(..., description="Potential fraud loss ($)")
    action: str = Field(..., description="Recommended action")
    review_cost: float = Field(..., description="Cost to review this transaction ($)")


class PredictionResponse(BaseModel):
    """Response for a single prediction."""

    fraud_probability: float = Field(..., ge=0, le=1)
    decision: str = Field(..., pattern="^(FRAUD|LEGITIMATE)$")
    threshold_used: float = Field(..., ge=0, le=1)
    is_fraud: bool
    anomaly_score: Optional[float] = Field(None, ge=0, le=1)
    explanation: Optional[Explanation] = None
    business_impact: Optional[BusinessImpact] = None


class BatchPredictionItem(BaseModel):
    """Individual prediction in a batch response."""

    fraud_probability: float
    decision: str
    is_fraud: bool


class BatchSummary(BaseModel):
    """Summary statistics for a batch prediction."""

    total: int
    flagged_fraud: int
    flagged_legitimate: int
    estimated_review_cost: float


class BatchResponse(BaseModel):
    """Response for batch predictions."""

    predictions: List[BatchPredictionItem]
    summary: BatchSummary


class ExplanationResponse(BaseModel):
    """Response for an explain request."""

    fraud_probability: float
    decision: str
    shap_values: Dict[str, float]
    narrative: Optional[str] = None


class SimilarCase(BaseModel):
    """A similar historical case."""

    similarity_score: float
    actual_outcome: str
    features: Dict[str, float]


class SimilarCasesResponse(BaseModel):
    """Response for similar cases retrieval."""

    transaction_id: str
    similar_cases: List[SimilarCase]


class ChatResponse(BaseModel):
    """Response from the analyst copilot chat."""

    response: str
    tool_calls: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str = "2.0.0"


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_type: str
    threshold: float
    n_features: int
    features: List[str]
    avg_fraud_loss: float
    review_cost: float
    pr_auc: Optional[float] = None
