"""FastAPI router modules for FraudLens."""

from . import admin
from . import chat
from . import explain
from . import predict
from . import similar_cases

__all__ = ["admin", "chat", "explain", "predict", "similar_cases"]
