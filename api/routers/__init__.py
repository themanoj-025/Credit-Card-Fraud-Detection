"""FastAPI router modules for FraudLens."""

from . import admin, chat, explain, predict, similar_cases

__all__ = ["admin", "chat", "explain", "predict", "similar_cases"]
