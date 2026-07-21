"""
FraudLens API — Rate Limiter

Shared rate limiter instance for FastAPI + slowapi.
Extracted into its own module to avoid circular imports between
api/main.py and api/routers/*.py.

Default: Redis-backed storage (safe for multi-worker / multi-replica).
Opt-in: Set RATE_LIMIT_BACKEND=memory for local dev without Docker.

Usage:
    from api.rate_limit import limiter
    from api.main import app  # or just use limiter directly
"""

import logging
import os

from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# ─── Storage Backend Selection ─────────────────────────────────────────────
# Default: Redis (safe for horizontal scaling).
# Opt-in memory: Set RATE_LIMIT_BACKEND=memory for local dev without Redis.

_backend = os.environ.get("RATE_LIMIT_BACKEND", "redis").lower()
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

if _backend == "memory":
    # Explicit opt-in for local dev — not safe for multi-worker deployments
    limiter = Limiter(key_func=get_remote_address)
    logger.warning(
        "Rate limiter using in-memory storage — NOT safe for "
        "multi-worker/multi-replica deployments! Set RATE_LIMIT_BACKEND=redis "
        "and REDIS_URL for production use."
    )
else:
    # Redis-backed (default) — safe for horizontal scaling
    limiter = Limiter(key_func=get_remote_address, storage_uri=redis_url)
    logger.info("Rate limiter using Redis backend: %s", redis_url)
