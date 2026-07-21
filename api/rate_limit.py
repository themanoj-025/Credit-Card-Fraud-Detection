"""
FraudLens API — Rate Limiter

Shared rate limiter instance for FastAPI + slowapi.
Extracted into its own module to avoid circular imports between
api/main.py and api/routers/*.py.

Usage:
    from api.rate_limit import limiter
    from api.main import app  # or just use limiter directly
"""

import logging
import os

from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# Use Redis if REDIS_URL is set, otherwise fall back to in-memory
redis_url = os.environ.get("REDIS_URL", "")
if redis_url:

    limiter = Limiter(key_func=get_remote_address, storage_uri=redis_url)
    logger.info("Rate limiter using Redis: %s", redis_url)
else:
    limiter = Limiter(key_func=get_remote_address)
    logger.info("Rate limiter using in-memory storage (dev mode)")
