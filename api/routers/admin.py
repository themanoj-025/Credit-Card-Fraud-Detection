"""
FraudLens API — Admin Router

Administrative endpoints for API key management.
Accessible only with admin-level API keys.
"""

import hashlib
import logging
import os
import secrets
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from api.auth import require_admin_key
from api.rate_limit import limiter
from src.fraudlens.llm.cost_tracker import cost_tracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["admin"])


class GenerateKeyRequest(BaseModel):
    """Request to generate a new API key."""

    role: str = "readonly"  # "admin" or "readonly"
    description: str = ""


class GeneratedKey(BaseModel):
    """Response containing a newly generated API key."""

    api_key: str
    role: str
    description: str
    sha256_hash: str


class KeyListResponse(BaseModel):
    """Response listing all configured API keys (hashes only)."""

    keys: List[Dict[str, str]]
    count: int


@router.post("/keys", response_model=GeneratedKey)
@limiter.limit("10/hour")
async def generate_api_key(
    request: Request,
    key_request: GenerateKeyRequest,
    _admin_key: str = Depends(require_admin_key),
) -> GeneratedKey:
    """
    Generate a new API key.

    This endpoint returns the raw API key ONCE — it cannot be retrieved
    again. Store it securely. The key is hashed with SHA-256 before storage.

    Requires an admin-level API key in the X-API-Key header.
    """
    if key_request.role not in ("admin", "readonly"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Role must be 'admin' or 'readonly'",
        )

    # Generate a cryptographically secure random key
    raw_key = f"fl_{secrets.token_hex(24)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    # In a production system, this would store the hash in a database.
    # For this dev-focused implementation, we output the config line
    # that should be added to the FRAUDLENS_API_KEYS environment variable.
    config_line = f"{key_hash}={key_request.role}"

    logger.info(
        "Generated new API key (role=%s). Add to FRAUDLENS_API_KEYS: %s",
        key_request.role,
        config_line,
    )

    return GeneratedKey(
        api_key=raw_key,
        role=key_request.role,
        description=key_request.description,
        sha256_hash=key_hash,
    )


@router.get("/keys", response_model=KeyListResponse)
@limiter.limit("30/minute")
async def list_api_keys(
    request: Request,
    _admin_key: str = Depends(require_admin_key),
) -> KeyListResponse:
    """
    List all configured API keys (shows hashes only, not raw keys).

    Requires an admin-level API key.
    """
    config = os.environ.get("FRAUDLENS_API_KEYS", "")
    keys = []

    for entry in config.split(";"):
        entry = entry.strip()
        if "=" in entry:
            hash_val, role = entry.split("=", 1)
            keys.append({"sha256_hash": hash_val, "role": role})

    return KeyListResponse(keys=keys, count=len(keys))


# ─── LLM Usage Endpoint ───────────────────────────────────────────────────


class LLMUsageResponse(BaseModel):
    """LLM cost and usage summary."""

    date: str
    total_cost_usd: float
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    by_model: Dict[str, float]
    by_endpoint: Dict[str, float]


class LLMUsagePeriod(BaseModel):
    """Period selector for LLM usage."""

    period: str = "today"  # "today", "month", "total"


@router.get("/llm-usage", response_model=LLMUsageResponse)
@limiter.limit("30/minute")
async def get_llm_usage(
    request: Request,
    period: str = "today",
    _admin_key: str = Depends(require_admin_key),
) -> LLMUsageResponse:
    """
    Get LLM API cost and usage summary.

    Query param `period`: today | month | total
    Requires an admin-level API key.
    """
    if period == "month":
        summary = cost_tracker.get_month_summary()
    elif period == "total":
        summary = cost_tracker.get_total_summary()
    else:
        summary = cost_tracker.get_today_summary()

    return LLMUsageResponse(
        date=summary.date,
        total_cost_usd=summary.total_cost_usd,
        total_calls=summary.total_calls,
        total_input_tokens=summary.total_input_tokens,
        total_output_tokens=summary.total_output_tokens,
        by_model=summary.by_model,
        by_endpoint=summary.by_endpoint,
    )
