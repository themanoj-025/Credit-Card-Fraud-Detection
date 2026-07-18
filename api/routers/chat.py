"""
Analyst Copilot Router — /chat endpoint.

Provides a tool-use-enabled chat interface for analysts to ask
natural-language questions about the simulation state and cases.
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException

from api.state import get_copilot_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["copilot"])


def _get_transaction(tx_id: str) -> str:
    """Get transaction details by ID (tool function)."""
    return f"Transaction {tx_id} details would be retrieved from session data."


def _get_summary_stats() -> str:
    """Get summary statistics of the current session (tool function)."""
    return "Session stats would be computed from the current dashboard state."


TOOLS = {
    "get_transaction": _get_transaction,
    "get_summary_stats": _get_summary_stats,
}


@router.post("/chat")
async def analyst_chat(
    message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> dict:
    """
    Analyst copilot chat endpoint.

    Accepts natural-language questions and returns AI-powered responses
    with access to current simulation data via tool-use.
    """
    copilot_client = get_copilot_client()
    if copilot_client is None:
        raise HTTPException(
            status_code=503,
            detail="Copilot not available. Set ANTHROPIC_API_KEY environment variable.",
        )

    try:
        history = conversation_history or []

        system_prompt = (
            "You are a fraud analysis copilot. You help analysts understand "
            "transactions, model behavior, and simulation statistics. "
            "You have access to these tools:\n"
            "- get_transaction(id): Get details of a specific transaction\n"
            "- get_summary_stats(): Get current simulation statistics\n\n"
            "Answer concisely and accurately. If you need data, use the tools."
        )

        messages = []
        for h in history:
            messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        messages.append({"role": "user", "content": f"{system_prompt}\n\nAnalyst question: {message}"})

        response = copilot_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0.3,
            messages=messages,
        )

        return {"response": response.content[0].text.strip(), "tool_calls": []}
    except Exception as e:
        logger.error("Chat failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
