"""
LLM Case Narrator Module

Translates SHAP values and transaction features into plain-English
analyst-readable narratives using an LLM.

This turns a wall of SHAP numbers into something a non-technical fraud
analyst could actually read and act on — a real, current best practice
(LLM as an explanation layer on top of a traditional ML model).
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from src.fraudlens.config import LLM_MAX_TOKENS, LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


class CaseNarrator:
    """
    Generate plain-English narratives for flagged transactions.

    Takes SHAP values and transaction features, sends them to an LLM,
    and returns a short analyst-readable explanation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = LLM_MODEL,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> None:
        """
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: LLM model identifier
            max_tokens: Maximum tokens in the response
            temperature: LLM temperature (lower = more deterministic)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _init_client(self):
        """Initialize the Anthropic client."""
        try:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)
        except ImportError:
            logger.error(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )
            raise

    def narrate(
        self,
        transaction: Dict[str, Any],
        fraud_probability: float,
        shap_explanation: List[Dict[str, Any]],
        is_fraud: bool,
    ) -> str:
        """
        Generate a natural-language narrative for a flagged transaction.

        Args:
            transaction: The raw transaction features
            fraud_probability: Model's fraud probability score
            shap_explanation: List of top SHAP features with values
            is_fraud: Whether the transaction was flagged as fraud

        Returns:
            Plain-English narrative string

        Raises:
            RuntimeError: If LLM call fails or API key is missing
        """
        if not self.api_key:
            return self._fallback_narrative(shap_explanation, fraud_probability, is_fraud)

        self._init_client()

        prompt = self._build_prompt(transaction, fraud_probability, shap_explanation, is_fraud)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            narrative = response.content[0].text.strip()
            logger.info("LLM narrative generated (%d chars)", len(narrative))
            return narrative
        except Exception as e:
            logger.warning("LLM call failed: %s. Using fallback narrative.", e)
            return self._fallback_narrative(shap_explanation, fraud_probability, is_fraud)

    def _build_prompt(
        self,
        transaction: Dict[str, Any],
        probability: float,
        shap_explanation: List[Dict[str, Any]],
        is_fraud: bool,
    ) -> str:
        """Build the prompt for the LLM."""
        top_features = shap_explanation[:5]
        features_str = json.dumps(top_features, indent=2)
        amount = transaction.get("Amount", "N/A")
        time_val = transaction.get("Time", "N/A")
        hour = round((float(time_val) % 86400) / 3600, 1) if time_val != "N/A" else "N/A"

        return f"""You are a fraud analysis assistant. Given transaction details and SHAP values, write a short, plain-English paragraph explaining this transaction.

Transaction details:
- Amount: ${amount}
- Time: {hour} hours since first transaction
- Fraud Probability: {probability:.1%}
- Flagged as Fraud: {is_fraud}

Top contributing features (SHAP analysis):
{features_str}

Write 2-3 sentences explaining:
1. What about this transaction is unusual
2. Which features contributed most to the decision
3. Whether this matches known fraud patterns

Be specific, concise, and avoid technical jargon. Write as if for a non-technical fraud analyst."""

    def _fallback_narrative(
        self,
        shap_explanation: List[Dict[str, Any]],
        fraud_probability: float,
        is_fraud: bool,
    ) -> str:
        """Generate a template-based narrative when LLM is unavailable."""
        top = shap_explanation[:3] if shap_explanation else []
        top_features_str = ", ".join(f"{f['feature']} ({f['impact']})" for f in top)

        if is_fraud:
            narrative = (
                f"This transaction was flagged with {fraud_probability:.1%} confidence as potentially fraudulent. "
                f"The primary indicators were {top_features_str}. "
                f"This pattern is consistent with the profile of fraudulent transactions in our reference dataset. "
                f"Recommended action: manual review by a fraud analyst."
            )
        else:
            narrative = (
                f"This transaction appears legitimate ({1-fraud_probability:.1%} confidence). "
                f"No strong fraud indicators were detected in the feature analysis."
            )

        return narrative


def create_case_narrator() -> CaseNarrator:
    """Create a CaseNarrator instance (reads API key from environment)."""
    return CaseNarrator()
