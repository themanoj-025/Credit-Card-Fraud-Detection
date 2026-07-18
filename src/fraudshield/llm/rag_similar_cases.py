"""
RAG Similar-Case Retrieval Module

Embeds historical flagged transactions into a local vector store (FAISS)
and retrieves the 3 most similar historical cases for any new flagged
transaction, giving the analyst precedent to reason from.

This is a light, honestly-scoped RAG implementation — not overengineered,
uses FAISS (no hosted vector DB needed).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.fraudshield.config import (
    ALL_FEATURES,
    EMBEDDING_DIM,
    MODELS_DIR,
    RAG_TOP_K,
)

logger = logging.getLogger(__name__)


class SimilarCaseRetriever:
    """
    Retrieve similar historical flagged transactions using FAISS.

    Embeds transactions by their feature vectors and finds nearest neighbors,
    returning their features and known outcomes (confirmed fraud / false positive).
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        top_k: int = RAG_TOP_K,
    ) -> None:
        """
        Args:
            embedding_dim: Dimension of feature vectors
            top_k: Number of similar cases to retrieve
        """
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.index = None
        self.historical_cases: Optional[pd.DataFrame] = None
        self._initialized = False

    def _init_faiss(self):
        """Lazy import FAISS to avoid hard dependency."""
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            logger.warning(
                "faiss not installed. Install with: pip install faiss-cpu"
            )
            raise

    def build_index(
        self,
        historical_data: pd.DataFrame,
        outcome_column: str = "Class",
        feature_columns: Optional[List[str]] = None,
    ) -> "SimilarCaseRetriever":
        """
        Build the FAISS index from historical transaction data.

        Args:
            historical_data: DataFrame with historical transactions
            outcome_column: Column name with fraud labels (0/1)
            feature_columns: Feature columns to embed (default from config)

        Returns:
            Self for chaining
        """
        self._init_faiss()

        features = feature_columns or [c for c in ALL_FEATURES if c in historical_data.columns]
        self.embedding_dim = len(features)

        # Store historical cases with outcomes
        self.historical_cases = historical_data[features + [outcome_column]].copy()
        self.historical_cases.rename(columns={outcome_column: "actual_outcome"}, inplace=True)
        self.historical_cases["is_fraud"] = self.historical_cases["actual_outcome"] == 1

        # Build FAISS index
        embeddings = self.historical_cases[features].values.astype(np.float32)
        self._faiss.normalize_L2(embeddings)

        self.index = self._faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)

        self._initialized = True
        logger.info(
            "FAISS index built: %d cases, %d dimensions",
            len(self.historical_cases),
            self.embedding_dim,
        )
        return self

    def retrieve(
        self,
        transaction: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar historical cases for a given transaction.

        Args:
            transaction: Feature dictionary of the flagged transaction
            top_k: Number of similar cases to return (default from config)

        Returns:
            List of similar cases with similarity scores and outcomes

        Raises:
            RuntimeError: If index hasn't been built
        """
        if not self._initialized or self.index is None:
            raise RuntimeError(
                "Index not built. Call build_index() first with historical data."
            )

        k = top_k or self.top_k

        # Extract features
        features = [c for c in ALL_FEATURES if c in transaction]
        query = np.array([[transaction.get(f, 0.0) for f in features]], dtype=np.float32)
        self._faiss.normalize_L2(query)

        # Search
        scores, indices = self.index.search(query, min(k, self.index.ntotal))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            case = self.historical_cases.iloc[idx]
            results.append({
                "similarity_score": round(float(score), 4),
                "actual_outcome": "confirmed_fraud" if case["is_fraud"] else "false_positive",
                "features": {
                    col: round(float(case[col]), 4)
                    for col in features[:5]  # Top 5 features for display
                },
            })

        return results

    def save(self, path: Optional[str] = None) -> str:
        """Save the FAISS index and historical data to disk."""
        if not self._initialized or self.index is None:
            raise RuntimeError("Index not built. Nothing to save.")

        if path is None:
            path = str(MODELS_DIR / "rag_index")

        Path(path).mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(Path(path) / "index.faiss"))
        self.historical_cases.to_csv(Path(path) / "historical_cases.csv", index=False)
        logger.info("RAG index saved to %s", path)
        return path

    def load(self, path: str) -> "SimilarCaseRetriever":
        """Load a saved FAISS index and historical data."""
        self._init_faiss()
        load_path = Path(path)

        self.index = self._faiss.read_index(str(load_path / "index.faiss"))
        self.historical_cases = pd.read_csv(load_path / "historical_cases.csv")
        self.embedding_dim = self.index.d
        self._initialized = True
        logger.info("RAG index loaded from %s (%d cases)", path, len(self.historical_cases))
        return self


def create_retriever(
    historical_data: Optional[pd.DataFrame] = None,
) -> SimilarCaseRetriever:
    """Create and optionally build a SimilarCaseRetriever."""
    retriever = SimilarCaseRetriever()
    if historical_data is not None:
        retriever.build_index(historical_data)
    return retriever
