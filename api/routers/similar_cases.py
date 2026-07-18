"""
Similar Cases Router — /similar-cases endpoint.

Retrieves similar historical flagged transactions using FAISS-based RAG.
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from api.schemas import SimilarCase, SimilarCasesResponse
from api.state import get_case_retriever
from src.fraudshield.config import RAG_TOP_K

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])


@router.post("/similar-cases", response_model=SimilarCasesResponse)
async def get_similar_cases(
    transaction: dict,
    top_k: int = Query(default=RAG_TOP_K, ge=1, le=20),
) -> SimilarCasesResponse:
    """
    Retrieve similar historical cases for a flagged transaction.
    """
    case_retriever = get_case_retriever()
    if case_retriever is None or not case_retriever._initialized:
        raise HTTPException(
            status_code=503,
            detail="Case retriever not initialized. Build a RAG index first.",
        )

    try:
        similar = case_retriever.retrieve(transaction, top_k=top_k)
        cases = [
            SimilarCase(
                similarity_score=c["similarity_score"],
                actual_outcome=c["actual_outcome"],
                features=c["features"],
            )
            for c in similar
        ]

        return SimilarCasesResponse(
            transaction_id=f"tx_{hash(str(transaction)) & 0xFFFFFFFF}",
            similar_cases=cases,
        )
    except Exception as e:
        logger.error("Similar cases retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
