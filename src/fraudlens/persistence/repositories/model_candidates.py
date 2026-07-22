"""
FraudLens — Model Candidate Repository

Handles database operations for model candidate records.
These are the "waiting for human review" entries created by
the automated retraining trigger.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import ModelCandidateModel
from .base import BaseRepository


class ModelCandidateRepository(BaseRepository[ModelCandidateModel]):
    """Repository for model candidate records."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ModelCandidateModel)

    async def create_candidate(
        self,
        model_version: str,
        trigger: str,
        trigger_detail: Optional[str] = None,
        pr_auc: Optional[float] = None,
        f1_score: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        threshold: Optional[float] = None,
        mlflow_run_id: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> ModelCandidateModel:
        """Create a new model candidate record."""
        return await self.create(
            model_version=model_version,
            trigger=trigger,
            trigger_detail=trigger_detail,
            pr_auc=pr_auc,
            f1_score=f1_score,
            precision=precision,
            recall=recall,
            threshold=threshold,
            mlflow_run_id=mlflow_run_id,
            model_path=model_path,
            status="candidate",
        )

    async def get_candidates(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ModelCandidateModel]:
        """Get model candidates, optionally filtered by status."""
        stmt = select(ModelCandidateModel).order_by(
            ModelCandidateModel.created_at.desc()
        )
        if status:
            stmt = stmt.where(ModelCandidateModel.status == status)
        stmt = stmt.offset(offset).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_version(
        self, model_version: str
    ) -> Optional[ModelCandidateModel]:
        """Get a candidate by its version string."""
        result = await self.session.execute(
            select(ModelCandidateModel).where(
                ModelCandidateModel.model_version == model_version
            )
        )
        return result.scalar_one_or_none()

    async def promote(self, model_version: str) -> Optional[ModelCandidateModel]:
        """Promote a candidate to production (human-gated)."""
        candidate = await self.get_by_version(model_version)
        if candidate is None:
            return None
        if candidate.status != "candidate":
            return None  # Already promoted or rejected

        candidate.status = "promoted"
        candidate.promoted_at = datetime.utcnow()
        await self.session.commit()
        await self.session.refresh(candidate)
        return candidate

    async def reject(
        self, model_version: str
    ) -> Optional[ModelCandidateModel]:
        """Reject a candidate (mark as rejected)."""
        candidate = await self.get_by_version(model_version)
        if candidate is None:
            return None
        if candidate.status != "candidate":
            return None

        candidate.status = "rejected"
        await self.session.commit()
        await self.session.refresh(candidate)
        return candidate

    async def get_statistics(self) -> Dict[str, Any]:
        """Get candidate statistics."""
        stmt = select(
            func.count(ModelCandidateModel.id).label("total"),
        )
        result = await self.session.execute(stmt)
        total = result.scalar() or 0

        # Count by status
        status_counts: Dict[str, int] = {}
        for status_val in ("candidate", "promoted", "rejected"):
            stmt_status = select(func.count(ModelCandidateModel.id)).where(
                ModelCandidateModel.status == status_val
            )
            result_status = await self.session.execute(stmt_status)
            status_counts[status_val] = result_status.scalar() or 0

        return {
            "total_candidates": total,
            **status_counts,
        }

    async def get_latest_promoted(self) -> Optional[ModelCandidateModel]:
        """Get the most recently promoted candidate (current production model)."""
        stmt = (
            select(ModelCandidateModel)
            .where(ModelCandidateModel.status == "promoted")
            .order_by(ModelCandidateModel.promoted_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
