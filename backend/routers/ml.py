# backend/routers/ml.py
from fastapi import APIRouter, HTTPException

from backend.ml.prediction import ActivityStats
from backend.ml.summarizer import summarize_text
from backend.models.schemas import (
    ClusterPayload,
    ProductivityPayload,
    SummarizePayload,
    TopicPayload,
)
from backend.services import ml_service

router = APIRouter(prefix="/ml", tags=["ml"])


@router.post("/summarize")
def summarize(payload: SummarizePayload):
    try:
        text = payload.text
        if not text:
            combined = "\n".join(t for t in (payload.texts or []) if t)
            text = combined

        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="text is required")

        summary = summarize_text(text, max_length=180, min_length=24)
        return {"summary": summary}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/cluster")
def cluster(payload: ClusterPayload):
    try:
        return ml_service.cluster_activity_texts(payload.texts, k=payload.k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/topics")
def topics(payload: TopicPayload):
    try:
        return ml_service.topic_model_texts(
            payload.texts,
            num_topics=payload.num_topics,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/predictivity")
def predictivity(payload: ProductivityPayload):
    try:
        stats = [
            ActivityStats(
                commits=s.commits,
                issues=s.issues,
                additions=s.additions,
                deletions=s.deletions,
                active_days=s.active_days,
                prs=s.prs,
            )
            for s in payload.stats
        ]
        return ml_service.predict_productivity(stats, payload.scores)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
