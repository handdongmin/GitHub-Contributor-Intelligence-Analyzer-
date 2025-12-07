# backend/routers/analyze.py
from fastapi import APIRouter, HTTPException, Response
from typing import Any
import json

import numpy as np

from backend.models.schemas import RepoRequest
from backend.services import github_service, ml_service, preprocess_service

router = APIRouter(prefix="/analyze", tags=["analyze"])


# numpy / ndarray 를 전부 JSON 가능 타입으로 변환
def to_serializable(obj: Any):
    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]

    return obj


@router.post("/heatmap")
def heatmap(payload: RepoRequest):
    try:
        commits = github_service.fetch_commits(
            payload.owner, payload.repo, since=payload.since, until=payload.until
        )
        heatmap_points = preprocess_service.commits_to_heatmap_points(commits)

        response = {"heatmap": heatmap_points}
        safe = to_serializable(response)

        # Response로 JSON 문자열 직접 반환
        return Response(
            content=json.dumps(safe, ensure_ascii=False),
            media_type="application/json",
        )

    except Exception as exc: 
        msg = str(exc)
        if "rate limit" in msg.lower():
            raise HTTPException(
                status_code=429,
                detail="GitHub Rate Limit exceeded. Please try again later.",
            )
        raise HTTPException(status_code=502, detail=msg)


@router.post("/overview")
def overview(payload: RepoRequest):
    try:
        commits = github_service.fetch_commits(
            payload.owner,
            payload.repo,
            since=payload.since,
            until=payload.until,
        )
        issues = github_service.fetch_issues(
            payload.owner,
            payload.repo,
            since=payload.since,
            until=payload.until,
        )

        heatmap_points = preprocess_service.commits_to_heatmap_points(commits)
        messages = preprocess_service.commits_to_messages(commits)
        issue_texts = preprocess_service.issues_to_bodies(issues)
        topic_docs = preprocess_service.build_topic_documents(commits, issues)

        clusters = ml_service.cluster_activity_texts(topic_docs, k=None)
        summaries = ml_service.summarize_commits_and_issues(commits, issues)
        topics = ml_service.topic_model_texts(topic_docs, num_topics=None)

        response = {
            "heatmap": heatmap_points,
            "clusters": clusters,
            "summaries": summaries,
            "topics": topics,
            "counts": {
                "commits": len(commits),
                "issues": len(issues),
            },
        }

        safe = to_serializable(response)

        return Response(
            content=json.dumps(safe, ensure_ascii=False),
            media_type="application/json",
        )

    except Exception as exc: 
        msg = str(exc)
        if "rate limit" in msg.lower():
            raise HTTPException(
                status_code=429,
                detail="GitHub Rate Limit exceeded. Please try again later.",
            )
        raise HTTPException(status_code=502, detail=msg)
