from fastapi import APIRouter, HTTPException
from requests import HTTPError, Timeout

from backend.models.schemas import RepoRequest
from backend.services import github_service

router = APIRouter(prefix="/fetch", tags=["fetch"])


@router.post("/commits")
def get_commits(payload: RepoRequest):
    try:
        return github_service.fetch_commits(
            payload.owner, payload.repo, since=payload.since, until=payload.until
        )
    except HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/issues")
def get_issues(payload: RepoRequest):
    try:
        return github_service.fetch_issues(
            payload.owner, payload.repo, since=payload.since, until=payload.until
        )
    except HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/user/{username}/repos")
def get_user_repos(username: str):
    """
    Return public repositories for the given GitHub username.
    """
    try:
        return github_service.fetch_user_repos(username)
    except Timeout as exc:
        raise HTTPException(status_code=504, detail="GitHub 요청 타임아웃") from exc
    except HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
