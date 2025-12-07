"""
Thin wrapper around GitHub REST API for commits, issues, and user repos.
"""

from datetime import datetime
from time import sleep
from typing import Dict, List, Optional, Tuple

import requests

from backend.core.config import get_settings

API_BASE = "https://api.github.com"


def _normalize_owner_repo(owner: str, repo: str) -> Tuple[str, str]:
    """
    Allow users to pass either (owner, repo) separately or a combined 'owner/repo' string.
    """
    owner = (owner or "").strip()
    repo = (repo or "").strip()
    if "/" in repo and not owner:
        parts = repo.split("/", 1)
        if len(parts) == 2:
            owner, repo = parts[0].strip(), parts[1].strip()
    if "/" in owner and not repo:
        parts = owner.split("/", 1)
        if len(parts) == 2:
            owner, repo = parts[0].strip(), parts[1].strip()
    return owner, repo


def _headers(token: Optional[str]) -> Dict[str, str]:
    # 플레이스홀더/짧은 토큰은 무시하고 비인증 호출로 처리
    if token:
        token = token.strip()
        if not token or "your_token" in token or len(token) < 20:
            token = None

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "OSS-Contributor-Analyzer",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def _paginate(
    url: str,
    token: Optional[str],
    params: Optional[Dict[str, str]] = None,
    max_pages: int = 10,
) -> List[Dict]:
    results: List[Dict] = []
    session = requests.Session()
    page_count = 0
    use_token = token
    while url and page_count < max_pages:
        resp = session.get(
            url,
            headers=_headers(use_token),
            params=params,
            timeout=15,
        )

        # 인증 실패 시 토큰 제거 후 한 번 재시도
        if resp.status_code == 401 and use_token:
            use_token = None
            continue

        # 간단 백오프 처리: 5xx 시 재시도
        if resp.status_code >= 500 and page_count < 2:
            sleep(2 ** page_count)
            continue

        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            items = data.get("items", [])
        else:
            items = data
        results.extend(items)

        link = resp.headers.get("Link")
        next_url = None
        if link:
            for part in link.split(","):
                if 'rel="next"' in part:
                    next_url = part.split(";")[0].strip().strip("<>")
                    break
        url = next_url
        params = None
        page_count += 1

    return results


def fetch_commits(
    owner: str,
    repo: str,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> List[Dict]:
    owner, repo = _normalize_owner_repo(owner, repo)
    settings = get_settings()
    url = f"{API_BASE}/repos/{owner}/{repo}/commits"
    params: Dict[str, str] = {}
    if since:
        params["since"] = since.isoformat()
    if until:
        params["until"] = until.isoformat()
    return _paginate(url, settings.github_token, params=params, max_pages=10)


def fetch_issues(
    owner: str,
    repo: str,
    state: str = "all",
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> List[Dict]:
    owner, repo = _normalize_owner_repo(owner, repo)
    settings = get_settings()
    url = f"{API_BASE}/repos/{owner}/{repo}/issues"
    params: Dict[str, str] = {"state": state}
    if since:
        params["since"] = since.isoformat()
    issues = _paginate(url, settings.github_token, params=params, max_pages=10)

    filtered: List[Dict] = []
    for issue in issues:
        if "pull_request" in issue:
            continue
        if until:
            created_at = issue.get("created_at")
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    if dt > until:
                        continue
                except Exception:
                    # If parsing fails, keep the issue to avoid undercounting.
                    pass
        filtered.append(issue)

    return filtered


def fetch_user_repos(username: str) -> List[Dict]:
    """
    공개 리포는 토큰 없이도 조회 가능; /users/{username}/repos 사용.
    """
    settings = get_settings()
    url = f"{API_BASE}/users/{username}/repos"
    params = {
        "sort": "updated",
        "per_page": "100",
        "type": "owner",
    }
    return _paginate(url, settings.github_token, params=params, max_pages=2)
