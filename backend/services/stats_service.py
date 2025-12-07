from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List


def contributor_stats(commits: Iterable[Dict]) -> List[Dict]:
    """
    GitHub commit 리스트를 받아서 author 단위 통계를 계산.
    """
    by_author: Dict[str, Dict] = defaultdict(
        lambda: {
            "commits": 0,
            "dates": set(),
            "first_commit": None,
            "last_commit": None,
        }
    )

    for commit in commits:
        author = (commit.get("author") or {}).get("login") or "unknown"
        date_str = (
            commit.get("commit", {})
            .get("author", {})
            .get("date", "")
        )
        if not date_str:
            continue
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        info = by_author[author]
        info["commits"] += 1
        info["dates"].add(dt.date())
        if info["first_commit"] is None or dt < info["first_commit"]:
            info["first_commit"] = dt
        if info["last_commit"] is None or dt > info["last_commit"]:
            info["last_commit"] = dt

    results: List[Dict] = []
    for author, info in by_author.items():
        active_days = len(info["dates"])
        commits = info["commits"]
        avg_per_day = commits / active_days if active_days else commits
        results.append(
            {
                "author": author,
                "commits": commits,
                "active_days": active_days,
                "avg_commits_per_day": avg_per_day,
                "first_commit": info["first_commit"],
                "last_commit": info["last_commit"],
            }
        )
    # commit 수 기준 내림차순 정렬
    results.sort(key=lambda r: r["commits"], reverse=True)
    return results
