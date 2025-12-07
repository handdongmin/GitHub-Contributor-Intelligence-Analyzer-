"""
Preprocessing utilities for commit and issue data.
"""

import re
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

BAD_PATTERNS = [
    "merge pull request",
    "merge branch",
    "merge remote-tracking branch",
    "merge remote tracking branch",
    "chore: merge",
    "merge changes",
    "update readme",
    "fix typo",
    "update docs",
    "add files via upload",
    "files upload",
    "upload files",
    "initial commit",
    "initialization",
    "main branch initialization",
    "dir initialization",
    "directory initialization",
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_noise(text: str) -> str:
    """
    Remove URLs, hashes, issue/PR numbers, backtick code chunks, and collapse spaces.
    """
    if not text:
        return ""
    cleaned = re.sub(r"https?://\S+|git://\S+|www\.\S+", " ", text)
    cleaned = re.sub(r"`[^`]+`", " ", cleaned)
    cleaned = re.sub(r"\b#[0-9]+\b", " ", cleaned)
    cleaned = re.sub(r"\bgh-\d+\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b[0-9a-f]{7,40}\b", " ", cleaned)
    cleaned = re.sub(r"[_*=]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def is_meaningful_text(text: str) -> bool:
    """
    Filter out trivial/meaningless commit or issue texts.
    """
    if not text:
        return False
    lower = text.lower()
    if any(pat in lower for pat in BAD_PATTERNS):
        return False
    tokens = lower.split()
    if len(tokens) < 4:
        return False
    # Reject strings with too many digits/symbols compared to letters.
    letters = sum(ch.isalpha() for ch in text)
    noise = sum(not ch.isalpha() and not ch.isspace() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    if letters and noise / max(1, letters) > 1.0:
        return False
    if letters and digits / max(1, letters) > 0.6:
        return False
    upper_letters = sum(ch.isupper() for ch in text)
    if letters and upper_letters / max(1, letters) > 0.7:
        return False
    return True


def commits_to_messages(commits: Iterable[Dict]) -> List[str]:
    messages: List[str] = []
    for commit in commits:
        message = commit.get("commit", {}).get("message", "")
        messages.append(clean_text(message))
    return messages


def issues_to_bodies(issues: Iterable[Dict]) -> List[str]:
    bodies: List[str] = []
    for issue in issues:
        body = issue.get("title", "") + " " + (issue.get("body") or "")
        bodies.append(clean_text(body))
    return bodies


def prepare_texts_for_topics(texts: Iterable[str]) -> List[str]:
    """
    Clean and filter texts for clustering/topic modeling.
    - Remove URLs, hashes, issue numbers.
    - Drop trivial/short messages.
    """
    cleaned: List[str] = []
    for txt in texts:
        tmp = _strip_noise(txt)
        if not tmp:
            continue
        if not is_meaningful_text(tmp):
            continue
        cleaned.append(tmp)
    return cleaned


def _aggregate_by_author_week(items: Iterable[Tuple[str, str, datetime]], min_tokens: int = 5) -> List[str]:
    """
    Aggregate texts by (author, ISO-year, ISO-week) to build longer documents.
    """
    bucket: Dict[Tuple[str, int, int], List[str]] = {}
    for author, text, dt in items:
        if not text or not dt:
            continue
        key = (author or "unknown", dt.isocalendar()[0], dt.isocalendar()[1])
        bucket.setdefault(key, []).append(text)
    docs: List[str] = []
    for parts in bucket.values():
        doc = " ".join(parts).strip()
        if len(doc.split()) >= min_tokens:
            docs.append(doc)
    return docs


def aggregate_commits_by_author_week(commits: Iterable[Dict]) -> List[str]:
    rows: List[Tuple[str, str, datetime]] = []
    for commit in commits:
        msg = commit.get("commit", {}).get("message", "")
        dt_str = commit.get("commit", {}).get("author", {}).get("date", "")
        author = (
            (commit.get("author") or {}).get("login")
            or (commit.get("commit", {}).get("author", {}) or {}).get("name")
            or "unknown"
        )
        if not dt_str:
            continue
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            continue
        cleaned = _strip_noise(msg)
        if not is_meaningful_text(cleaned):
            continue
        rows.append((author, cleaned, dt))
    return _aggregate_by_author_week(rows)


def aggregate_issues_by_author_week(issues: Iterable[Dict]) -> List[str]:
    rows: List[Tuple[str, str, datetime]] = []
    for issue in issues:
        text = (issue.get("title", "") + " " + (issue.get("body") or "")).strip()
        dt_str = issue.get("created_at") or issue.get("updated_at") or ""
        author = (issue.get("user") or {}).get("login") or "unknown"
        if not dt_str:
            continue
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            continue
        cleaned = _strip_noise(text)
        if not is_meaningful_text(cleaned):
            continue
        rows.append((author, cleaned, dt))
    return _aggregate_by_author_week(rows)


def build_topic_documents(commits: Iterable[Dict], issues: Iterable[Dict]) -> List[str]:
    """
    Build higher-quality documents for clustering/topic modeling:
    - Aggregate by author/week for longer context.
    - Clean/strip noise.
    - Filter trivial messages.
    Fallback: cleaned individual messages when aggregation is empty.
    """
    docs = aggregate_commits_by_author_week(commits) + aggregate_issues_by_author_week(issues)
    docs = [doc for doc in docs if is_meaningful_text(doc)]
    if not docs:
        # Fallback to cleaned individual texts
        raw = commits_to_messages(commits) + issues_to_bodies(issues)
        docs = prepare_texts_for_topics(raw)
    return docs


def to_heatmap_points(commits: Iterable[Dict]) -> List[Dict]:
    """
    Convert commits into {date, count} heatmap points (YYYY-MM-DD).
    """
    counter: Dict[str, int] = {}
    for commit in commits:
        date_str = (
            commit.get("commit", {})
            .get("author", {})
            .get("date", "")
        )
        if not date_str:
            continue
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        key = dt.date().isoformat()
        counter[key] = counter.get(key, 0) + 1
    return [{"date": k, "count": v} for k, v in sorted(counter.items())]


# Alias to keep router calls stable.
def commits_to_heatmap_points(commits: Iterable[Dict]) -> List[Dict]:
    return to_heatmap_points(commits)
