# backend/services/ml_service.py
"""
Convenience layer combining ML utilities for API routes.
"""

from collections import defaultdict
import math
import re
from typing import Dict, List, Optional, Sequence

from backend.ml.clustering import cluster_messages
from backend.ml.prediction import ActivityStats, ProductivityRegressor
from backend.ml.summarizer import summarize_batch, _extractive_summary
from backend.ml.topic_modeling import (
    build_lda_model,
    describe_topics,
    infer_topics,
    prepare_corpus,
    train_lda,
)
from backend.services.preprocess_service import (
    build_topic_documents,
    commits_to_messages,
    issues_to_bodies,
    prepare_texts_for_topics,
)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel

MAX_DOCS_FOR_ML = 500  # 너무 많으면 최근 N개만 사용
TOP_KEYWORDS = 5
TOPIC_STOPWORDS = {
    # 도메인/릴리즈 노트 공통 노이즈
    "git",
    "kernel",
    "linux",
    "org",
    "com",
    "merge",
    "tag",
    "pull",
    "signed",
    "off",
    "by",
    "driver",
    "drivers",
    "support",
    "add",
    "fix",
    "update",
    "updates",
    "branch",
    "branches",
    "tree",
    "upload",
    "uploads",
    "file",
    "files",
    "folder",
    "dir",
    "directory",
    "directories",
    "initial",
    "initialization",
    "init",
    "main",
    "remote",
    "tracking",
    "repo",
    "repository",
    "project",
    "source",
    "code",
    "module",
    "modules",
    "package",
    "packages",
    "service",
    "services",
    "app",
    "apps",
    "application",
}
# Meta/release note boilerplate stopwords
META_STOPWORDS = {
    "fixes",
    "fix",
    "stable",
    "cc",
    "link",
    "links",
    "signed",
    "signed-off-by",
    "reviewed",
    "reviewed-by",
    "from",
}
# Simple heuristic label overrides for common GitHub themes.
TOPIC_LABEL_MAP = {
    "security": {"cve", "security", "vuln", "xss", "csrf", "jwt", "secret", "encryption"},
    "authentication/login": {"auth", "login", "token", "oauth", "sso", "session"},
    "documentation": {"docs", "doc", "readme", "guide", "tutorial", "cookbook"},
    "tests/ci": {"test", "tests", "ci", "pipeline", "coverage", "unit", "integration", "e2e", "jest", "pytest", "cypress", "playwright"},
    "ci/cd/actions": {"github", "actions", "gha", "workflow", "workflows", "runner"},
    "performance": {"perf", "performance", "optimize", "latency", "throughput", "cache", "caching"},
    "refactor/cleanup": {"refactor", "cleanup", "tidy", "simplify"},
    "deps/bump": {"deps", "dependency", "bump", "upgrade", "version"},
    "release/changelog": {"release", "changelog", "tag", "milestone"},
    "frontend/ui": {"ui", "frontend", "layout", "css", "style", "react", "vue", "svelte", "angular", "tailwind", "component"},
    "backend/api": {"api", "endpoint", "handler", "controller", "rest", "graphql", "grpc", "router"},
    "database/storage": {"db", "database", "sql", "schema", "migration", "redis", "postgres", "mysql", "mongo", "sqlite", "elasticsearch", "s3", "storage"},
    "logging/observability": {"log", "logging", "telemetry", "monitor", "metrics", "tracing", "otel", "sentry", "prometheus", "grafana"},
    "infra/devops": {"docker", "container", "compose", "k8s", "kubernetes", "helm", "terraform", "ansible"},
    "build/tooling": {"build", "lint", "linter", "eslint", "prettier", "ruff", "black", "isort", "flake8", "formatter", "webpack", "vite", "rollup", "bazel", "gradle", "maven"},
    "cli/tool": {"cli", "command", "tool", "script"},
}
KOR_STOPWORDS = {
    "및",
    "그리고",
    "이슈",
    "커밋",
    "업데이트",
    "추가",
    "제거",
    "수정",
    "개선",
    "기능",
    "문제",
    "요청",
    "변경",
    "파일",
    "업로드",
    "초기화",
    "브랜치",
    "디렉토리",
    "폴더",
    "병합",
    "메인",
    "프로젝트",
    "레포",
    "저장소",
    "서비스",
    "모듈",
    "코드",
    "디렉토리",
    "폴더",
}

def _cluster_cap(num_docs: int) -> int:
    """
    Cap clusters: small corpora stay compact, large corpora can grow a bit more.
    """
    # Allow a few more clusters by default for medium-sized corpora.
    return 6 if num_docs < 80 else 12


def _topic_cap(num_docs: int) -> int:
    """
    Cap topic counts to avoid fragmented topics on small corpora.
    """
    return 4 if num_docs < 80 else 10


def _shorten(text: str, max_chars: int = 520) -> str:
    """Trim long example text for readability."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _clean_release_note(text: str) -> str:
    """
    Heuristic cleanup for release-note style lines:
    - Drop URLs
    - Trim leading 'Merge tag ...:' / 'Pull ...:' headers
    - Keep only first couple of sentences
    - Drop boilerplate lines (Signed-off-by/Reviewed-by/Fixes/Cc/Link)
    """
    if not text:
        return ""
    cleaned = re.sub(r"https?://\S+|git://\S+|www\.\S+", " ", text)
    lower = cleaned.lower()
    if lower.startswith("merge tag") or lower.startswith("pull "):
        parts = cleaned.split(":", 1)
        if len(parts) > 1 and len(parts[1].strip()) > 0:
            cleaned = parts[1]
    # Remove boilerplate lines
    lines = []
    for line in cleaned.splitlines():
        l = line.strip()
        low = l.lower()
        if not l:
            continue
        if low.startswith(("signed-off-by", "reviewed-by", "fixes:", "cc:", "link:")):
            continue
        lines.append(l)
    cleaned = " ".join(lines) if lines else cleaned

    # Keep first two sentences to reduce noise.
    sentences = re.split(r"(?<=[.!?])\s+", cleaned.strip())
    if len(sentences) > 2:
        cleaned = " ".join(sentences[:2])
    return cleaned.strip()


def _map_label(keywords: List[str], fallback_tokens: List[str]) -> str:
    """
    Map keywords to a readable label using TOPIC_LABEL_MAP.
    If no match, fall back to provided tokens.
    """
    def _explode_tokens(items: List[str]) -> set:
        parts = []
        for item in items:
            low = item.lower()
            parts.append(low)
            for sep in ["/", "-", "_"]:
                if sep in low:
                    parts.extend(low.replace(sep, " ").split())
            parts.extend(low.split())
        return {p for p in parts if p}

    kw_parts = _explode_tokens(keywords)
    fallback_parts = _explode_tokens(fallback_tokens)
    all_parts = kw_parts | fallback_parts

    for label, triggers in TOPIC_LABEL_MAP.items():
        if all_parts & triggers:
            return label
    return ", ".join(fallback_tokens[:3])


def _humanize_label(label: str) -> str:
    """
    Make labels more readable for humans.
    """
    if not label:
        return ""
    tmp = label.replace("_", " ").replace("/", " · ").replace("-", " ")
    tmp = re.sub(r"\s+", " ", tmp).strip()
    return tmp.title()


def _normalize_texts(texts: Sequence[str]) -> List[str]:
    """
    Basic cleaning: remove digits/punctuation, lowercase, drop stopwords.
    Keep shape (empty string stays empty) so indices align with original texts.
    """
    extra_stopwords = ENGLISH_STOP_WORDS | KOR_STOPWORDS | TOPIC_STOPWORDS | META_STOPWORDS
    cleaned: List[str] = []
    for text in texts:
        if not text:
            cleaned.append("")
            continue
        tmp = re.sub(r"[^\w\s]", " ", text)
        tmp = re.sub(r"\d+", " ", tmp)
        tmp = re.sub(r"\s+", " ", tmp).strip().lower()
        tokens = [
            tok
            for tok in tmp.split()
            if tok not in extra_stopwords
        ]
        cleaned.append(" ".join(tokens))
    return cleaned


def _extract_keywords(texts: Sequence[str], top_k: int = TOP_KEYWORDS) -> List[str]:
    """
    Lightweight keyword extractor using TF-IDF (unigram/bigram).
    """
    texts = [t for t in texts if t]
    if not texts:
        return []
    try:
        vec = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            stop_words=list(ENGLISH_STOP_WORDS | TOPIC_STOPWORDS | META_STOPWORDS),
        )
        mat = vec.fit_transform(texts)
        scores = mat.sum(axis=0).A1
        order = scores.argsort()[::-1]
        vocab = vec.get_feature_names_out()
        keywords: List[str] = []
        for idx in order[:top_k]:
            if scores[idx] <= 0:
                continue
            keywords.append(vocab[idx])
        return keywords
    except Exception:
        return []


def summarize_commits_and_issues(commits: List[Dict], issues: List[Dict]) -> Dict:
    messages = commits_to_messages(commits)
    issue_texts = issues_to_bodies(issues)

    # 요약은 시간이 오래 걸릴 수 있으므로 최신 데이터 기준으로 개수를 제한한다.
    total = len(messages) + len(issue_texts)
    if total > MAX_DOCS_FOR_ML:
        max_commits = min(len(messages), MAX_DOCS_FOR_ML // 2)
        max_issues = MAX_DOCS_FOR_ML - max_commits
        messages = messages[-max_commits:] if max_commits else []
        issue_texts = issue_texts[-max_issues:] if max_issues else []

    all_texts = messages + issue_texts
    summaries = summarize_batch(all_texts)
    split = len(messages)
    return {
        "commit_summaries": summaries[:split],
        "issue_summaries": summaries[split:],
    }


def _auto_k(num_docs: int) -> int:
    """
    Pick cluster count from document volume (slightly more aggressive than sqrt).
    """
    if num_docs <= 1:
        return 1
    cap = _cluster_cap(num_docs)
    if num_docs < 4:
        return num_docs
    # Default to at least 4 clusters when possible, then scale with volume.
    base = max(4, int(round(math.sqrt(num_docs) * (1.35 if num_docs >= 80 else 1.2))))
    return min(cap, base, num_docs)


def cluster_activity_texts(texts: List[str], k: Optional[int] = None) -> Dict:
    filtered = prepare_texts_for_topics(texts)
    if not filtered:
        return {"labels": [], "clusters": [], "k_used": 0}

    # 너무 많은 텍스트가 오면 뒤에서부터 잘라서 사용
    if len(filtered) > MAX_DOCS_FOR_ML:
        filtered = filtered[-MAX_DOCS_FOR_ML:]

    cleaned_inputs = [_clean_release_note(t) for t in filtered]
    normalized = _normalize_texts(cleaned_inputs)

    valid_indices: List[int] = [i for i, norm in enumerate(normalized) if norm]
    if not valid_indices:
        return {"labels": [], "clusters": [], "k_used": 0}

    norm_docs = [normalized[i] for i in valid_indices]
    cap = _cluster_cap(len(norm_docs))
    # Prefer a slightly higher base k for small/medium corpora; keep within cap.
    k_base = k if k else max(3, min(cap, int(round(math.sqrt(len(norm_docs) * 1.4)))))
    k_base = min(k_base, cap, len(norm_docs))

    # Use the chosen k directly to avoid collapsing to too few clusters.
    labels_valid = cluster_messages(norm_docs, k=k_base, auto_select_k=False)
    k_used = max(labels_valid) + 1 if labels_valid else 0

    labels = [-1] * len(texts)
    for orig_idx, lbl in zip(valid_indices, labels_valid):
        labels[orig_idx] = lbl
    cluster_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        cluster_indices[label].append(idx)

    cluster_docs: List[str] = []
    ordered_ids: List[int] = sorted(cluster_indices.keys())
    for cid in ordered_ids:
        joined = "\n".join(cleaned_inputs[i] for i in cluster_indices[cid])
        cluster_docs.append(joined)

    # Allow longer summaries for clusters so they don't get cut mid-thought.
    cluster_summaries = summarize_batch(cluster_docs, max_length=220, min_length=60)
    clusters: List[Dict] = []
    for cid, summary in zip(ordered_ids, cluster_summaries):
        indices = cluster_indices[cid]
        examples = [_shorten(cleaned_inputs[i]) for i in indices[:5]]
        keywords = _extract_keywords([normalized[i] for i in indices])
        clusters.append(
            {
                "cluster_id": cid,
                "size": len(indices),
                "summary": summary,
                "keywords": keywords,
                "example_texts": examples,
            }
        )

    return {"labels": labels, "clusters": clusters, "k_used": k_used}



def topic_model_texts(texts: List[str], num_topics: Optional[int] = None):
    texts = prepare_texts_for_topics(texts)
    if len(texts) > MAX_DOCS_FOR_ML:
        texts = texts[-MAX_DOCS_FOR_ML:]

    # Pre-clean long release-note style rows before normalization.
    cleaned_inputs = [_clean_release_note(t) for t in texts]

    normalized = _normalize_texts(cleaned_inputs)
    tokenized, dictionary, corpus = prepare_corpus(normalized)
    if dictionary is None or len(dictionary) == 0 or all(len(vec) == 0 for vec in corpus):
        # Fallback: keep lighter normalization so topics are not empty
        fallback = [re.sub(r"[^\w\s]", " ", t).lower() for t in cleaned_inputs if t]
        normalized = fallback
        tokenized, dictionary, corpus = prepare_corpus(normalized)
        if dictionary is None or len(dictionary) == 0 or all(len(vec) == 0 for vec in corpus):
            return {"topics": [], "doc_topics": []}

    topic_cap = _topic_cap(len(cleaned_inputs))
    auto_topics = max(2, min(topic_cap, int(round(math.sqrt(max(1, len(cleaned_inputs)))))))
    topics_to_use = num_topics if num_topics and num_topics >= 2 else auto_topics

    # Optional coherence-based selection when num_topics is not provided.
    if num_topics is None and len(tokenized) >= 3:
        k_min = 2
        k_max = min(topic_cap, max(auto_topics + 2, 3))
        best_k = None
        best_score = -1.0
        for k in range(k_min, k_max + 1):
            model_try = train_lda(corpus, dictionary, num_topics=k, passes=8)
            if model_try is None:
                continue
            try:
                coh = CoherenceModel(model=model_try, texts=tokenized, dictionary=dictionary, coherence="c_v")
                score = coh.get_coherence()
            except Exception:
                score = -1
            if score > best_score:
                best_score = score
                best_k = k
        if best_k:
            topics_to_use = best_k

    model = train_lda(corpus, dictionary, num_topics=min(topic_cap, topics_to_use), passes=8)
    if model is None:
        return {"topics": [], "doc_topics": []}

    raw_topics = describe_topics(model)
    doc_topics_raw = infer_topics(model, dictionary, normalized)
    doc_topics = [
        [(int(tid), float(score)) for tid, score in doc_list] for doc_list in doc_topics_raw
    ]

    # Assign each document to its strongest topic (for summaries/keywords).
    topic_docs: Dict[int, List[str]] = defaultdict(list)
    for idx, doc_dist in enumerate(doc_topics):
        if not doc_dist:
            continue
        top_tid, _ = max(doc_dist, key=lambda x: x[1])
        topic_docs[top_tid].append(cleaned_inputs[idx])

    topics = []
    for tid, words in raw_topics:
        # Convert numpy types from gensim outputs into plain Python types for JSON serialization.
        converted_words = [(word, float(weight)) for word, weight in words]
        filtered_words = [(w, wt) for w, wt in converted_words if w not in TOPIC_STOPWORDS]
        display_words = filtered_words if filtered_words else converted_words

        docs_for_topic = topic_docs.get(tid, [])
        keywords = _extract_keywords(_normalize_texts(docs_for_topic))
        label_tokens = keywords if keywords else [w for w, _ in display_words]
        raw_label = _map_label(label_tokens, label_tokens)
        label = _humanize_label(raw_label)

        # Brief extractive summary of representative docs.
        summary = (
            _extractive_summary("\n".join(docs_for_topic), max_sentences=2, max_chars=220)
            if docs_for_topic
            else ""
        )
        examples = [_shorten(doc, 200) for doc in docs_for_topic[:3]]

        topics.append(
            {
                "topic_id": int(tid),
                "words": display_words,
                "label": label,
                "keywords": keywords,
                "summary": summary,
                "example_texts": examples,
            }
        )

    return {
        "topics": topics,
        "doc_topics": doc_topics,
    }



def predict_productivity(stats: List[ActivityStats], scores: List[float]) -> Dict:
    # Gracefully handle empty or misaligned inputs to avoid NotFitted errors.
    if not stats or not scores or len(stats) != len(scores):
        return {"mae": None, "r2": None, "predictions": [], "feature_importance": []}

    reg = ProductivityRegressor()
    mae, r2 = reg.evaluate(stats, scores)
    predictions = [float(p) for p in reg.predict(stats)]
    importance = [(feat, float(coef)) for feat, coef in reg.feature_importance()]
    def _clean(val):
        try:
            import math
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return None
            return float(val)
        except Exception:
            return None
    return {
        "mae": _clean(mae),
        "r2": _clean(r2),
        "predictions": predictions,
        "feature_importance": importance,
    }
