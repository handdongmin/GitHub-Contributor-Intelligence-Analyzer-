import os
from functools import lru_cache
import math
from typing import Iterable, List, Optional, Sequence

from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Lazy-load sentence transformer with env override.
    CLUSTER_MODEL can be set to larger models (e.g., intfloat/e5-base-v2, intfloat/e5-large-v2).
    Heavier models give better quality but use more RAM/VRAM.
    """
    # Balanced default (good quality, CPU-usable); override with CLUSTER_MODEL if needed.
    model_name = os.getenv("CLUSTER_MODEL", "all-mpnet-base-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def _best_k_by_silhouette(embeddings, k_min: int = 2, k_max: int = 12) -> Optional[int]:
    n = len(embeddings)
    if n < k_min:
        return None
    best_k = None
    best_score = -1.0
    for k in range(k_min, min(k_max, n) + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        try:
            score = silhouette_score(embeddings, labels, metric="euclidean")
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def cluster_messages(messages: Sequence[str], k: int = 3, auto_select_k: bool = False) -> List[int]:
    msgs = list(messages)
    if not msgs:
        return []

    # Keep cluster counts reasonable for small corpora, but allow more granularity.
    cluster_cap = 8 if len(msgs) < 120 else 16

    model = get_model()
    emb = model.encode(msgs)

    effective_k = min(max(1, k), len(msgs), cluster_cap)
    if auto_select_k and len(msgs) >= 3:
        # Grow candidate clusters with corpus size, cap at 12 to avoid overfragmentation.
        search_max = min(cluster_cap, len(msgs), max(4, int(round(math.sqrt(len(msgs)) * 1.8))))
        chosen = _best_k_by_silhouette(
            emb,
            k_min=2,
            k_max=search_max,
        )
        if chosen:
            effective_k = min(chosen, len(msgs), cluster_cap)

    km = KMeans(n_clusters=effective_k, random_state=42, n_init=10)
    labels = km.fit_predict(emb)
    return labels.tolist()
