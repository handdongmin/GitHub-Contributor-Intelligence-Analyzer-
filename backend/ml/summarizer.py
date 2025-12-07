# backend/ml/summarizer.py
"""
Summarization utilities for commit/issue text.
"""

import os
import re
from functools import lru_cache
from typing import Iterable, List

import torch
import warnings
from transformers import logging as hf_logging
from transformers import pipeline

# Env override; default to a multilingual summarization model (ko/en).
# 필요시 .env에서 distilbart/koBART 등으로 변경하세요.
DEFAULT_MODEL = os.getenv("SUMMARIZER_MODEL", "csebuetnlp/mT5_multilingual_XLSum")


@lru_cache(maxsize=1)
def get_summarizer(model_name: str = DEFAULT_MODEL):
    """
    HF summarization 모델 lazy-load, 실패 시 fallback.
    """
    # Silence noisy HF warnings about max_length/input_length.
    warnings.filterwarnings(
        "ignore",
        message=".*max_length.*input_length.*",
    )
    warnings.filterwarnings(
        "ignore",
        message="Asking to truncate to max_length",
    )
    hf_logging.set_verbosity_error()

    try:
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("summarization", model=model_name, tokenizer=model_name, device=device)
        # Clamp model default generation lengths to avoid HF internal defaults.
        try:
            if hasattr(pipe, "model") and hasattr(pipe.model, "config"):
                cfg = pipe.model.config
                #max_cfg = 32  # hard cap to avoid overlong generations
                #cfg.max_length = max_cfg
                #if hasattr(pipe.model, "generation_config"):
                #    pipe.model.generation_config.max_length = max_cfg
        except Exception:
            pass
        return pipe
    except Exception:
        def _fallback(text, **kwargs):
            return [{"summary_text": text[:150]}]
        return lambda text, **kwargs: _fallback(text, **kwargs)


def _chunk_text(text: str, max_chars: int = 320) -> List[str]:
    """
    긴 텍스트를 문장 단위로 잘라 여러 번 요약 후 합치는 방식.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > max_chars and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent)

    if current:
        chunks.append(" ".join(current))
    return chunks


def _extractive_summary(text: str, max_sentences: int = 3, max_chars: int = 300) -> str:
    """
    TF-IDF 기반의 간단한 추출 요약. 모델 출력이 너무 길 때 길이를 줄여준다.
    """
    try:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return text[:max_chars]

        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(sentences)
        scores = mat.sum(axis=1).A1
        order = scores.argsort()[::-1]

        picked = []
        length = 0
        for idx in order:
            sent = sentences[idx]
            if sent in picked:
                continue
            if length + len(sent) > max_chars:
                continue
            picked.append(sent)
            length += len(sent)
            if len(picked) >= max_sentences:
                break

        if not picked:
            return text[:max_chars]
        return " ".join(picked)
    except Exception:
        return text[:max_chars]


def _preclean_text(text: str) -> str:
    """
    Remove URLs/boilerplate lines and deduplicate sentences to improve summary focus.
    """
    if not text:
        return ""
    text = re.sub(r"https?://\S+|git://\S+|www\.\S+", " ", text)
    lines = []
    for line in text.splitlines():
        l = line.strip()
        low = l.lower()
        if not l:
            continue
        if low.startswith(("signed-off-by", "reviewed-by", "fixes:", "cc:", "link:")):
            continue
        lines.append(l)
    cleaned = " ".join(lines)
    # Deduplicate sentences
    sentences = re.split(r"(?<=[.!?])\s+", cleaned.strip())
    seen = set()
    deduped = []
    for s in sentences:
        ss = s.strip()
        if not ss:
            continue
        if ss in seen:
            continue
        seen.add(ss)
        deduped.append(ss)
    return " ".join(deduped)


def summarize_text(
    text: str,
    max_length: int = 110,
    min_length: int = 8,
    model_name: str = DEFAULT_MODEL,
    enable_model: bool = True,
) -> str:
    if not text or not text.strip():
        return ""

    cleaned = _preclean_text(text)
    precompressed = (
        _extractive_summary(cleaned, max_sentences=6, max_chars=600) if len(cleaned) > 600 else cleaned
    )

    # 길이 파라미터: 입력보다 적당히 짧게, 하지만 헤드라인이 너무 잘리지 않도록 여유를 둡니다.
    wc = max(1, len(precompressed.split()))
    dyn_max = min(max_length, max(min_length + 6, min(150, int(wc * 1.4) + 20)))
    dyn_min = min(dyn_max - 2, max(min_length, int(wc * 0.45)))
    if dyn_min >= dyn_max:
        dyn_min = max(min_length, dyn_max - 4)

    use_model = enable_model and wc >= 4
    if not use_model:
        return _extractive_summary(precompressed, max_sentences=2, max_chars=dyn_max)

    summarizer = get_summarizer(model_name)
    tokenizer = getattr(summarizer, "tokenizer", None)
    chunks = _chunk_text(precompressed, max_chars=520)
    results: List[str] = []
    for chunk in chunks:
        wc_chunk = max(1, len(chunk.split()))
        # Conservative lengths based on input size; keep max_length <= input_len.
        max_len_for_chunk = max(min_length + 2, min(130, wc_chunk + 32))
        # Target roughly 70-80% of input words for headline-style summaries.
        target_len = max(18, min(110, int(wc_chunk * 0.75) + 6))
        max_len_for_chunk = min(max_len_for_chunk, target_len)
        min_len_for_chunk = max(min_length, min(max_len_for_chunk - 2, int(wc_chunk * 0.5)))
        if tokenizer:
            try:
                tokenized = tokenizer(chunk, truncation=True, return_attention_mask=False)
                input_len = len(tokenized["input_ids"])
                if input_len <= 12:
                    results.append(_extractive_summary(chunk, max_sentences=1, max_chars=dyn_max))
                    continue
                max_len_for_chunk = min(
                    max_len_for_chunk,
                    max(16, int(input_len * 0.8)),
                )
                min_len_for_chunk = max(2, min(min_len_for_chunk, max_len_for_chunk - 2))
            except Exception:
                # Fallback: keep conservative lengths
                max_len_for_chunk = max(min_length + 2, min(max_len_for_chunk, wc_chunk + 16))
                min_len_for_chunk = max(2, min(min_len_for_chunk, max_len_for_chunk - 2))
        if min_len_for_chunk >= max_len_for_chunk:
            min_len_for_chunk = max(2, max_len_for_chunk - 2)
        try:
            out = summarizer(
                chunk,
                max_length=max_len_for_chunk,
                min_length=min_len_for_chunk,
                num_beams=4,
                no_repeat_ngram_size=3,
                truncation=True,
            )[0]["summary_text"]
            result = out.strip()
            norm_out = result.lower().strip(". ")
            norm_in = chunk.lower().strip(". ")
            orig_words = chunk.split()
            res_words = result.split()
            if norm_out == norm_in or len(res_words) >= len(orig_words) * 0.9:
                # Force a shorter extractive summary when the model parrots the input or barely compresses.
                result = _extractive_summary(
                    chunk,
                    max_sentences=1,
                    max_chars=min(max(dyn_max, int(len(chunk) * 0.85)), 240),
                )
            results.append(result)
        except Exception:
            results.append(chunk[:dyn_max])

    joined = " ".join(results)
    if len(results) > 1 or len(joined) > dyn_max * 2:
        joined = _extractive_summary(joined, max_sentences=3, max_chars=dyn_max * 2)
    return joined


def summarize_batch(
    texts: Iterable[str],
    max_length: int = 48,
    min_length: int = 10,
    model_name: str = DEFAULT_MODEL,
) -> List[str]:
    items = list(texts)

    # Skip heavy model-based summarization when the batch is large;
    # this keeps the /analyze endpoint responsive and avoids timeouts.
    use_model = len(items) <= 80

    return [
        summarize_text(
            text,
            max_length=max_length,
            min_length=min_length,
            model_name=model_name,
            enable_model=use_model,
        )
        for text in items
    ]
