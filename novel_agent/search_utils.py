from __future__ import annotations

import re
from datetime import date
from typing import Any, Sequence


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+")
DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def tokenize_for_search(text: str) -> list[str]:
    clean = text.lower().strip()
    if not clean:
        return []

    tokens: list[str] = []
    for piece in TOKEN_PATTERN.findall(clean):
        if not piece:
            continue
        if piece.isascii():
            if len(piece) > 1:
                tokens.append(piece)
            continue

        tokens.append(piece)
        for size in (2, 3):
            if len(piece) < size:
                continue
            for index in range(len(piece) - size + 1):
                tokens.append(piece[index : index + size])
    return tokens


def hybrid_search_score(query: str, text: str, *, embedding_backend: Any) -> float:
    scores = hybrid_search_scores(query, [text], embedding_backend=embedding_backend)
    return scores[0] if scores else 0.0


def hybrid_search_scores(query: str, texts: Sequence[str], *, embedding_backend: Any) -> list[float]:
    clean_query = query.strip()
    if not clean_query:
        return [0.0 for _ in texts]
    if embedding_backend is None:
        raise RuntimeError("embedding_backend_not_configured")

    indexed_texts = [(index, text.strip()) for index, text in enumerate(texts)]
    non_empty = [(index, text) for index, text in indexed_texts if text]
    scores = [0.0 for _ in indexed_texts]
    if not non_empty:
        return scores

    batch_fn = getattr(embedding_backend, "similarity_batch", None)
    if callable(batch_fn):
        raw_scores = list(batch_fn(clean_query, [text for _, text in non_empty]))
    else:
        single_fn = getattr(embedding_backend, "similarity", None)
        if not callable(single_fn):
            raise RuntimeError("embedding_backend_missing_similarity_methods")
        raw_scores = [single_fn(clean_query, text) for _, text in non_empty]

    for (index, _), score in zip(non_empty, raw_scores):
        scores[index] = max(float(score), 0.0)
    return scores


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(tokenize_for_search(left))
    right_tokens = set(tokenize_for_search(right))
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def mmr_rerank(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    text_key: str = "text",
    lambda_param: float = 0.74,
) -> list[dict[str, Any]]:
    if limit <= 0 or not candidates:
        return []

    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    while remaining and len(selected) < limit:
        best_index = 0
        best_score = float("-inf")
        for index, item in enumerate(remaining):
            relevance = float(item.get("score", 0.0))
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    jaccard_similarity(str(item.get(text_key, "")), str(existing.get(text_key, "")))
                    for existing in selected
                )
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * diversity_penalty
            if mmr_score > best_score:
                best_index = index
                best_score = mmr_score
        selected.append(remaining.pop(best_index))
    return selected


def extract_snippet(text: str, query: str, max_chars: int = 220) -> str:
    clean_text = text.strip()
    if len(clean_text) <= max_chars:
        return clean_text

    lowered_text = clean_text.lower()
    lowered_query = query.lower().strip()
    pivot = lowered_text.find(lowered_query) if lowered_query else -1

    if pivot < 0:
        unique_tokens = []
        seen = set()
        for token in tokenize_for_search(query):
            if token in seen:
                continue
            seen.add(token)
            unique_tokens.append(token)
        unique_tokens.sort(key=len, reverse=True)
        for token in unique_tokens:
            pivot = lowered_text.find(token)
            if pivot >= 0:
                break

    if pivot < 0:
        pivot = 0

    start = max(0, pivot - max_chars // 3)
    end = min(len(clean_text), start + max_chars)
    snippet = clean_text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(clean_text):
        snippet = snippet + "..."
    return snippet


def recency_multiplier(source_id: str, *, half_life_days: float = 14.0, floor: float = 0.72) -> float:
    match = DATE_PATTERN.search(source_id)
    if not match:
        return 1.0
    try:
        target = date.fromisoformat(match.group(1))
    except ValueError:
        return 1.0
    age_days = max((date.today() - target).days, 0)
    decay = 0.5 ** (age_days / max(half_life_days, 1.0))
    return floor + (1.0 - floor) * decay


def format_line_target(base_target: str, line_start: int, line_end: int) -> str:
    if line_start <= 0 or line_end <= 0:
        return base_target
    if line_start == line_end:
        return f"{base_target}#L{line_start}"
    return f"{base_target}#L{line_start}-L{line_end}"
