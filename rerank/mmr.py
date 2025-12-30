from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)


def build_item_vectors_from_genres(items_df) -> Dict[int, np.ndarray]:
    """
    Build normalized item vectors from MovieLens item genre one-hot columns.

    Expected items_df columns:
      item_id, title, <genre columns...>
    where genre columns are numeric 0/1.

    Returns:
      dict: item_id -> unit vector
    """
    cols = list(items_df.columns)
    if "item_id" not in cols:
        raise ValueError("items_df must contain 'item_id' column")

    # Assume any non-id/title columns are features
    feature_cols = [c for c in cols if c not in ("item_id", "title")]
    if not feature_cols:
        raise ValueError(
            "items_df has no feature columns. "
            "MMR needs item vectors (e.g., genre one-hot columns)."
        )

    vectors: Dict[int, np.ndarray] = {}
    for row in items_df[["item_id"] + feature_cols].itertuples(index=False):
        item_id = int(row[0])
        vec = np.asarray(row[1:], dtype=float)
        vectors[item_id] = _safe_normalize(vec)
    return vectors


def mmr_rerank(
    candidates: Sequence[int],
    relevance_scores: Dict[int, float],
    item_vectors: Dict[int, np.ndarray],
    k: int = 10,
    lambda_relevance: float = 0.7,
) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) re-ranking.

    Selects items greedily:
      argmax_i [ lambda * rel(i) - (1-lambda) * max_{j in S} sim(i, j) ]

    - candidates: ordered or unordered pool of item_ids (size N)
    - relevance_scores: base-model relevance score per candidate item_id
    - item_vectors: normalized vectors for cosine similarity (dot product)
    - k: final list size
    - lambda_relevance: tradeoff in [0,1]; higher => more relevance, less diversity

    Behavior if vectors missing:
      sim treated as 0 (no diversity penalty).
    """
    if not 0.0 <= lambda_relevance <= 1.0:
        raise ValueError("lambda_relevance must be in [0,1]")
    if k <= 0:
        raise ValueError("k must be positive")
    if not candidates:
        return []

    # Keep unique candidates in original order
    seen = set()
    pool: List[int] = []
    for it in candidates:
        if it not in seen:
            pool.append(int(it))
            seen.add(int(it))

    k = min(k, len(pool))

    selected: List[int] = []
    selected_vecs: List[np.ndarray] = []

    # Pre-fetch vectors for pool (optional micro-optimization)
    vec_cache: Dict[int, np.ndarray | None] = {}
    for it in pool:
        vec_cache[it] = item_vectors.get(it)

    for _ in range(k):
        best_item = None
        best_score = -float("inf")

        for it in pool:
            if it in selected:
                continue

            rel = float(relevance_scores.get(it, 0.0))

            # diversity penalty = max cosine similarity to already selected
            penalty = 0.0
            v = vec_cache[it]
            if v is not None and selected_vecs:
                # cosine similarity since vectors are normalized
                penalty = max(float(np.dot(v, sv)) for sv in selected_vecs)

            score = lambda_relevance * rel - (1.0 - lambda_relevance) * penalty
            if score > best_score:
                best_score = score
                best_item = it

        if best_item is None:
            break

        selected.append(best_item)
        vbest = vec_cache[best_item]
        if vbest is not None:
            selected_vecs.append(vbest)

    return selected


def rank_based_relevance(candidates: Sequence[int]) -> Dict[int, float]:
    """
    Convert a ranked candidate list into decreasing relevance scores.

    rel(rank) = 1 / (rank + 1)

    Works for popularity/content (ranked lists) and for MF if you don't expose raw scores.
    """
    scores: Dict[int, float] = {}
    for r, it in enumerate(candidates):
        scores[int(it)] = 1.0 / float(r + 1)
    return scores
