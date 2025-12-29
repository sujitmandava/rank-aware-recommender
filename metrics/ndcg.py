from __future__ import annotations

from typing import Sequence, Set
import math


def _dcg_at_k(relevances: Sequence[int], k: int) -> float:
    """
    Discounted Cumulative Gain at K.
    Assumes relevances are ordered by rank.
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel <= 0:
            continue
        dcg += (2.0 ** rel - 1.0) / math.log2(i + 1.0)
    return dcg


def ndcg_at_k(
    ranked_items: Sequence[int],
    relevant_items: Set[int],
    k: int = 10,
) -> float:
    """
    Binary NDCG@K.

    Args:
        ranked_items: ordered item IDs (best → worst)
        relevant_items: set of ground-truth relevant item IDs
        k: cutoff

    Returns:
        NDCG score in [0, 1]
    """
    if k <= 0 or not ranked_items or not relevant_items:
        return 0.0

    relevances = [1 if item in relevant_items else 0 for item in ranked_items[:k]]
    dcg = _dcg_at_k(relevances, k)

    ideal_relevances = [1] * min(len(relevant_items), k)
    idcg = _dcg_at_k(ideal_relevances, k)

    return 0.0 if idcg == 0.0 else dcg / idcg


def recall_at_k(
    ranked_items: Sequence[int],
    relevant_items: Set[int],
    k: int = 10,
) -> float:
    """
    Recall@K.

    Recall = |relevant ∩ retrieved@K| / |relevant|
    """
    if k <= 0 or not ranked_items or not relevant_items:
        return 0.0

    hits = sum(1 for item in ranked_items[:k] if item in relevant_items)
    return hits / float(len(relevant_items))
