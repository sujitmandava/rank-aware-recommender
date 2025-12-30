from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
from typing import Dict, List, Set, Optional

import pandas as pd

from rerank.mmr import build_item_vectors_from_genres, mmr_rerank, rank_based_relevance
from data.utils import load_movielens_100k, time_split_per_user, binarize_relevance
from models import content, mf, popularity
from metrics.ndcg import ndcg_at_k, recall_at_k
from metrics.diversity import coverage_at_k


# -----------------------------
# Popularity model (baseline)
# -----------------------------

# def fit_popularity(train_df: pd.DataFrame) -> List[int]:
#     """
#     Fits a popularity model by interaction count.
#     Returns items ranked from most → least popular.
#     """
#     pop = (
#         train_df.groupby("item_id")["user_id"]
#         .count()
#         .sort_values(ascending=False)
#     )
#     return pop.index.tolist()


# def recommend_popularity(
#     ranked_items: List[int],
#     seen_items: Set[int],
#     k: int,
# ) -> List[int]:
#     """
#     Recommend top-K popular items excluding seen items.
#     """
#     recs = []
#     for item in ranked_items:
#         if item not in seen_items:
#             recs.append(item)
#         if len(recs) == k:
#             break
#     return recs


# -----------------------------
# Evaluation utilities
# -----------------------------

def build_seen_items(train_df: pd.DataFrame) -> Dict[int, Set[int]]:
    seen = defaultdict(set)
    for row in train_df.itertuples(index=False):
        seen[int(row.user_id)].add(int(row.item_id))
    return seen


def build_relevant_items(test_df: pd.DataFrame) -> Dict[int, Set[int]]:
    relevant = defaultdict(set)
    for row in test_df.itertuples(index=False):
        if int(row.relevant) == 1:
            relevant[int(row.user_id)].add(int(row.item_id))
    return relevant


# -----------------------------
# Main experiment
# -----------------------------

def main():
    K = 10
    CANDIDATES = 200
    LAMBDA = 0.7
    TEST_RATIO = 0.2
    THRESHOLD = 3.5

    # 1. Load + split data
    ds = load_movielens_100k(root="data", download=True)
    train, test = time_split_per_user(ds.ratings, test_ratio=TEST_RATIO)

    train = binarize_relevance(train, threshold=THRESHOLD)
    test = binarize_relevance(test, threshold=THRESHOLD)

    # 2. Prepare evaluation structures
    seen_by_user = build_seen_items(train)
    relevant_by_user = build_relevant_items(test)

    users = sorted(relevant_by_user.keys())

    catalog_size = ds.ratings["item_id"].nunique()
    item_vectors = build_item_vectors_from_genres(ds.items)

    def evaluate_and_log(
        model_name: str,
        recs_by_user: Dict[int, List[int]],
        extra: Optional[Dict[str, float]] = None,
    ):
        ndcgs = []
        recalls = []

        for u in users:
            ndcgs.append(
                ndcg_at_k(
                    recs_by_user[u],
                    relevant_by_user[u],
                    k=K,
                )
            )
            recalls.append(
                recall_at_k(
                    recs_by_user[u],
                    relevant_by_user[u],
                    k=K,
                )
            )

        mean_ndcg = sum(ndcgs) / len(ndcgs)
        mean_recall = sum(recalls) / len(recalls)

        coverage = coverage_at_k(recs_by_user, k=K)
        coverage_norm = coverage / catalog_size

        print(f"=== {model_name} Evaluation ===")
        print(f"NDCG@{K}:        {mean_ndcg:.4f}")
        print(f"Recall@{K}:      {mean_recall:.4f}")
        print(f"Coverage@{K}:    {coverage} / {catalog_size} = {coverage_norm:.4f}")
        print()

        log_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model_name,
            "k": K,
            "candidates": CANDIDATES,
            "lambda": LAMBDA if "mmr" in model_name else None,
            "test_ratio": TEST_RATIO,
            "threshold": THRESHOLD,
            "users_evaluated": len(users),
            "ndcg": mean_ndcg,
            "recall": mean_recall,
            "coverage": coverage,
            "catalog_size": catalog_size,
            "coverage_norm": coverage_norm,
        }
        if extra:
            log_results.update(extra)

        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / "experiments.csv"
        header_needed = not results_path.exists()

        with results_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_results.keys()))
            if header_needed:
                writer.writeheader()
            writer.writerow(log_results)

    # 3. Train + eval popularity
    pop_model = popularity.fit(train)
    pop_recs: Dict[int, List[int]] = {}
    pop_mmr_recs: Dict[int, List[int]] = {}
    for u in users:
        pop_candidates = popularity.recommend(
            model=pop_model,
            user_id=u,
            exclude_items=seen_by_user[u],
            k=CANDIDATES,
        )
        pop_recs[u] = pop_candidates[:K]
        rel_scores = rank_based_relevance(pop_candidates)
        pop_mmr_recs[u] = mmr_rerank(
            candidates=pop_candidates,
            relevance_scores=rel_scores,
            item_vectors=item_vectors,
            k=K,
            lambda_relevance=LAMBDA,
        )
    evaluate_and_log("popularity", pop_recs)
    evaluate_and_log("popularity_mmr", pop_mmr_recs)

    # 4. Train + eval MF (BPR on positives)
    positives = train[train["relevant"] == 1][["user_id", "item_id"]]
    mf_model = mf.fit_bpr(
        positives,
        k=64,
        lr=0.05,
        reg=0.01,
        epochs=50,
        n_neg=3,
        seed=42,
    )
    mf_recs: Dict[int, List[int]] = {}
    mf_mmr_recs: Dict[int, List[int]] = {}
    for u in users:
        mf_candidates = mf.recommend(
            model=mf_model,
            user_id=u,
            exclude_items=seen_by_user[u],
            k=CANDIDATES,
        )
        mf_recs[u] = mf_candidates[:K]
        rel_scores = rank_based_relevance(mf_candidates)
        mf_mmr_recs[u] = mmr_rerank(
            candidates=mf_candidates,
            relevance_scores=rel_scores,
            item_vectors=item_vectors,
            k=K,
            lambda_relevance=LAMBDA,
        )
    evaluate_and_log("mf_bpr", mf_recs)
    evaluate_and_log("mf_bpr_mmr", mf_mmr_recs)

    # 5. Train + eval content-based (genre features)
    content_model = content.fit(
        train_df=train,
        items_df=ds.items,
    )
    content_recs: Dict[int, List[int]] = {}
    content_mmr_recs: Dict[int, List[int]] = {}
    for u in users:
        content_candidates = content.recommend(
            model=content_model,
            user_id=u,
            exclude_items=seen_by_user[u],
            k=CANDIDATES,
        )
        content_recs[u] = content_candidates[:K]
        rel_scores = rank_based_relevance(content_candidates)
        content_mmr_recs[u] = mmr_rerank(
            candidates=content_candidates,
            relevance_scores=rel_scores,
            item_vectors=item_vectors,
            k=K,
            lambda_relevance=LAMBDA,
        )
    evaluate_and_log("content", content_recs)
    evaluate_and_log("content_mmr", content_mmr_recs)


if __name__ == "__main__":
    main()
