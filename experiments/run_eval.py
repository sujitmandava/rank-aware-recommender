from __future__ import annotations
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
from typing import Dict, List, Set

import pandas as pd

from data.utils import load_movielens_100k, time_split_per_user, binarize_relevance
from models.popularity import fit as p_fit
from models.popularity import recommend as p_recommend
from models.mf import fit, recommend
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

    # 3. Train popularity model
    popular_items = fit(train)

    # 4. Generate recommendations
    recs_by_user: Dict[int, List[int]] = {}
    for u in users:
        recs_by_user[u] = recommend(
            model=popular_items,
            user_id=u,
            exclude_items=seen_by_user[u],
            k=K,
        )

    # 5. Compute metrics
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
    catalog_size = ds.ratings["item_id"].nunique()
    coverage_norm = coverage / catalog_size

    # 6. Report
    print("=== Popularity Baseline Evaluation ===")
    print(f"NDCG@{K}:        {mean_ndcg:.4f}")
    print(f"Recall@{K}:      {mean_recall:.4f}")
    print(f"Coverage@{K}:    {coverage} / {catalog_size} = {coverage_norm:.4f}")

    log_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": "popularity",
        "k": K,
        "test_ratio": TEST_RATIO,
        "threshold": THRESHOLD,
        "users_evaluated": len(users),
        "ndcg": mean_ndcg,
        "recall": mean_recall,
        "coverage": coverage,
        "catalog_size": catalog_size,
        "coverage_norm": coverage_norm,
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "experiments.csv"
    header_needed = not results_path.exists()

    with results_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_results.keys()))
        if header_needed:
            writer.writeheader()
        writer.writerow(log_results)

    print(f"\nLogged run to {results_path}")


if __name__ == "__main__":
    main()
