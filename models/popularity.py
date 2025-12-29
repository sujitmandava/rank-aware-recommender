from __future__ import annotations
from typing import Dict, List, Set
import pandas as pd


class PopularityModel:
    """
    Global popularity-based candidate generator.

    Scores items by interaction count in the training data.
    Produces a single global ranking shared across users.
    """

    def __init__(self, ranked_items: List[int]):
        self.ranked_items = ranked_items


def fit(train_df: pd.DataFrame) -> PopularityModel:
    """
    Fit popularity model on training interactions.

    Args:
        train_df: DataFrame with at least columns [user_id, item_id]

    Returns:
        PopularityModel with items ranked by popularity (desc).
    """
    popularity = (
        train_df.groupby("item_id")["user_id"]
        .count()
        .sort_values(ascending=False)
    )

    ranked_items = popularity.index.tolist()
    return PopularityModel(ranked_items=ranked_items)


def recommend(
    model: PopularityModel,
    user_id: int,
    k: int,
    exclude_items: Set[int] | None = None,
) -> List[int]:
    """
    Recommend top-K popular items excluding seen items.

    Args:
        model: fitted PopularityModel
        user_id: included for interface consistency (unused)
        k: number of items to recommend
        exclude_items: items to filter out (e.g., seen items)

    Returns:
        List of item_ids (length ≤ k)
    """
    if exclude_items is None:
        exclude_items = set()

    recs: List[int] = []
    for item in model.ranked_items:
        if item not in exclude_items:
            recs.append(item)
        if len(recs) == k:
            break

    return recs
