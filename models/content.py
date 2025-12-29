from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set
import numpy as np
import pandas as pd


@dataclass
class ContentModel:
    """
    Content-based recommendation model using item features.

    item_features: shape (num_items, d)
    item_index: mapping item_id -> row index
    index_item: reverse mapping row index -> item_id
    user_profiles: mapping user_id -> preference vector
    """
    item_features: np.ndarray
    item_index: Dict[int, int]
    index_item: Dict[int, int]
    user_profiles: Dict[int, np.ndarray]


def _build_item_features(items_df: pd.DataFrame) -> tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    """
    Build item feature matrix from MovieLens genre flags.

    Expects columns:
      item_id, title, unknown, Action, Adventure, ...
    """
    # Genre columns start at index 2 onward in MovieLens u.item
    genre_cols = items_df.columns[2:]
    features = items_df[genre_cols].values.astype(float)

    # Normalize item vectors to unit length (cosine similarity)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features = features / norms

    item_ids = items_df["item_id"].tolist()
    item_index = {iid: idx for idx, iid in enumerate(item_ids)}
    index_item = {idx: iid for iid, idx in item_index.items()}

    return features, item_index, index_item


def _build_user_profiles(
    train_df: pd.DataFrame,
    item_features: np.ndarray,
    item_index: Dict[int, int],
) -> Dict[int, np.ndarray]:
    """
    User profile = mean of item feature vectors for positively rated items.
    """
    user_profiles: Dict[int, np.ndarray] = {}

    for user_id, df_u in train_df.groupby("user_id"):
        liked = df_u[df_u["rating"] >= 4]["item_id"]
        if liked.empty:
            continue

        vecs = np.array([item_features[item_index[i]] for i in liked if i in item_index])
        if len(vecs) == 0:
            continue

        profile = vecs.mean(axis=0)
        profile /= np.linalg.norm(profile) or 1.0
        user_profiles[int(user_id)] = profile

    return user_profiles


def fit(
    train_df: pd.DataFrame,
    items_df: pd.DataFrame,
) -> ContentModel:
    """
    Fit content-based model.

    Args:
        train_df: columns [user_id, item_id, rating]
        items_df: item metadata with genre one-hot columns

    Returns:
        ContentModel
    """
    item_features, item_index, index_item = _build_item_features(items_df)
    user_profiles = _build_user_profiles(train_df, item_features, item_index)

    return ContentModel(
        item_features=item_features,
        item_index=item_index,
        index_item=index_item,
        user_profiles=user_profiles,
    )


def recommend(
    model: ContentModel,
    user_id: int,
    k: int,
    exclude_items: Set[int] | None = None,
) -> List[int]:
    """
    Recommend top-K items using cosine similarity between
    user profile and item features.
    """
    if exclude_items is None:
        exclude_items = set()

    if user_id not in model.user_profiles:
        return []

    user_vec = model.user_profiles[user_id]

    scores = model.item_features @ user_vec
    ranked_indices = np.argsort(scores)[::-1]

    recs: List[int] = []
    for idx in ranked_indices:
        item_id = model.index_item[idx]
        if item_id not in exclude_items:
            recs.append(item_id)
        if len(recs) == k:
            break

    return recs
