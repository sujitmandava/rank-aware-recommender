from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


@dataclass
class MFModel:
    """
    Matrix Factorization model (explicit ratings, SGD).

    user_factors: shape (num_users, k)
    item_factors: shape (num_items, k)
    user_index: mapping user_id -> row index
    item_index: mapping item_id -> row index
    index_item: reverse mapping row index -> item_id
    """
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_index: Dict[int, int]
    item_index: Dict[int, int]
    index_item: Dict[int, int]


def fit(
    train_df: pd.DataFrame,
    k: int = 32,
    lr: float = 0.01,
    reg: float = 0.05,
    epochs: int = 20,
    seed: int = 42,
) -> MFModel:
    """
    Fit MF using SGD on explicit ratings.

    Args:
        train_df: columns [user_id, item_id, rating]
        k: latent dimension
        lr: learning rate
        reg: L2 regularization
        epochs: number of SGD passes
    """
    rng = np.random.default_rng(seed)

    users = train_df["user_id"].unique()
    items = train_df["item_id"].unique()

    user_index = {u: i for i, u in enumerate(users)}
    item_index = {i: j for j, i in enumerate(items)}
    index_item = {j: i for i, j in item_index.items()}

    n_users = len(users)
    n_items = len(items)

    # Initialize latent factors
    U = rng.normal(0, 0.1, size=(n_users, k))
    V = rng.normal(0, 0.1, size=(n_items, k))

    # SGD training
    for _ in range(epochs):
        shuffled = train_df.sample(frac=1.0, random_state=seed)
        for row in shuffled.itertuples(index=False):
            u = user_index[int(row.user_id)]
            i = item_index[int(row.item_id)]
            r = float(row.rating)

            pred = np.dot(U[u], V[i])
            err = r - pred

            # gradients
            U[u] += lr * (err * V[i] - reg * U[u])
            V[i] += lr * (err * U[u] - reg * V[i])

    return MFModel(
        user_factors=U,
        item_factors=V,
        user_index=user_index,
        item_index=item_index,
        index_item=index_item,
    )


def recommend(
    model: MFModel,
    user_id: int,
    k: int,
    exclude_items: Set[int] | None = None,
) -> List[int]:
    """
    Recommend top-K items for a user using MF scores.

    Args:
        model: trained MFModel
        user_id: user to recommend for
        k: number of items
        exclude_items: items to filter (seen items)

    Returns:
        Ranked list of item_ids
    """
    if exclude_items is None:
        exclude_items = set()

    if user_id not in model.user_index:
        return []

    u_idx = model.user_index[user_id]
    scores = model.item_factors @ model.user_factors[u_idx]

    # Rank all items
    ranked_indices = np.argsort(scores)[::-1]

    recs: List[int] = []
    for idx in ranked_indices:
        item_id = model.index_item[idx]
        if item_id not in exclude_items:
            recs.append(item_id)
        if len(recs) == k:
            break

    return recs
