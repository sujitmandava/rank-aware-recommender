from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set
import numpy as np
import pandas as pd


@dataclass
class MFModel:
    """
    Matrix Factorization model (explicit ratings, SGD).

    user_factors: shape (num_users, k)
    item_factors: shape (num_items, k)
    user_bias: shape (num_users,)
    item_bias: shape (num_items,)
    global_mean: scalar rating mean
    user_index: mapping user_id -> row index
    item_index: mapping item_id -> row index
    index_item: reverse mapping row index -> item_id
    """
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_bias: np.ndarray
    item_bias: np.ndarray
    global_mean: float
    user_index: Dict[int, int]
    item_index: Dict[int, int]
    index_item: Dict[int, int]


def fit(
    train_df: pd.DataFrame,
    k: int = 64,
    lr: float = 0.01,
    reg: float = 0.02,
    epochs: int = 40,
    seed: int = 42,
) -> MFModel:
    """
    Fit MF using SGD on explicit ratings with bias terms.

    Args:
        train_df: columns [user_id, item_id, rating]
        k: latent dimension
        lr: learning rate
        reg: L2 regularization
        epochs: number of SGD passes
        seed: RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    users = train_df["user_id"].unique()
    items = train_df["item_id"].unique()

    user_index = {u: i for i, u in enumerate(users)}
    item_index = {i: j for j, i in enumerate(items)}
    index_item = {j: i for i, j in item_index.items()}

    n_users = len(users)
    n_items = len(items)

    # Initialize latent factors and biases
    U = rng.normal(0, 0.1, size=(n_users, k))
    V = rng.normal(0, 0.1, size=(n_items, k))
    bu = np.zeros(n_users, dtype=np.float32)
    bi = np.zeros(n_items, dtype=np.float32)
    mu = float(train_df["rating"].mean())

    # Precompute index arrays for faster loops
    u_idx = train_df["user_id"].map(user_index).to_numpy(dtype=np.int64)
    i_idx = train_df["item_id"].map(item_index).to_numpy(dtype=np.int64)
    ratings = train_df["rating"].to_numpy(dtype=np.float32)
    n_obs = len(ratings)

    # SGD training
    for _ in range(epochs):
        perm = rng.permutation(n_obs)
        for idx in perm:
            u = u_idx[idx]
            i = i_idx[idx]
            r = ratings[idx]

            pred = mu + bu[u] + bi[i] + np.dot(U[u], V[i])
            err = r - pred

            # gradients
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])

            U[u] += lr * (err * V[i] - reg * U[u])
            V[i] += lr * (err * U[u] - reg * V[i])

    return MFModel(
        user_factors=U,
        item_factors=V,
        user_bias=bu,
        item_bias=bi,
        global_mean=mu,
        user_index=user_index,
        item_index=item_index,
        index_item=index_item,
    )


def fit_bpr(
    train_df: pd.DataFrame,
    k: int = 64,
    lr: float = 0.05,
    reg: float = 0.01,
    epochs: int = 50,
    n_neg: int = 3,
    seed: int = 42,
) -> MFModel:
    """
    Fit MF with a BPR (pairwise) loss for implicit feedback.

    Args:
        train_df: columns [user_id, item_id] representing positive interactions
        k: latent dimension
        lr: learning rate
        reg: L2 regularization
        epochs: number of passes over interactions
        n_neg: negatives sampled per positive per epoch
        seed: RNG seed
    """
    rng = np.random.default_rng(seed)

    # Deduplicate positives to avoid overweighting repeats
    positives = train_df[["user_id", "item_id"]].drop_duplicates()

    users = positives["user_id"].unique()
    items = positives["item_id"].unique()

    user_index = {u: i for i, u in enumerate(users)}
    item_index = {i: j for j, i in enumerate(items)}
    index_item = {j: i for i, j in item_index.items()}

    n_users = len(users)
    n_items = len(items)

    U = rng.normal(0, 0.1, size=(n_users, k))
    V = rng.normal(0, 0.1, size=(n_items, k))
    bu = np.zeros(n_users, dtype=np.float32)
    bi = np.zeros(n_items, dtype=np.float32)
    mu = 0.0  # not used in BPR scoring

    user_pos_items: Dict[int, np.ndarray] = {}
    for u, df_u in positives.groupby("user_id"):
        user_pos_items[user_index[u]] = df_u["item_id"].map(item_index).to_numpy(dtype=np.int64)

    user_list = list(user_pos_items.keys())
    if not user_list:
        raise ValueError("No positive interactions to train BPR model.")

    for _ in range(epochs):
        for u in user_list:
            pos_items = user_pos_items[u]
            if len(pos_items) == 0:
                continue

            # sample positives for this user
            pos_samples = rng.choice(pos_items, size=len(pos_items), replace=True)
            for pos in pos_samples:
                for _ in range(n_neg):
                    neg = rng.integers(0, n_items)
                    # resample until negative
                    while neg in pos_items:
                        neg = rng.integers(0, n_items)

                    # scores
                    x_ui = bu[u] + bi[pos] + np.dot(U[u], V[pos])
                    x_uj = bu[u] + bi[neg] + np.dot(U[u], V[neg])
                    x_uij = x_ui - x_uj

                    # sigmoid approximation
                    sig = 1.0 / (1.0 + np.exp(-x_uij))
                    grad = 1.0 - sig

                    # updates
                    bu[u] += lr * (grad - reg * bu[u])
                    bi[pos] += lr * (grad - reg * bi[pos])
                    bi[neg] += lr * (-grad - reg * bi[neg])

                    U[u] += lr * (grad * (V[pos] - V[neg]) - reg * U[u])
                    V[pos] += lr * (grad * U[u] - reg * V[pos])
                    V[neg] += lr * (-grad * U[u] - reg * V[neg])

    return MFModel(
        user_factors=U,
        item_factors=V,
        user_bias=bu,
        item_bias=bi,
        global_mean=mu,
        user_index=user_index,
        item_index=item_index,
        index_item=index_item,
    )


def _user_scores(model: MFModel, user_idx: int) -> np.ndarray:
    """Compute raw item scores for a user (includes biases)."""
    scores = model.global_mean + model.item_bias + model.item_factors @ model.user_factors[user_idx]
    scores += model.user_bias[user_idx]
    return scores


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
    scores = _user_scores(model, u_idx)

    if exclude_items:
        for item in exclude_items:
            if item in model.item_index:
                scores[model.item_index[item]] = -np.inf

    # Fast top-k: argpartition then sort that slice
    k = min(k, len(scores))
    if k == 0:
        return []

    topk_idx = np.argpartition(scores, -k)[-k:]
    topk_sorted = topk_idx[np.argsort(scores[topk_idx])[::-1]]

    recs: List[int] = []
    for idx in topk_sorted:
        item_id = model.index_item[idx]
        if np.isfinite(scores[idx]) and item_id not in exclude_items:
            recs.append(item_id)

    return recs
