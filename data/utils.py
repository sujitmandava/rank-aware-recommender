from __future__ import annotations
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import urllib.request

ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


@dataclass(frozen=True)
class MovieLens100K:
    ratings: pd.DataFrame  # user_id, item_id, rating, timestamp
    items: pd.DataFrame    # item_id, title


def _download(url: str, dst: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "ranking-eval/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        dst.write_bytes(r.read())


def load_movielens_100k(
    root: str | Path = "data",
    download: bool = True
) -> MovieLens100K:
    """
    Loads MovieLens 100K ratings and item metadata.

    Returns:
        MovieLens100K with:
          ratings: user_id, item_id, rating, timestamp
          items:   item_id, title
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    zip_path = root / "ml-100k.zip"
    extract_dir = root / "ml-100k"

    if not extract_dir.exists():
        if not zip_path.exists():
            if not download:
                raise FileNotFoundError("MovieLens 100K not found and download=False")
            _download(ML_100K_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)

    ratings = pd.read_csv(
        extract_dir / "u.data",
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={
            "user_id": np.int32,
            "item_id": np.int32,
            "rating": np.int16,
            "timestamp": np.int64,
        },
    )

    items = pd.read_csv(
        extract_dir / "u.item",
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1],
        names=["item_id", "title"],
        dtype={"item_id": np.int32, "title": str},
    )

    return MovieLens100K(ratings=ratings, items=items)


def time_split_per_user(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    min_train_items: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered split per user.

    - For each user, holds out the most recent interactions.
    - Guarantees at least `min_train_items` in train when possible.
    - Prevents test-only users.
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0,1)")

    ratings = ratings.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    test_indices = []

    for user_id, df_u in ratings.groupby("user_id", sort=False):
        n = len(df_u)
        if n <= 1:
            continue

        n_test = int(np.ceil(test_ratio * n))
        n_train = n - n_test

        if n_train < min_train_items:
            n_train = min(min_train_items, n - 1)
            n_test = n - n_train

        if n_test > 0:
            test_indices.extend(df_u.index[-n_test:].tolist())

    test = ratings.loc[test_indices].copy()
    train = ratings.drop(index=test_indices).copy()

    # Remove users from test that somehow lost all train data
    valid_users = set(train["user_id"].unique())
    test = test[test["user_id"].isin(valid_users)].copy()

    return train.reset_index(drop=True), test.reset_index(drop=True)


def binarize_relevance(
    ratings: pd.DataFrame,
    threshold: int = 4,
    out_col: str = "relevant",
) -> pd.DataFrame:
    """
    Converts ratings to binary relevance.

    relevant = 1 if rating >= threshold else 0
    """
    df = ratings.copy()
    df[out_col] = (df["rating"].astype(int) >= int(threshold)).astype(np.int8)
    return df
