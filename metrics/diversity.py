from __future__ import annotations

from typing import Dict, Sequence, Set


def coverage_at_k(
    recs_by_user: Dict[int, Sequence[int]],
    k: int = 10,
) -> int:
    """
    Item Coverage@K.

    Returns:
        Number of unique items appearing in users' top-K lists.
    """
    if k <= 0 or not recs_by_user:
        return 0

    unique_items: Set[int] = set()
    for items in recs_by_user.values():
        unique_items.update(items[:k])

    return len(unique_items)
