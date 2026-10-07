"""String-id interactions <-> scipy sparse matrices, shared by the numpy/scipy-based models."""

from dataclasses import dataclass
from typing import Self

import numpy as np
import polars as pl
from scipy import sparse

from recsys.models.base import REC_SCHEMA, empty_recs


@dataclass(frozen=True)
class InteractionMatrix:
    users: pl.Series  # row index -> user_id (sorted)
    items: pl.Series  # column index -> item_id (sorted)
    matrix: sparse.csr_matrix  # binary users x items, float32

    @classmethod
    def from_interactions(cls, interactions: pl.DataFrame, items: pl.Series | None = None) -> Self:
        """Binary matrix; ``items`` fixes the column universe (default: interacted items)."""
        pairs = interactions.select("user_id", "item_id").unique()
        users = pairs["user_id"].unique().sort()
        columns = (items if items is not None else pairs["item_id"]).unique().sort()
        indexed = pairs.join(_index(users, "user_id", "row"), on="user_id").join(
            _index(columns, "item_id", "col"), on="item_id"
        )
        matrix = sparse.csr_matrix(
            (
                np.ones(indexed.height, dtype=np.float32),
                (indexed["row"].to_numpy(), indexed["col"].to_numpy()),
            ),
            shape=(users.len(), columns.len()),
        )
        return cls(users=users, items=columns, matrix=matrix)

    def user_indices(self, users: pl.Series) -> pl.DataFrame:
        """``user_id, row`` for the requested users known to the matrix."""
        return (
            users.unique()
            .to_frame("user_id")
            .join(_index(self.users, "user_id", "row"), on="user_id")
        )


def top_k_from_scores(
    scores: sparse.csr_matrix, user_ids: pl.Series, items: pl.Series, k: int
) -> pl.DataFrame:
    """Top-k non-zero scores per row (ties -> lower item index, i.e. item_id order)."""
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for r in range(scores.shape[0]):
        start, end = scores.indptr[r], scores.indptr[r + 1]
        if start == end:
            continue
        idx, data = scores.indices[start:end], scores.data[start:end]
        order = np.lexsort((idx, -data))[:k]
        rows.append(np.full(order.size, r))
        cols.append(idx[order])
        vals.append(data[order])
    if not rows:
        return empty_recs()
    row_idx, col_idx = np.concatenate(rows), np.concatenate(cols)
    return pl.DataFrame(
        {
            "user_id": user_ids.to_numpy()[row_idx],
            "item_id": items.to_numpy()[col_idx],
            "score": np.concatenate(vals).astype(np.float64),
        },
        schema=REC_SCHEMA,
    )


def _index(values: pl.Series, name: str, index_name: str) -> pl.DataFrame:
    return pl.DataFrame({name: values, index_name: np.arange(values.len())})
