import numpy as np
import polars as pl
from scipy import sparse

from recsys.models.sparse import InteractionMatrix, top_k_from_scores

FRAME = pl.DataFrame({"user_id": ["u2", "u1", "u1"], "item_id": ["b", "a", "b"]})


def test_matrix_is_binary_users_by_items_with_sorted_ids():
    m = InteractionMatrix.from_interactions(FRAME)

    assert m.users.to_list() == ["u1", "u2"]
    assert m.items.to_list() == ["a", "b"]
    assert m.matrix.toarray().tolist() == [[1.0, 1.0], [0.0, 1.0]]


def test_user_indices_skip_unknown_users():
    m = InteractionMatrix.from_interactions(FRAME)

    found = m.user_indices(pl.Series(["u2", "ghost"]))

    assert found["user_id"].to_list() == ["u2"]
    assert found["row"].to_list() == [1]


def test_duplicate_pairs_do_not_inflate_weights():
    dup = pl.concat([FRAME, FRAME])

    assert InteractionMatrix.from_interactions(dup).matrix.max() == 1.0


def test_top_k_from_scores_skips_empty_rows_and_breaks_ties_by_item_order():
    scores = sparse.csr_matrix(np.array([[0.0, 0.0, 0.0], [0.5, 0.9, 0.5]], dtype=np.float32))

    recs = top_k_from_scores(scores, pl.Series(["u1", "u2"]), pl.Series(["a", "b", "c"]), k=2)

    assert recs["user_id"].to_list() == ["u2", "u2"]
    assert recs["item_id"].to_list() == ["b", "a"]  # a beats c on the tie


def test_top_k_from_scores_on_all_empty_matrix_returns_empty_frame():
    scores = sparse.csr_matrix((2, 3), dtype=np.float32)

    recs = top_k_from_scores(scores, pl.Series(["u1", "u2"]), pl.Series(["a", "b", "c"]), k=2)

    assert recs.is_empty() and recs.columns == ["user_id", "item_id", "score"]
