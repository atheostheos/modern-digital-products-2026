from datetime import datetime

import polars as pl
import pytest

from recsys.models.base import TrainData
from recsys.models.content import ContentBased

ITEMS = pl.DataFrame(
    {
        "item_id": ["e_guitar", "a_guitar", "kit", "sticks", "blank"],
        "text": [
            "electric guitar maple neck pickups",
            "acoustic guitar spruce top neck",
            "drum kit snare cymbal",
            "drum sticks hickory snare",
            None,
        ],
        "brand": ["Fender", "Fender", "Pearl", "Vic Firth", None],
    }
)


def train(rows: list[tuple[str, str, int]]) -> TrainData:
    return TrainData(
        interactions=pl.DataFrame(
            {
                "user_id": [u for u, _, _ in rows],
                "item_id": [i for _, i, _ in rows],
                "timestamp": [datetime(2022, 1, d) for _, _, d in rows],
            }
        ),
        items=ITEMS,
    )


def model(profile_items: int | None = None) -> ContentBased:
    return ContentBased(
        "content",
        max_features=100,
        min_df=1,
        svd_components=3,
        profile_items=profile_items,
        seed=0,
        threads=1,
    )


def test_recommends_textually_similar_unseen_items():
    m = model().fit(train([("u1", "e_guitar", 1)]))

    recs = m.recommend(pl.Series(["u1"]), k=1)

    assert recs["item_id"].to_list() == ["a_guitar"]
    assert recs.columns == ["user_id", "item_id", "score"]


def test_profile_window_uses_only_the_most_recent_items():
    rows = [("u1", "e_guitar", 1), ("u1", "kit", 2)]

    recent = model(profile_items=1).fit(train(rows)).recommend(pl.Series(["u1"]), k=1)

    assert recent["item_id"].to_list() == ["sticks"]  # profile = drum kit only


def test_can_recommend_items_nobody_interacted_with():
    m = model().fit(train([("u1", "kit", 1)]))

    recs = m.recommend(pl.Series(["u1"]), k=3)

    assert "sticks" in recs["item_id"].to_list()  # cold item, reachable via its text
    assert "kit" not in recs["item_id"].to_list()


def test_cold_users_are_not_served_and_scores_are_sorted():
    m = model().fit(train([("u1", "e_guitar", 1)]))

    recs = m.recommend(pl.Series(["u1", "cold"]), k=4)

    assert set(recs["user_id"]) == {"u1"}
    assert recs["score"].is_sorted(descending=True)


def test_recommend_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        model().recommend(pl.Series(["u1"]), k=1)


def test_only_unknown_users_give_empty_frame():
    recs = model().fit(train([("u1", "kit", 1)])).recommend(pl.Series(["x"]), k=2)

    assert recs.is_empty()


def test_heavy_users_still_get_k_recommendations():
    many = pl.DataFrame(
        {
            "item_id": [f"g{i:03d}" for i in range(300)],
            "text": [f"guitar string set gauge {i}" for i in range(300)],
            "brand": ["Ernie Ball"] * 300,
        }
    )
    items = pl.concat([ITEMS, many])
    seen = [("heavy", f"g{i:03d}", 1) for i in range(250)]  # more than any fixed buffer
    data = TrainData(
        interactions=pl.DataFrame(
            {
                "user_id": [u for u, _, _ in seen],
                "item_id": [i for _, i, _ in seen],
                "timestamp": [datetime(2022, 1, 1)] * len(seen),
            }
        ),
        items=items,
    )

    recs = model().fit(data).recommend(pl.Series(["heavy"]), k=10)

    assert recs.height == 10


def test_profile_window_breaks_timestamp_ties_by_item_id():
    rows = [("u1", "e_guitar", 1), ("u1", "kit", 1)]  # same timestamp

    first = model(profile_items=1).fit(train(rows)).recommend(pl.Series(["u1"]), k=1)
    second = model(profile_items=1).fit(train(rows)).recommend(pl.Series(["u1"]), k=1)

    assert first["item_id"].to_list() == second["item_id"].to_list() == ["sticks"]  # kit > e_guitar


def test_users_whose_items_have_no_text_get_no_recommendations():
    m = model().fit(train([("u1", "blank", 1), ("u2", "kit", 1)]))

    recs = m.recommend(pl.Series(["u1", "u2"]), k=2)

    assert set(recs["user_id"]) == {"u2"}


def test_duplicate_interactions_do_not_double_weight_the_profile():
    once = model().fit(train([("u1", "e_guitar", 1), ("u1", "kit", 2)]))
    twice = model().fit(train([("u1", "e_guitar", 1), ("u1", "e_guitar", 3), ("u1", "kit", 2)]))

    a = once.recommend(pl.Series(["u1"]), k=2)
    b = twice.recommend(pl.Series(["u1"]), k=2)

    assert a["item_id"].to_list() == b["item_id"].to_list()
    assert a["score"].to_list() == pytest.approx(b["score"].to_list())


def test_vocabulary_too_small_for_svd_is_rejected():
    tiny = TrainData(
        interactions=pl.DataFrame(
            {"user_id": ["u"], "item_id": ["x"], "timestamp": [datetime(2022, 1, 1)]}
        ),
        items=pl.DataFrame(
            {"item_id": ["x", "y"], "text": ["guitar", "guitar"], "brand": [None, None]},
            schema_overrides={"brand": pl.String},
        ),
    )

    with pytest.raises(ValueError, match="vocabulary"):
        model().fit(tiny)


def test_only_textless_histories_give_empty_frame():
    recs = model().fit(train([("u1", "blank", 1)])).recommend(pl.Series(["u1"]), k=2)

    assert recs.is_empty()


def test_interactions_on_items_without_metadata_are_ignored_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        m = model().fit(train([("u1", "kit", 1), ("u1", "unknown_item", 2)]))

    assert "1 interactions on items without metadata ignored" in caplog.text
    assert m.recommend(pl.Series(["u1"]), k=1)["item_id"].to_list() == ["sticks"]
