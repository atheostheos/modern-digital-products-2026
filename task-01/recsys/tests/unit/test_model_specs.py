import pytest
from pydantic import TypeAdapter, ValidationError

from recsys.models.specs import (
    AlsSpec,
    AssociationRulesSpec,
    ContentSpec,
    ItemKnnSpec,
    ModelSpec,
    PopularitySpec,
    UserKnnSpec,
)

SPECS = TypeAdapter(dict[str, ModelSpec])


def test_specs_are_parsed_by_kind():
    specs = SPECS.validate_python(
        {
            "popular": {"kind": "popularity"},
            "trending": {"kind": "popularity", "window_days": 90},
            "category_popular": {"kind": "popularity", "key": "category"},
            "also_bought": {"kind": "association_rules", "min_item_count": 5, "min_pair_count": 3},
            "item_knn": {"kind": "item_knn", "num_neighbours": 50, "shrink": 10.0},
        }
    )

    assert specs["trending"] == PopularitySpec(kind="popularity", window_days=90)
    assert isinstance(specs["also_bought"], AssociationRulesSpec)
    assert specs["item_knn"] == ItemKnnSpec(kind="item_knn", num_neighbours=50, shrink=10.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "unknown"},
        {"kind": "popularity", "key": "price"},  # only category / brand
        {"kind": "popularity", "window_days": 0},
        {"kind": "item_knn", "num_neighbours": 0},
        {"kind": "item_knn", "typo": 1},
    ],
)
def test_invalid_specs_are_rejected(bad):
    with pytest.raises(ValidationError):
        SPECS.validate_python({"m": bad})


def test_phase4_specs_have_sensible_defaults():
    specs = SPECS.validate_python(
        {
            "als": {"kind": "als"},
            "user_knn": {"kind": "user_knn", "similarity": "bm25"},
            "content": {"kind": "content", "profile_items": 5},
        }
    )

    assert specs["als"] == AlsSpec(kind="als", rank=64)
    assert specs["user_knn"] == UserKnnSpec(kind="user_knn", num_neighbours=50, similarity="bm25")
    assert specs["content"].profile_items == 5
    assert isinstance(specs["content"], ContentSpec)
    assert specs["content"].svd_components == 256


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "als", "rank": 0},
        {"kind": "user_knn", "similarity": "jaccard"},
        {"kind": "content", "min_df": 0},
        {"kind": "content", "profile_items": 0},
    ],
)
def test_invalid_phase4_specs_are_rejected(bad):
    with pytest.raises(ValidationError):
        SPECS.validate_python({"m": bad})


@pytest.mark.parametrize(
    "bad",
    [
        {"kind": "hybrid", "components": {}},
        {"kind": "hybrid", "components": {"a": 0.0}},
        {"kind": "hybrid", "components": {"a": 1.0}, "depth": 0},
    ],
)
def test_invalid_hybrid_specs_are_rejected(bad):
    with pytest.raises(ValidationError):
        SPECS.validate_python({"m": bad})
