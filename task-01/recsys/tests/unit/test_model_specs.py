import pytest
from pydantic import TypeAdapter, ValidationError

from recsys.models.specs import (
    AssociationRulesSpec,
    ItemKnnSpec,
    ModelSpec,
    PopularitySpec,
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
