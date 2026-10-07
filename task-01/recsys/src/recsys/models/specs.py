"""Typed model specifications (the ``models:`` section of the config), keyed by ``kind``."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveInt


class _Spec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PopularitySpec(_Spec):
    """Global (``key=None``) or per-user-category/brand popularity, optionally trending."""

    kind: Literal["popularity"]
    key: Literal["category", "brand"] | None = None
    window_days: PositiveInt | None = None  # popularity over the last N days of train


class AssociationRulesSpec(_Spec):
    """ "Also bought": items co-reviewed by the same users (RePlay AssociationRulesItemRec)."""

    kind: Literal["association_rules"]
    min_item_count: PositiveInt = 5
    min_pair_count: PositiveInt = 5
    num_neighbours: PositiveInt = 1000
    similarity_metric: Literal["confidence", "lift"] = "confidence"


class ItemKnnSpec(_Spec):
    """Item-based collaborative filtering (RePlay ItemKNN)."""

    kind: Literal["item_knn"]
    num_neighbours: PositiveInt = 10
    shrink: NonNegativeFloat = 0.0
    weighting: Literal["tf_idf", "bm25"] | None = None


ModelSpec = Annotated[
    PopularitySpec | AssociationRulesSpec | ItemKnnSpec, Field(discriminator="kind")
]
