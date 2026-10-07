"""Build a ``Recommender`` from its config spec."""

from typing import assert_never

from pyspark.sql import SparkSession
from replay.models import AssociationRulesItemRec, ItemKNN

from recsys.models.base import Recommender
from recsys.models.popularity import KeyedPopularity
from recsys.models.replay_models import ReplayModel
from recsys.models.specs import AssociationRulesSpec, ItemKnnSpec, ModelSpec, PopularitySpec


def build_model(name: str, spec: ModelSpec, spark: SparkSession) -> Recommender:
    """Instantiate the (unfitted) model described by ``spec``."""
    match spec:
        case PopularitySpec():
            return KeyedPopularity(name, spark, key=spec.key, window_days=spec.window_days)
        case AssociationRulesSpec():
            return ReplayModel(
                name,
                lambda: AssociationRulesItemRec(
                    session_column="user_id",  # co-reviewed by the same user
                    min_item_count=spec.min_item_count,
                    min_pair_count=spec.min_pair_count,
                    num_neighbours=spec.num_neighbours,
                    similarity_metric=spec.similarity_metric,
                ),
                spark,
            )
        case ItemKnnSpec():
            return ReplayModel(
                name,
                lambda: ItemKNN(
                    num_neighbours=spec.num_neighbours,
                    shrink=spec.shrink,
                    weighting=spec.weighting,
                ),
                spark,
            )
        case _:  # pragma: no cover - exhaustive over ModelSpec
            assert_never(spec)
