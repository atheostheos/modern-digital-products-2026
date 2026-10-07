"""Build a ``Recommender`` from its config spec."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import assert_never

from pyspark.sql import SparkSession
from replay.models import ALSWrap, AssociationRulesItemRec, ItemKNN

from recsys.models.base import Recommender
from recsys.models.content import ContentBased
from recsys.models.hybrid import Hybrid
from recsys.models.popularity import KeyedPopularity
from recsys.models.replay_models import ReplayModel
from recsys.models.specs import (
    AlsSpec,
    AssociationRulesSpec,
    ContentSpec,
    HybridSpec,
    ItemKnnSpec,
    ModelSpec,
    PopularitySpec,
    UserKnnSpec,
)
from recsys.models.user_knn import UserKnn


@dataclass(frozen=True)
class BuildContext:
    """Everything models need besides their own spec."""

    spark: SparkSession
    seed: int  # fixes ALS / SVD randomness
    threads: int  # thread cap for implicit / sklearn / BLAS (Spark is capped by its session)
    catalog: Mapping[str, ModelSpec] = field(default_factory=dict)  # for hybrid components


def build_model(name: str, spec: ModelSpec, ctx: BuildContext) -> Recommender:
    """Instantiate the (unfitted) model described by ``spec``."""
    match spec:
        case PopularitySpec():
            return KeyedPopularity(name, ctx.spark, key=spec.key, window_days=spec.window_days)
        case AssociationRulesSpec() | ItemKnnSpec() | AlsSpec():
            return ReplayModel(name, lambda: _replay_model(spec, ctx.seed), ctx.spark)
        case UserKnnSpec():
            return UserKnn(
                name,
                num_neighbours=spec.num_neighbours,
                similarity=spec.similarity,
                num_threads=ctx.threads,
            )
        case ContentSpec():
            return ContentBased(
                name,
                max_features=spec.max_features,
                min_df=spec.min_df,
                svd_components=spec.svd_components,
                profile_items=spec.profile_items,
                seed=ctx.seed,
                threads=ctx.threads,
            )
        case HybridSpec():
            missing = [part for part in spec.components if part not in ctx.catalog]
            if missing:
                raise ValueError(f"{name}: hybrid components {missing} are not in the catalog")
            components = [
                (build_model(part, ctx.catalog[part], ctx), weight)
                for part, weight in spec.components.items()
            ]
            return Hybrid(name, components, depth=spec.depth, rrf_k=spec.rrf_k)
        case _:  # pragma: no cover - exhaustive over ModelSpec
            assert_never(spec)


def _replay_model(
    spec: AssociationRulesSpec | ItemKnnSpec | AlsSpec, seed: int
) -> AssociationRulesItemRec | ItemKNN | ALSWrap:
    """A fresh RePlay estimator for the personalized RePlay-backed kinds."""
    match spec:
        case AssociationRulesSpec():
            return AssociationRulesItemRec(
                session_column="user_id",  # co-reviewed by the same user
                min_item_count=spec.min_item_count,
                min_pair_count=spec.min_pair_count,
                num_neighbours=spec.num_neighbours,
                similarity_metric=spec.similarity_metric,
            )
        case ItemKnnSpec():
            return ItemKNN(
                num_neighbours=spec.num_neighbours, shrink=spec.shrink, weighting=spec.weighting
            )
        case AlsSpec():
            return ALSWrap(rank=spec.rank, implicit_prefs=True, seed=seed)
        case _:  # pragma: no cover - exhaustive
            assert_never(spec)
