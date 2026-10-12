"""Typed model specifications (the ``models:`` section of the config), keyed by ``kind``."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, PositiveInt


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


class AlsSpec(_Spec):
    """Implicit-feedback matrix factorization (ALS from the ``implicit`` library)."""

    kind: Literal["als"]
    rank: PositiveInt = 64  # latent factors per user / item
    iterations: PositiveInt = 15
    regularization: PositiveFloat = 0.01
    alpha: PositiveFloat = 1.0  # confidence weight of observed interactions


class UserKnnSpec(_Spec):
    """User-based collaborative filtering (implicit nearest neighbours on users)."""

    kind: Literal["user_knn"]
    num_neighbours: PositiveInt = 50
    similarity: Literal["cosine", "bm25"] = "cosine"


class ContentSpec(_Spec):
    """Content-based: TF-IDF of item text + brand, reduced with TruncatedSVD (LSA);
    user profile = mean vector of the user's (last ``profile_items``) items."""

    kind: Literal["content"]
    max_features: PositiveInt = 50_000
    min_df: PositiveInt = 3
    svd_components: PositiveInt = 256
    profile_items: PositiveInt | None = None  # None = whole history


class HybridSpec(_Spec):
    """Weighted reciprocal rank fusion of other configured (non-hybrid) models."""

    kind: Literal["hybrid"]
    components: dict[str, PositiveFloat] = Field(min_length=1)  # model name -> weight
    depth: PositiveInt = 100  # how many recommendations to take from each component
    rrf_k: PositiveInt = 60


ModelSpec = Annotated[
    PopularitySpec
    | AssociationRulesSpec
    | ItemKnnSpec
    | AlsSpec
    | UserKnnSpec
    | ContentSpec
    | HybridSpec,
    Field(discriminator="kind"),
]


def hybrid_reference_errors(models: dict[str, "ModelSpec"]) -> list[str]:
    """Problems with hybrid component references (unknown names, nested hybrids)."""
    errors = []
    for name, spec in models.items():
        if not isinstance(spec, HybridSpec):
            continue
        for component in spec.components:
            if component not in models:
                errors.append(f"hybrid '{name}' references unknown model '{component}'")
            elif isinstance(models[component], HybridSpec):
                errors.append(f"hybrid '{name}' cannot nest hybrid '{component}'")
    return errors
