import polars as pl
import pytest

from recsys.config import load_config
from recsys.data.pipeline import prepare
from recsys.evaluation.runner import UnknownModelError, load_stage, run_experiment
from recsys.models.registry import build_model
from recsys.models.specs import AssociationRulesSpec, ItemKnnSpec, PopularitySpec

pytestmark = pytest.mark.spark


@pytest.mark.parametrize(
    "spec",
    [
        PopularitySpec(kind="popularity"),
        PopularitySpec(kind="popularity", key="brand", window_days=30),
        AssociationRulesSpec(kind="association_rules"),
        ItemKnnSpec(kind="item_knn", num_neighbours=3, shrink=1.0, weighting="bm25"),
    ],
)
def test_registry_builds_a_named_recommender_for_every_kind(spark, spec):
    model = build_model("m", spec, spark)

    assert model.name == "m"
    assert callable(model.fit) and callable(model.recommend)


def test_run_experiment_evaluates_every_model_on_the_stage(spark, toy_project):
    cfg = load_config(toy_project / "config.yaml").resolve(toy_project)
    prepare(cfg)

    table = run_experiment(cfg, load_stage(cfg, "test"), ["popular", "item_knn"], spark)

    assert set(table["model"]) == {"popular", "item_knn"}
    assert {"segment", "users", "HitRate@1", "fit_seconds", "recommend_seconds"} <= set(
        table.columns
    )
    popular_all = table.filter((pl.col("model") == "popular") & (pl.col("segment") == "all"))
    assert popular_all["users"].item() == 2  # test users u1 (warm) and u3 (cold)


def test_run_experiment_rejects_unknown_model_names(spark, toy_project):
    cfg = load_config(toy_project / "config.yaml").resolve(toy_project)
    prepare(cfg)

    with pytest.raises(UnknownModelError, match="nope"):
        run_experiment(cfg, load_stage(cfg, "val"), ["nope"], spark)
