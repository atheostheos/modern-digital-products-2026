import polars as pl
import pytest

from recsys.config import load_config, with_models_from
from recsys.data.pipeline import prepare
from recsys.evaluation import runner as runner_module
from recsys.evaluation.runner import UnknownModelError, load_stage, run_experiment
from recsys.models.als import ImplicitAls
from recsys.models.registry import BuildContext, build_model
from recsys.models.specs import (
    AlsSpec,
    AssociationRulesSpec,
    ContentSpec,
    HybridSpec,
    ItemKnnSpec,
    PopularitySpec,
    UserKnnSpec,
)

pytestmark = pytest.mark.spark


@pytest.mark.parametrize(
    "spec",
    [
        PopularitySpec(kind="popularity"),
        PopularitySpec(kind="popularity", key="brand", window_days=30),
        AssociationRulesSpec(kind="association_rules"),
        ItemKnnSpec(kind="item_knn", num_neighbours=3, shrink=1.0, weighting="bm25"),
        AlsSpec(kind="als", rank=4),
        UserKnnSpec(kind="user_knn"),
        ContentSpec(kind="content"),
    ],
)
def test_registry_builds_a_named_recommender_for_every_kind(spark, spec):
    model = build_model("m", spec, BuildContext(spark=spark, seed=0, threads=1))

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


def test_registry_builds_a_hybrid_from_its_configured_components(spark):
    catalog = {"pop": PopularitySpec(kind="popularity")}
    spec = HybridSpec(kind="hybrid", components={"pop": 1.0})

    model = build_model("h", spec, BuildContext(spark=spark, seed=0, threads=1, catalog=catalog))

    assert model.name == "h"


def test_hybrid_with_components_missing_from_catalog_is_rejected(spark):
    spec = HybridSpec(kind="hybrid", components={"ghost": 1.0})

    with pytest.raises(ValueError, match="ghost"):
        build_model("h", spec, BuildContext(spark=spark, seed=0, threads=1))


def test_run_experiment_releases_spark_caches_after_each_model(spark, toy_project):
    cfg = load_config(toy_project / "config.yaml").resolve(toy_project)
    prepare(cfg)

    run_experiment(cfg, load_stage(cfg, "test"), ["popular", "item_knn"], spark)

    assert spark.sparkContext._jsc.getPersistentRDDs().isEmpty()


def test_registry_builds_implicit_als_for_als_specs(spark):
    model = build_model(
        "als", AlsSpec(kind="als", rank=4), BuildContext(spark=spark, seed=0, threads=1)
    )

    assert isinstance(model, ImplicitAls)


HYBRID_MODELS = (
    "popular: {kind: popularity}\n"
    "recent: {kind: popularity, window_days: 400}\n"
    "mix: {kind: hybrid, components: {popular: 1.0, recent: 0.5}, depth: 5}\n"
)


def test_hybrid_reuses_component_runs_and_matches_a_fresh_build(spark, toy_project, monkeypatch):
    models_file = toy_project / "hybrid.yaml"
    models_file.write_text(HYBRID_MODELS)
    cfg = with_models_from(
        load_config(toy_project / "config.yaml").resolve(toy_project), models_file
    )
    prepare(cfg)
    data = load_stage(cfg, "test")
    fresh = run_experiment(cfg, data, ["mix"], spark).drop(
        "fit_seconds", "recommend_seconds", "run_at"
    )
    built: list[str] = []
    original = runner_module.build_model
    monkeypatch.setattr(
        runner_module,
        "build_model",
        lambda name, spec, ctx: built.append(name) or original(name, spec, ctx),
    )

    table = run_experiment(cfg, data, ["mix", "popular", "recent"], spark)

    assert built == ["popular", "recent"]  # the hybrid was fused from cached component runs
    reused = table.filter(pl.col("model") == "mix").drop(
        "fit_seconds", "recommend_seconds", "run_at"
    )
    assert reused.equals(fresh)
    assert table["model"].unique(maintain_order=True).to_list() == ["popular", "recent", "mix"]
