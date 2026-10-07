from contextlib import contextmanager

import polars as pl
import pytest
from typer.testing import CliRunner

from recsys import cli
from recsys.cli import app
from recsys.config import PROJECT_ROOT
from recsys.data.download import DatasetPaths, DownloadError

runner = CliRunner()


def test_show_config_prints_resolved_default_config():
    result = runner.invoke(app, ["show-config"])

    assert result.exit_code == 0
    assert "Musical_Instruments" in result.output
    assert "data/interim" in result.output
    assert "test_start: '2023-01-01'" in result.output


def test_show_config_fails_cleanly_on_missing_file(tmp_path):
    result = runner.invoke(app, ["show-config", "--config", str(tmp_path / "nope.yaml")])

    assert result.exit_code == 1
    assert "not found" in result.output


def test_download_command_reports_downloaded_paths(monkeypatch, tmp_path):

    fake = DatasetPaths(reviews=tmp_path / "r.jsonl", meta=tmp_path / "m.jsonl")
    seen = []
    monkeypatch.setattr(cli, "download_dataset", lambda data: seen.append(data) or fake)

    result = runner.invoke(app, ["download", "--root", str(tmp_path)])

    assert result.exit_code == 0
    assert "r.jsonl" in result.output
    assert "m.jsonl" in result.output
    assert seen[0].download_dir == tmp_path / "data"


def test_download_command_exits_nonzero_on_failure(monkeypatch):

    def boom(data):
        raise DownloadError("network down")

    monkeypatch.setattr(cli, "download_dataset", boom)

    result = runner.invoke(app, ["download"])

    assert result.exit_code == 1
    assert "network down" in result.output


def test_prepare_command_runs_full_pipeline(toy_project):
    result = runner.invoke(
        app, ["prepare", "--config", str(toy_project / "config.yaml"), "--root", str(toy_project)]
    )

    assert result.exit_code == 0, result.output
    assert "train=2 val=1 test=2 items=3" in result.output
    assert (toy_project / "results/data/dataset_stats.json").is_file()


def test_prepare_command_hints_to_download_when_raw_missing(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text((PROJECT_ROOT / "configs/default.yaml").read_text())

    result = runner.invoke(app, ["prepare", "--config", str(cfg), "--root", str(tmp_path)])

    assert result.exit_code == 1
    assert "make data" in result.output


def test_prepare_command_reports_empty_split_windows(toy_project):
    cfg = toy_project / "config.yaml"
    cfg.write_text(
        cfg.read_text()
        .replace("2022-10-01", "2030-01-01")
        .replace("2023-01-01", "2030-02-01")
        .replace("2023-04-01", "2030-03-01")
    )

    result = runner.invoke(app, ["prepare", "--config", str(cfg), "--root", str(toy_project)])

    assert result.exit_code == 1
    assert "split dates" in result.output


@pytest.fixture
def shared_spark(monkeypatch, spark):
    """Make the CLI reuse the test session instead of creating and stopping its own."""

    @contextmanager
    def reuse(cfg):
        yield spark

    monkeypatch.setattr(cli, "open_spark", reuse)


@pytest.mark.spark
def test_evaluate_command_writes_metrics_for_selected_models(toy_project, shared_spark):
    args = ["--config", str(toy_project / "config.yaml"), "--root", str(toy_project)]
    assert runner.invoke(app, ["prepare", *args]).exit_code == 0

    result = runner.invoke(
        app, ["evaluate", "--stage", "test", "-m", "popular", "-m", "popular", *args]
    )

    assert result.exit_code == 0, result.output
    table = pl.read_parquet(toy_project / "results/metrics/test.parquet")
    assert table.filter(pl.col("segment") == "all")["model"].to_list() == ["popular"]  # deduped
    assert (toy_project / "results/metrics/test.md").is_file()


def test_evaluate_command_rejects_unknown_model_before_starting_spark(toy_project, monkeypatch):
    def no_spark(cfg):
        raise AssertionError("Spark must not start for invalid input")

    monkeypatch.setattr(cli, "open_spark", no_spark)
    args = ["--config", str(toy_project / "config.yaml"), "--root", str(toy_project)]

    result = runner.invoke(app, ["evaluate", "-m", "nope", *args])

    assert result.exit_code == 1
    assert "Unknown model" in result.output


def test_evaluate_command_hints_to_prepare_when_processed_data_missing(toy_project, monkeypatch):
    monkeypatch.setattr(cli, "open_spark", lambda cfg: (_ for _ in ()).throw(AssertionError))
    args = ["--config", str(toy_project / "config.yaml"), "--root", str(toy_project)]

    result = runner.invoke(app, ["evaluate", "-m", "popular", *args])

    assert result.exit_code == 1
    assert "make prepare" in result.output


def test_evaluate_command_rejects_unknown_stage(toy_project):
    result = runner.invoke(
        app, ["evaluate", "--stage", "prod", "--config", str(toy_project / "config.yaml")]
    )

    assert result.exit_code != 0


@pytest.mark.spark
def test_evaluate_command_uses_models_file_and_output_name(toy_project, shared_spark):
    args = ["--config", str(toy_project / "config.yaml"), "--root", str(toy_project)]
    assert runner.invoke(app, ["prepare", *args]).exit_code == 0
    sweep = toy_project / "sweep.yaml"
    sweep.write_text("pop_a: {kind: popularity}\npop_b: {kind: popularity, window_days: 30}\n")

    result = runner.invoke(
        app, ["evaluate", "--models", str(sweep), "--output", "val_sweep", *args]
    )

    assert result.exit_code == 0, result.output
    table = pl.read_parquet(toy_project / "results/metrics/val_sweep.parquet")
    assert sorted(table["model"].unique()) == ["pop_a", "pop_b"]
    assert not (toy_project / "results/metrics/val.parquet").exists()


@pytest.mark.spark
def test_evaluate_command_cheap_trial_records_its_budget(toy_project, shared_spark):
    args = ["--config", str(toy_project / "config.yaml"), "--root", str(toy_project)]
    assert runner.invoke(app, ["prepare", *args]).exit_code == 0

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--stage",
            "test",
            "-m",
            "popular",
            "--sample-users",
            "1",
            "--train-days",
            "400",
            "--output",
            "cheap",
            *args,
        ],
    )

    assert result.exit_code == 0, result.output
    table = pl.read_parquet(toy_project / "results/metrics/cheap.parquet")
    overall = table.filter(pl.col("segment") == "all")
    assert overall["users"].item() == 1
    assert overall["sample_users"].item() == 1 and overall["train_days"].item() == 400


def test_evaluate_command_rejects_non_positive_budget(toy_project):
    result = runner.invoke(
        app, ["evaluate", "--sample-users", "0", "--config", str(toy_project / "config.yaml")]
    )

    assert result.exit_code != 0


@pytest.mark.spark
def test_evaluate_command_can_save_top_k_recommendations(toy_project, shared_spark):
    args = ["--config", str(toy_project / "config.yaml"), "--root", str(toy_project)]
    assert runner.invoke(app, ["prepare", *args]).exit_code == 0

    result = runner.invoke(
        app, ["evaluate", "--stage", "test", "-m", "popular", "--save-recs", *args]
    )

    assert result.exit_code == 0, result.output
    recs = pl.read_parquet(toy_project / "data/recs/test/popular.parquet")
    assert recs.columns == ["user_id", "item_id", "score"]
    assert recs.group_by("user_id").len()["len"].max() <= 2  # toy config: max k = 2
