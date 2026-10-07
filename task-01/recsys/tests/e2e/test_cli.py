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
