from typer.testing import CliRunner

from recsys.cli import app

runner = CliRunner()


def test_show_config_prints_resolved_default_config():
    result = runner.invoke(app, ["show-config"])

    assert result.exit_code == 0
    assert "Musical_Instruments" in result.output
    assert "data/raw" in result.output


def test_show_config_fails_cleanly_on_missing_file(tmp_path):
    result = runner.invoke(app, ["show-config", "--config", str(tmp_path / "nope.yaml")])

    assert result.exit_code == 1
    assert "not found" in result.output
