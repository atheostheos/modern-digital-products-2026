from datetime import date
from pathlib import Path

import pytest

from recsys.config import Config, ConfigError, load_config

VALID_YAML = """
seed: 7
results_dir: results
data:
  hf_repo: McAuley-Lab/Amazon-Reviews-2023
  category: Musical_Instruments
  download_dir: data
  interim_dir: data/interim
  processed_dir: data/processed
split:
  val_start: 2022-10-01
  test_start: 2023-01-01
  test_end: 2023-04-01
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_loads_valid_yaml_into_typed_config(tmp_path):
    config = load_config(write(tmp_path, VALID_YAML))

    assert isinstance(config, Config)
    assert config.seed == 7
    assert config.data.category == "Musical_Instruments"
    assert config.data.download_dir == Path("data")
    assert config.split.test_start == date(2023, 1, 1)


def test_relative_paths_resolve_against_given_root(tmp_path):
    config = load_config(write(tmp_path, VALID_YAML))

    resolved = config.data.resolve(root=tmp_path)

    assert resolved.download_dir == tmp_path / "data"
    assert resolved.interim_dir == tmp_path / "data/interim"
    assert resolved.processed_dir == tmp_path / "data/processed"
    assert config.data.download_dir == Path("data")  # original is not mutated


def test_config_resolve_anchors_data_and_results_paths(tmp_path):
    config = load_config(write(tmp_path, VALID_YAML))

    resolved = config.resolve(root=tmp_path)

    assert resolved.results_dir == tmp_path / "results"
    assert resolved.data.processed_dir == tmp_path / "data/processed"
    assert resolved.split == config.split
    assert config.results_dir == Path("results")


def test_config_is_immutable(tmp_path):
    config = load_config(write(tmp_path, VALID_YAML))

    with pytest.raises(ValueError, match="frozen"):
        config.seed = 1


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="typo_key"):
        load_config(write(tmp_path, VALID_YAML + "typo_key: 1\n"))


def test_invalid_category_name_is_rejected(tmp_path):
    bad = VALID_YAML.replace("Musical_Instruments", "../etc/passwd")

    with pytest.raises(ConfigError, match="category"):
        load_config(write(tmp_path, bad))


def test_non_mapping_yaml_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write(tmp_path, "- just\n- a list\n"))


def test_default_config_file_is_valid():
    project_root = Path(__file__).resolve().parents[2]

    config = load_config(project_root / "configs" / "default.yaml")

    assert config.data.category == "Musical_Instruments"


def test_malformed_yaml_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="YAML"):
        load_config(write(tmp_path, "data: [unclosed\n"))


@pytest.mark.parametrize(
    ("val_start", "test_start", "test_end"),
    [
        ("2023-01-01", "2023-01-01", "2023-04-01"),  # empty validation window
        ("2022-10-01", "2023-04-01", "2023-01-01"),  # test ends before it starts
        ("2023-02-01", "2023-01-01", "2023-04-01"),  # validation after test
    ],
)
def test_split_dates_must_be_strictly_increasing(tmp_path, val_start, test_start, test_end):
    bad = (
        VALID_YAML.replace("2022-10-01", val_start)
        .replace("test_start: 2023-01-01", f"test_start: {test_start}")
        .replace("2023-04-01", test_end)
    )

    with pytest.raises(ConfigError, match="val_start < test_start < test_end"):
        load_config(write(tmp_path, bad))
