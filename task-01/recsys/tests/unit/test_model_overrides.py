from pathlib import Path

import pytest

from recsys.config import ConfigError, load_config, with_models_from


def test_models_file_replaces_the_models_section(toy_project: Path):
    cfg = load_config(toy_project / "config.yaml")
    models_file = toy_project / "sweep.yaml"
    models_file.write_text(
        "knn_k5: {kind: item_knn, num_neighbours: 5}\n"
        "knn_k50: {kind: item_knn, num_neighbours: 50}\n"
    )

    swept = with_models_from(cfg, models_file)

    assert list(swept.models) == ["knn_k5", "knn_k50"]
    assert swept.split == cfg.split
    assert set(cfg.models) == {"popular", "item_knn"}  # original untouched


@pytest.mark.parametrize("text", ["- not a mapping\n", "bad: {kind: nope}\n", "{}\n"])
def test_invalid_models_file_is_rejected(toy_project: Path, text: str):
    cfg = load_config(toy_project / "config.yaml")
    models_file = toy_project / "sweep.yaml"
    models_file.write_text(text)

    with pytest.raises(ConfigError):
        with_models_from(cfg, models_file)


def test_missing_models_file_is_rejected(toy_project: Path):
    cfg = load_config(toy_project / "config.yaml")

    with pytest.raises(ConfigError, match="not found"):
        with_models_from(cfg, toy_project / "missing.yaml")


def test_hybrid_components_must_reference_configured_non_hybrid_models(toy_project: Path):
    cfg = load_config(toy_project / "config.yaml")
    ok = toy_project / "ok.yaml"
    ok.write_text("pop: {kind: popularity}\nhyb: {kind: hybrid, components: {pop: 1.0}}\n")
    unknown = toy_project / "unknown.yaml"
    unknown.write_text("hyb: {kind: hybrid, components: {ghost: 1.0}}\n")
    nested = toy_project / "nested.yaml"
    nested.write_text(
        "pop: {kind: popularity}\n"
        "h1: {kind: hybrid, components: {pop: 1.0}}\n"
        "h2: {kind: hybrid, components: {h1: 1.0}}\n"
    )

    assert with_models_from(cfg, ok).models["hyb"].components == {"pop": 1.0}
    with pytest.raises(ConfigError, match="ghost"):
        with_models_from(cfg, unknown)
    with pytest.raises(ConfigError, match="h1"):
        with_models_from(cfg, nested)
