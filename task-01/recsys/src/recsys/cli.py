"""Command-line entry points: ``uv run recsys <command>``."""

import logging
from pathlib import Path
from typing import Annotated

import typer
import yaml

from recsys.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, Config, ConfigError, load_config
from recsys.data.download import DownloadError, download_dataset
from recsys.data.ingest import IngestError
from recsys.data.pipeline import prepare as run_prepare
from recsys.data.split import SplitError
from recsys.evaluation.runner import (
    Stage,
    UnknownModelError,
    load_stage,
    run_experiment,
    validate_models,
    write_results,
)
from recsys.spark import open_spark

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

app = typer.Typer(no_args_is_help=True, add_completion=False)

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Path to a YAML config.")]
RootOption = Annotated[
    Path, typer.Option("--root", help="Directory that relative config paths resolve against.")
]


def _load_or_exit(path: Path, root: Path = PROJECT_ROOT) -> Config:
    try:
        return load_config(path).resolve(root)
    except ConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.callback()
def main() -> None:
    """Recommender systems study on Amazon Reviews 2023."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


@app.command("show-config")
def show_config(config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Print the validated config with paths resolved against the project root."""
    resolved = _load_or_exit(config)
    typer.echo(yaml.safe_dump(resolved.model_dump(mode="json"), sort_keys=False))


@app.command()
def download(config: ConfigOption = DEFAULT_CONFIG_PATH, root: RootOption = PROJECT_ROOT) -> None:
    """Download raw reviews + metadata JSONL for the configured category."""
    cfg = _load_or_exit(config, root)
    try:
        paths = download_dataset(cfg.data)
    except DownloadError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"reviews: {paths.reviews}\nmeta:    {paths.meta}")


@app.command()
def prepare(config: ConfigOption = DEFAULT_CONFIG_PATH, root: RootOption = PROJECT_ROOT) -> None:
    """Build processed train/val/test splits, the item table and dataset stats."""
    cfg = _load_or_exit(config, root)
    try:
        result = run_prepare(cfg)
    except IngestError as exc:
        typer.echo(f"Error: {exc}\nRun `make data` first if the raw files are missing.", err=True)
        raise typer.Exit(code=1) from exc
    except SplitError as exc:
        typer.echo(f"Error: {exc}\nCheck the split dates in the config.", err=True)
        raise typer.Exit(code=1) from exc
    splits = result.stats["splits"]
    typer.echo(
        f"train={result.splits.train.height:,} val={result.splits.val.height:,} "
        f"test={result.splits.test.height:,} items={result.items.height:,} | "
        f"test users with history: {splits['test']['users_with_history_1plus_share']:.1%}"
    )


@app.command("evaluate")
def evaluate_models(
    stage: Annotated[
        Stage, typer.Option(help="val = tune on Q4 2022, test = final Q1 2023")
    ] = "val",
    model: Annotated[
        list[str] | None, typer.Option("--model", "-m", help="Model name(s); default: all")
    ] = None,
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    root: RootOption = PROJECT_ROOT,
) -> None:
    """Fit models on the stage's train data, evaluate, and write results/metrics/<stage>.*"""
    cfg = _load_or_exit(config, root)
    names = list(dict.fromkeys(model or cfg.models))  # dedupe, keep order
    try:  # validate everything cheap before starting Spark
        validate_models(cfg, names)
        data = load_stage(cfg, stage)
    except UnknownModelError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}\nRun `make prepare` first.", err=True)
        raise typer.Exit(code=1) from exc
    with open_spark(cfg.spark) as spark:
        table = run_experiment(cfg, data, names, spark)
    paths = write_results(table, cfg.results_dir, stage)
    typer.echo(f"{len(names)} model(s) evaluated on {stage}: {paths.markdown}")
