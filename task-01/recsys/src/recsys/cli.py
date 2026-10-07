"""Command-line entry points: ``uv run recsys <command>``."""

import logging
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
import yaml

from recsys.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    Config,
    ConfigError,
    load_config,
    with_models_from,
)
from recsys.data.download import DownloadError, download_dataset
from recsys.data.ingest import IngestError
from recsys.data.pipeline import prepare as run_prepare
from recsys.data.split import SplitError
from recsys.evaluation.analysis import analyze_stage
from recsys.evaluation.benchmark import BenchSettings, precomputed_row, run_benchmark
from recsys.evaluation.runner import (
    SUMMARY_K,
    Stage,
    TrialBudget,
    UnknownModelError,
    load_stage,
    run_experiment,
    validate_models,
    write_results,
    write_table,
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
    *,
    stage: Annotated[
        Stage, typer.Option(help="val = tune on Q4 2022, test = final Q1 2023")
    ] = "val",
    model: Annotated[
        list[str] | None, typer.Option("--model", "-m", help="Model name(s); default: all")
    ] = None,
    models: Annotated[
        Path | None,
        typer.Option("--models", help="YAML mapping replacing the config's models (sweeps)"),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(help="Results name: results/metrics/<output>.*; default: the stage"),
    ] = None,
    sample_users: Annotated[
        int | None, typer.Option(min=1, help="Cheap trial: score a random sample of users")
    ] = None,
    train_days: Annotated[
        int | None, typer.Option(min=1, help="Cheap trial: fit on the last N days only")
    ] = None,
    save_recs: Annotated[
        bool, typer.Option(help="Also save top-k lists to data/recs/<output>/<model>.parquet")
    ] = False,
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    root: RootOption = PROJECT_ROOT,
) -> None:
    """Fit models on the stage's train data, evaluate, and write results/metrics/<output>.*"""
    cfg = _load_or_exit(config, root)
    budget = TrialBudget(sample_users=sample_users, train_days=train_days)
    try:  # validate everything cheap before starting Spark
        cfg = with_models_from(cfg, models) if models else cfg
        names = list(dict.fromkeys(model or cfg.models))  # dedupe, keep order
        validate_models(cfg, names)
        data = load_stage(cfg, stage, budget)
    except (ConfigError, UnknownModelError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}\nRun `make prepare` first.", err=True)
        raise typer.Exit(code=1) from exc
    recs_dir = cfg.data.processed_dir.parent / "recs" / (output or stage) if save_recs else None
    with open_spark(cfg.spark) as spark:
        table = run_experiment(cfg, data, names, spark, recs_dir=recs_dir)
    budget_columns = {"sample_users": budget.sample_users, "train_days": budget.train_days}
    table = table.with_columns(pl.lit(v, pl.Int64).alias(k) for k, v in budget_columns.items())
    paths = write_results(table, cfg.results_dir, output or stage)
    typer.echo(f"{len(names)} model(s) evaluated on {stage}: {paths.markdown}")


@app.command()
def analyze(
    stage: Annotated[Stage, typer.Option(help="Which saved run to analyze")] = "test",
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    root: RootOption = PROJECT_ROOT,
) -> None:
    """Business / fairness / cold-start / quality analysis of saved recommendation lists."""
    cfg = _load_or_exit(config, root)
    recs_dir = cfg.data.processed_dir.parent / "recs" / stage
    files = sorted(recs_dir.glob("*.parquet"))
    if not files:
        typer.echo(
            f"Error: no saved recommendations in {recs_dir}. "
            f"Run `recsys evaluate --stage {stage} --save-recs` first.",
            err=True,
        )
        raise typer.Exit(code=1)
    order = {name: i for i, name in enumerate(cfg.models)}
    saved = {
        f.stem: pl.read_parquet(f)
        for f in sorted(files, key=lambda f: (order.get(f.stem, len(order)), f.stem))
    }
    table = analyze_stage(load_stage(cfg, stage).window, saved, k=SUMMARY_K)
    paths = write_table(
        table, cfg.results_dir / "analysis" / stage, f"{stage} analysis @{SUMMARY_K}"
    )
    typer.echo(f"{len(saved)} model(s) analyzed: {paths.markdown}")


BENCH_MODELS = ["trending", "also_bought", "user_knn", "als", "content"]  # one per engine


@app.command()
def bench(
    *,
    model: Annotated[
        list[str] | None, typer.Option("--model", "-m", help="Default: one model per engine")
    ] = None,
    requests: Annotated[int, typer.Option(min=1, help="Single-user requests per model")] = 30,
    batch_size: Annotated[int, typer.Option(min=1, help="Users in the batch request")] = 1000,
    train_days: Annotated[int, typer.Option(min=1, help="Fit on the last N days")] = 730,
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    root: RootOption = PROJECT_ROOT,
) -> None:
    """Serving latency: single-request p50/p95, batch cost per user, precomputed lookup."""
    cfg = _load_or_exit(config, root)
    names = list(dict.fromkeys(model or BENCH_MODELS))
    try:
        validate_models(cfg, names)
        data = load_stage(cfg, "val", TrialBudget(train_days=train_days))
    except UnknownModelError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    settings = BenchSettings(requests=requests, batch_size=batch_size, seed=cfg.seed)
    with open_spark(cfg.spark) as spark:
        rows = run_benchmark(cfg, data, names, spark, settings)
    precomputed = cfg.data.processed_dir.parent / "recs" / "test" / "hybrid.parquet"
    if precomputed.is_file():
        rows.append(precomputed_row("hybrid (precomputed lookup)", precomputed, settings))
    title = f"Serving latency ({requests} single requests, fit on the last {train_days} days)"
    paths = write_table(
        pl.DataFrame(rows, strict=False), cfg.results_dir / "benchmark" / "latency", title
    )
    typer.echo(f"{len(rows)} benchmark row(s): {paths.markdown}")
