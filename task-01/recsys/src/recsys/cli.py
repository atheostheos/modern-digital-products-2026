"""Command-line entry points: ``uv run recsys <command>``."""

from pathlib import Path
from typing import Annotated

import typer
import yaml

from recsys.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, Config, ConfigError, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Path to a YAML config.")]


def _load_or_exit(path: Path) -> Config:
    try:
        return load_config(path)
    except ConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.callback()
def main() -> None:
    """Recommender systems study on Amazon Reviews 2023."""


@app.command("show-config")
def show_config(config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    """Print the validated config with paths resolved against the project root."""
    cfg = _load_or_exit(config)
    resolved = cfg.model_copy(update={"data": cfg.data.resolve(PROJECT_ROOT)})
    typer.echo(yaml.safe_dump(resolved.model_dump(mode="json"), sort_keys=False))
