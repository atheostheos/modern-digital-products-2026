# recsys — Amazon Reviews 2023 (Musical Instruments)

Recommender systems study: classic approaches (content-based, item/user-KNN,
hybrid) and business heuristics, evaluated offline on the
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) `Musical_Instruments`
category.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed automatically).

```bash
make setup     # install locked dependencies
make data      # download raw reviews + metadata (~2.2 GB, cached)
make prepare   # processed splits + results/data/dataset_stats.json
make eda       # execute the EDA notebook -> results/figures/
make check     # lint + tests (coverage >= 80%)
make help      # all targets
uv run recsys --help
```

## Layout

| Path | Purpose |
|---|---|
| `configs/` | YAML experiment configs (`default.yaml`) |
| `src/recsys/` | package: `data/`, `models/`, `evaluation/`, `config.py`, `cli.py` |
| `tests/` | `unit/`, `integration/`, `e2e/` (auto-marked by directory) |
| `notebooks/` | EDA notebook |
| `results/` | generated metrics tables and figures (committed) |
| `data/` | downloaded and processed data (gitignored) |
