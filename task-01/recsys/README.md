# recsys — Amazon Reviews 2023 (Musical Instruments)

Recommender systems study: business heuristics, classic approaches (content-based,
item/user-based CF, matrix factorization) and a hybrid, evaluated offline on the
[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) `Musical_Instruments`
category (3.0M reviews, 1.76M users, 214k items).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed automatically) and
Java 17+ (PySpark, used by the RePlay models).

```bash
make setup     # install locked dependencies
make all       # data -> prepare -> eda -> evaluate-test -> analyze -> bench (~15 min)
make check     # lint + tests (coverage >= 80%)
make help      # all targets
```

Step by step:

| Command | What it does | Output |
|---|---|---|
| `make data` | download raw reviews + metadata (~2.2 GB, cached) | `data/raw/` |
| `make prepare` | typed parquet, dedup, item table, temporal split | `data/processed/`, `results/data/dataset_stats.json` |
| `make eda` | execute the EDA notebook | `notebooks/01_eda.ipynb`, `results/figures/*.png` |
| `make evaluate-val` | fit on train, evaluate on val (Oct–Dec 2022) | `results/metrics/val.*` |
| `make sweep NAME=<grid>` | one hyperparameter grid from `configs/sweeps/` on val | `results/metrics/sweep_<grid>.*` |
| `make evaluate-test` | fit on train + val, evaluate on test (Jan–Mar 2023), save lists | `results/metrics/test.*`, `data/recs/test/` |
| `make analyze` | price / revenue, brand concentration, item cold start, low-rated exposure | `results/analysis/test.*` |
| `make bench` | serving latency per engine + precomputed lookup | `results/benchmark/latency.*` |

Compute is capped for a laptop in `configs/default.yaml` (`spark.cores: 4`,
`driver_memory_gb: 4`); sweeps and heavy targets run under `nice`.

## Models (`configs/default.yaml`)

| Name | Kind | Library |
|---|---|---|
| `popular`, `trending` (365 days) | global popularity | RePlay `PopRec` |
| `category_popular`, `brand_popular` | popularity within the user's main category / brand | RePlay `CatPopRec` |
| `also_bought` | items co-reviewed by the same users | RePlay `AssociationRulesItemRec` |
| `item_knn` | item-based CF | RePlay `ItemKNN` |
| `user_knn` | user-based CF | `implicit` (user-user neighbours) |
| `als` | matrix factorization (implicit-feedback ALS) | `implicit` |
| `content` | TF-IDF of text + brand → LSA; profile = last item | scikit-learn |
| `hybrid` | weighted reciprocal rank fusion; trending = cold-start fallback | own (`models/hybrid.py`) |

Metrics: RePlay `OfflineMetrics` (Precision, Recall, NDCG, MAP, MRR, HitRate, Coverage,
Surprisal, CategoricalDiversity) + own Gini, average recommendation popularity and
long-tail share, reported overall and per user-history segment (cold, 1, 2–4, 5+).

## Layout

| Path | Purpose |
|---|---|
| `configs/` | `default.yaml` (data, split, evaluation, compute caps, models), `sweeps/*.yaml` |
| `src/recsys/data/` | download, ingest, preprocess, split, stats, pipeline |
| `src/recsys/models/` | model contract, adapters (RePlay, implicit, sklearn), hybrid, registry |
| `src/recsys/evaluation/` | metrics, evaluator (segments), runner, analysis, benchmark |
| `src/recsys/cli.py` | `recsys` command (`uv run recsys --help`) |
| `tests/` | `unit/`, `integration/` (Spark), `e2e/` (CLI) — auto-marked by directory |
| `notebooks/` | EDA notebook |
| `results/` | committed, machine-generated tables and figures |
| `data/` | raw / processed data and saved lists (gitignored) |
