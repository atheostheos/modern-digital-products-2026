SAMPLE ?=          # e.g. SAMPLE=20000; off by default: hits are rare (~1%), small samples are noise
TRAIN_DAYS ?= 730

.PHONY: help setup lint format test check clean data prepare eda evaluate-val evaluate-test sweep analyze bench all

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup:  ## install locked dependencies (incl. dev)
	uv sync --locked

lint:  ## ruff lint + format check
	uv run ruff check src tests
	uv run ruff format --check src tests

format:  ## auto-fix lint issues and format
	uv run ruff check --fix src tests
	uv run ruff format src tests

test:  ## run all tests with coverage (fails under 80%)
	uv run pytest

check: lint test  ## lint + tests

data:  ## download raw dataset (~2.2 GB, cached)
	uv run recsys download

prepare:  ## raw JSONL -> processed splits + results/data/dataset_stats.json
	uv run recsys prepare

eda:  ## execute notebooks/01_eda.ipynb in place (figures -> results/figures/)
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

evaluate-val:  ## fit on train, evaluate all models on val -> results/metrics/val.*
	uv run recsys evaluate --stage val

evaluate-test:  ## fit on train+val, evaluate all models on test (+ saved lists) -> results/metrics/test.*
	nice -n 10 uv run recsys evaluate --stage test --save-recs

bench:  ## serving latency per engine (+ precomputed lookup) -> results/benchmark/latency.*
	nice -n 10 uv run recsys bench

analyze:  ## business / fairness / cold-start / quality tables from saved test lists -> results/analysis/
	uv run recsys analyze --stage test

sweep:  ## one cheap sweep, low CPU priority: make sweep NAME=item_knn [TRAIN_DAYS=730 SAMPLE=]
	@test -n "$(NAME)" || (echo "usage: make sweep NAME=<configs/sweeps/NAME.yaml>"; exit 1)
	nice -n 10 uv run recsys evaluate --stage val --models configs/sweeps/$(NAME).yaml \
		--output sweep_$(NAME) --train-days $(TRAIN_DAYS) $(if $(SAMPLE),--sample-users $(SAMPLE))

all: data prepare eda evaluate-test analyze bench  ## full pipeline: data -> final results (~15 min, 4 cores)

clean:  ## remove caches (keeps data/ and results/)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
