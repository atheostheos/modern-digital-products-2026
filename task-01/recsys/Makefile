.PHONY: help setup lint format test check clean data prepare eda

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

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

clean:  ## remove caches (keeps data/ and results/)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
