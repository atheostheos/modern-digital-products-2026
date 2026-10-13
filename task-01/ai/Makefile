LLM_MODEL ?= gemma3:1b
NODES ?=           # e.g. NODES=sentiment,llm; empty = all five nodes
THREADS ?=         # CPU threads for torch / CTranslate2; empty = server default
DURATION ?= 40     # harness run length, s
HOST ?= 127.0.0.1
PORT ?= 8000

UV_SERVER := uv run --project server
SERVER_URL := http://$(HOST):$(PORT)

.PHONY: help setup llm media lint format test test-models check ollama server server-offline health metrics \
	harness harness-live bench-transport td clean

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

setup:  ## install locked server dependencies (incl. dev, Python 3.12)
	cd server && uv sync --locked

llm:  ## pull the local LLM into Ollama (~0.8 GB, once)
	ollama pull $(LLM_MODEL)

media:  ## recorded test media -> media/ (gesture video, spoken sentences)
	$(UV_SERVER) python scripts/make_media.py

lint:  ## ruff lint + format check (server, harness, scripts)
	cd server && uv run ruff check src tests ../harness ../scripts
	cd server && uv run ruff format --check src tests ../harness ../scripts

format:  ## auto-fix lint issues and format
	cd server && uv run ruff check --fix src tests ../harness ../scripts
	cd server && uv run ruff format src tests ../harness ../scripts

test:  ## fast test suite with coverage (no models)
	cd server && uv run pytest --cov

test-models:  ## real-model tests (cached weights + running Ollama)
	cd server && uv run pytest -m models

check: lint test  ## lint + tests

ollama:  ## run the Ollama server (foreground; skip if the Ollama app runs)
	OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 ollama serve

server:  ## run the ML server (foreground): make server [NODES=sentiment,llm THREADS=2]
	cd server && uv run tdml-server --host $(HOST) --port $(PORT) --llm-model $(LLM_MODEL) \
		$(if $(strip $(NODES)),--nodes $(strip $(NODES))) $(if $(strip $(THREADS)),--threads $(strip $(THREADS)))

server-offline:  ## same as server, without Hugging Face network checks (weights already cached)
	HF_HUB_OFFLINE=1 $(MAKE) server

health:  ## which nodes are loaded
	@curl -fsS $(SERVER_URL)/health && echo

metrics:  ## per-node latency, fps, drops + server CPU/RAM
	@curl -fsS $(SERVER_URL)/metrics && echo

harness:  ## stream recorded media like TD and score it: make harness [DURATION=40]
	$(UV_SERVER) python harness/stream_media.py --server $(SERVER_URL) --duration $(strip $(DURATION))

harness-live:  ## same with webcam + microphone
	$(UV_SERVER) python harness/stream_media.py --server $(SERVER_URL) --duration $(strip $(DURATION)) --live

bench-transport:  ## TD -> server JPEG echo server for the transport benchmark (port 8765)
	$(UV_SERVER) python bench/transport_echo_server.py

td:  ## open the TouchDesigner project
	open ai.toe

clean:  ## remove caches (keeps media/, models/ and the venv)
	rm -rf server/.pytest_cache server/.ruff_cache server/.coverage .ruff_cache
	find . -name __pycache__ -type d -not -path "./server/.venv/*" -prune -exec rm -rf {} +
