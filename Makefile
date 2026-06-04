.PHONY: setup lint test run ingest ask eval docker fmt

setup:
	uv sync --extra eval

fmt:
	uv run ruff format src tests eval
	uv run ruff check --fix src tests eval

lint:
	uv run ruff check src tests eval
	uv run ruff format --check src tests eval
	uv run mypy src

test:
	uv run pytest

ingest:
	uv run python -m grounded_rag.ingest

run:
	uv run uvicorn grounded_rag.api:app --host 0.0.0.0 --port 8000

ask:
	uv run python -m grounded_rag.ask "$(Q)"

eval:
	uv run python eval/run_eval.py

docker:
	docker compose up --build
