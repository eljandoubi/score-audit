.PHONY: build lint format test data report clean

build:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

data:
	uv run scoreaudit download

report:
	uv run scoreaudit report

clean:
	rm -rf .venv .pytest_cache .ruff_cache
