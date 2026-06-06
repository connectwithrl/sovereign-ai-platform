.PHONY: install test lint fmt run seed eval compose-up compose-down clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests

run:
	sovereign-api

seed:
	python scripts/seed.py

eval:
	python scripts/run_eval.py

compose-up:
	docker compose up --build

compose-down:
	docker compose down -v

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info