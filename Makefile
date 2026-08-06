UV_CACHE = /sgoinfre/.uv_cache

.PHONY: install run clean lint debug

install:
	UV_CACHE_DIR=$(UV_CACHE) uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb main.py

clean:
	rm -rf src/__pycache__

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

