.PHONY: install test eval lint

install:
	uv pip install -e ".[dev]"

test:
	pytest -q

eval:
	python -m spotter.eval.run

lint:
	ruff check spotter tests
