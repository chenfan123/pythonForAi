PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help install dev test lint format run-lesson clean

help:
	@echo "make install   - create venv and install package + dev tools"
	@echo "make test      - run pytest"
	@echo "make lint      - run ruff and pyright"
	@echo "make format    - run ruff format"
	@echo "make run-lesson - run lessons/01_prompt.py"
	@echo "make clean     - remove build artifacts"

install:
	test -d .venv || $(PYTHON) -m venv .venv
	$(PIP) install -e ".[dev]"

dev: install

test:
	$(PYTHON) -m pytest

lint:
	.venv/bin/ruff check .
	.venv/bin/pyright

format:
	.venv/bin/ruff format .

run-lesson:
	$(PYTHON) lessons/01_prompt.py

clean:
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
