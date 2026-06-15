PYTHON := .venv/bin/python
PIP := .venv/bin/pip
VENV_PYTHON := $(shell for p in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do \
	if $$p -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then \
		echo $$p; exit 0; \
	fi; \
done)

.PHONY: help install dev test lint format run-lesson clean

help:
	@echo "make install   - create venv and install package + dev tools"
	@echo "make test      - run pytest"
	@echo "make lint      - run ruff and pyright"
	@echo "make format    - run ruff format"
	@echo "make run-lesson - run lessons/01_prompt.py"
	@echo "make clean     - remove build artifacts"

install:
	@test -n "$(VENV_PYTHON)" || (echo "需要 Python 3.10+，请先安装: brew install python@3.12" && exit 1)
	test -d .venv || $(VENV_PYTHON) -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel
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
