# Apple Silicon：始终用原生 arm64 创建 venv / 安装依赖（避免 x86_64 下 cryptography 无 wheel 需源码编译失败）
ifeq ($(shell sysctl -n hw.optional.arm64 2>/dev/null),1)
  ARCH := arch -arm64
else
  ARCH :=
endif

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
VENV_PYTHON := $(shell for p in python3.12 python3.13 python3.14 python3.11 python3.10 python3; do \
	if $$p -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then \
		echo $$p; exit 0; \
	fi; \
done)

.PHONY: help install dev test lint format run-lesson run-qa clean

help:
	@echo "make install   - create venv and install package + dev tools"
	@echo "make test      - run pytest"
	@echo "make lint      - run ruff and pyright"
	@echo "make format    - run ruff format"
	@echo "make run-lesson - run lessons/01_prompt.py"
	@echo "make run-qa     - run lessons/03-langchain/06-questionAndAnswer.py"
	@echo "make clean     - remove build artifacts"

install:
	@test -n "$(VENV_PYTHON)" || (echo "需要 Python 3.10+，请先安装: brew install python@3.12" && exit 1)
	test -d .venv || $(ARCH) $(VENV_PYTHON) -m venv .venv
	$(ARCH) $(PIP) install --upgrade pip setuptools wheel
	$(ARCH) $(PIP) install -e ".[dev]"

dev: install

test:
	$(PYTHON) -m pytest

lint:
	.venv/bin/ruff check .
	.venv/bin/pyright

format:
	.venv/bin/ruff format .

run-lesson:
	$(ARCH) $(PYTHON) lessons/01_prompt.py

run-qa:
	$(ARCH) $(PYTHON) lessons/03-langchain/06-questionAndAnswer.py

clean:
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
