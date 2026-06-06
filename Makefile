.PHONY: help install lint typecheck test merge bench plots clean

BASE ?= sshleifer/tiny-gpt2
PARENTS ?= sshleifer/tiny-gpt2,sshleifer/tiny-gpt2
METHODS ?= linear,slerp,ties,dare,model_stock

help:
	@echo "make install                       - deps"
	@echo "make lint / typecheck / test       - quality gates"
	@echo "make merge                         - run a synthetic merge sweep"
	@echo "make plots                         - regenerate the 5 chart types"

install: ; uv sync --all-extras
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
typecheck: ; uv run mypy src
test: ; uv run pytest -m "not slow"
merge: ; uv run merge sweep --methods $(METHODS)
plots: ; uv run merge plots
clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
