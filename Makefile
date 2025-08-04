# Colors for pretty output
BLUE := \033[36m
BOLD := \033[1m
GREEN := \033[32m
RESET := \033[0m

-include .env
ifneq (,$(wildcard .env))
    export $(shell sed 's/=.*//' .env)
endif

# Default values if not set in .env
SOURCE_FOLDER ?= chebpy
TESTS_FOLDER ?= tests
MARIMO_FOLDER ?= book/marimo

.DEFAULT_GOAL := help

.PHONY: help uv install fmt lint check test build docs clean

##@ Development Setup

uv: ## Install uv and uvx
	@printf "$(BLUE)Creating virtual environment...$(RESET)\n"
	@curl -LsSf https://astral.sh/uv/install.sh | sh

install: uv ## Install all dependencies using uv
	@printf "$(BLUE)Installing dependencies...$(RESET)\n"
	@uv venv --clear --python 3.12
	@uv sync --all-extras --frozen

##@ Code Quality

fmt: uv ## Run code formatters only
	@printf "$(BLUE)Running formatters...$(RESET)\n"
	@uvx ruff format $(SOURCE_FOLDER) $(TESTS_FOLDER) docs

lint: uv ## Run linters only
	@printf "$(BLUE)Running linters...$(RESET)\n"
	#@uvx ruff check run --files tests
	@uvx ruff check --unsafe-fixes --fix chebpy
	@uvx ruff check --unsafe-fixes --fix tests
	@uvx ruff check --unsafe-fixes --fix docs
	@uvx ruff check --unsafe-fixes --fix book

check: fmt lint test ## Run all checks (lint and test)
	@printf "$(GREEN)All checks passed!$(RESET)\n"

##@ Help

help: ## Display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Testing

test: install ## Run all tests
	@printf "$(BLUE)Running tests...$(RESET)\n"
	@uv run pytest $(TESTS_FOLDER) --cov=$(SOURCE_FOLDER) --cov-report=term

##@ Building

build: install ## Build the package
	@printf "$(BLUE)Building package...$(RESET)\n"
	@uv pip install hatch
	@uv run hatch build

##@ Documentation

docs: install ## Build documentation
	@printf "$(BLUE)Building documentation...$(RESET)\n"
	@uv pip install pdoc
	@{ \
		uv run pdoc -o pdoc $(SOURCE_FOLDER); \
		if command -v xdg-open >/dev/null 2>&1; then \
			xdg-open "pdoc/index.html"; \
		elif command -v open >/dev/null 2>&1; then \
			open "pdoc/index.html"; \
		else \
			echo "Documentation generated. Open pdoc/index.html manually"; \
		fi; \
	}

##@ Cleanup

clean: ## Clean generated files and directories
	@printf "$(BLUE)Cleaning project...$(RESET)\n"
	@git clean -d -X -f
	@rm -rf dist build *.egg-info .coverage .pytest_cache
	@printf "$(BLUE)Removing local branches with no remote counterpart...$(RESET)\n"
	@git fetch -p
	@git branch -vv | grep ': gone]' | awk '{print $$1}' | xargs -r git branch -D

##@ Marimo

marimo: install ## Start a Marimo server (use FILE=filename.py to specify a file)
	@if [ -z "$(FILE)" ]; then \
		echo "❌ FILE is required. Usage: make marimo FILE=demo.py" >&2; \
		exit 1; \
	fi

	@printf "$(BLUE)Start Marimo server with $(MARIMO_FOLDER)/$(FILE)...$(RESET)\n"
	@uv run marimo edit $(MARIMO_FOLDER)/$(FILE)
