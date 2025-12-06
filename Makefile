## Makefile for config-templates: developer tasks orchestrated via go-task
#
# This Makefile wraps the Taskfile.yml commands and provides a friendly
# `make help` index. Lines with `##` after a target are parsed into help text,
# and lines starting with `##@` create section headers in the help output.
#
# Colors for pretty output in help messages
BLUE := \033[36m
BOLD := \033[1m
GREEN := \033[32m
RED := \033[31m
RESET := \033[0m

# Default goal when running `make` with no target
.DEFAULT_GOAL := help

# Declare phony targets (they don't produce files)
.PHONY: install-task install clean test marimo book fmt deptry help all

##@ Bootstrap
install-task: ## ensure go-task (Taskfile) is installed
	@if ! command -v task >/dev/null 2>&1; then \
		printf "$(BLUE)Installing go-task (Taskfile)$(RESET)\n"; \
		mkdir -p ~/.local/bin; \
		curl --location https://taskfile.dev/install.sh | sh -s -- -d -b ~/.local/bin; \
		printf "$(YELLOW)Note: Add ~/.local/bin to your PATH if not already done$(RESET)\n"; \
	fi


UV_BIN := uv
UVX_BIN := uvx
UV_NO_MODIFY_PATH := 1
SCRIPTS_FOLDER := .github/scripts
CUSTOM_SCRIPTS_FOLDER := .github/scripts/customisations

export UV_BIN
export UVX_BIN
export UV_NO_MODIFY_PATH

install: ## install dependencies
	@printf "$(BLUE)Installing dependencies...$(RESET)\n"
	@$(UV_BIN) venv --python 3.12
	@if [ -f pyproject.toml ]; then \
		$(UV_BIN) sync --all-extras; \
	fi
	@if [ -x "$(CUSTOM_SCRIPTS_FOLDER)/build-extras.sh" ]; then \
		$(CUSTOM_SCRIPTS_FOLDER)/build-extras.sh; \
	fi

clean: ## clean generated files
	@printf "$(BLUE)Cleaning generated files...$(RESET)\n"
	@rm -rf _tests _pdoc _marimushka _book .pytest_cache .coverage htmlcov
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

##@ Development and Testing
test: ## run all tests
	@printf "$(BLUE)Running tests...$(RESET)\n"
	@mkdir -p _tests/html-coverage _tests/html-report
	@$(UV_BIN) run pytest tests --cov=src --cov-report=term --cov-report=html:_tests/html-coverage --html=_tests/html-report/report.html

marimo: ## fire up Marimo server
	@printf "$(BLUE)Starting Marimo server...$(RESET)\n"
	@$(UV_BIN) run marimo edit book/marimo

##@ Documentation
book: ## compile the companion book
	@$(MAKE) test
	@printf "$(BLUE)Building API docs...$(RESET)\n"
	@$(UV_BIN) run pdoc -o _pdoc src/chebpy
	@printf "$(BLUE)Exporting notebooks...$(RESET)\n"
	@$(SCRIPTS_FOLDER)/marimushka.sh
	@printf "$(BLUE)Assembling book...$(RESET)\n"
	@$(SCRIPTS_FOLDER)/book.sh
	@printf "$(BLUE)Rendering book with minibook...$(RESET)\n"
	@$(UVX_BIN) minibook --title "ChebPy" --subtitle "Documentation and Reports" --output _book _book/links.json

fmt: ## check the pre-commit hooks and the linting
	@printf "$(BLUE)Running pre-commit hooks...$(RESET)\n"
	@$(UVX_BIN) pre-commit run --all-files

deptry: ## run deptry if pyproject.toml exists
	@if [ -f pyproject.toml ]; then \
		printf "$(BLUE)Running deptry...$(RESET)\n"; \
		$(UVX_BIN) deptry src; \
	else \
		printf "$(YELLOW) ⚠ Skipping deptry (no pyproject.toml)$(RESET)\n"; \
	fi

all: fmt deptry test book ## Run everything
	echo "Run fmt, deptry, test and book"

##@ Meta
help: ## Display this help message
	+@printf "$(BOLD)Usage:$(RESET)\n"
	+@printf "  make $(BLUE)<target>$(RESET)\n\n"
	+@printf "$(BOLD)Targets:$(RESET)\n"
	+@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)

# Debug targets for tests
print-%:
	@echo '$* = $($*)'
