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
RESET := \033[0m

# Default goal when running `make` with no target
.DEFAULT_GOAL := help

# Declare phony targets (they don't produce files)
.PHONY: install-task install clean test marimo book fmt deptry help all

##@ Bootstrap
install-task: ## ensure go-task (Taskfile) is installed
	@mkdir -p ./bin;
	export PATH=".bin:$$PATH"
	# install task
	@if command -v ./bin/task >/dev/null 2>&1; then \
		printf "$(GREEN)task is already installed$(RESET)\n"; \
	else \
		printf "$(BLUE)Installing go-task (Taskfile)$(RESET)\n"; \
		sh -c "$$(curl --location https://taskfile.dev/install.sh)" -- -d -b ./bin; \
	fi
	# install uv
	@if [ -x "./bin/uv" ]; then \
		printf "${BLUE}[INFO] uv already present in ./bin, skipping installation${RESET}\n"; \
	else \
		curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="./bin" sh || { printf "${RED}[ERROR] Failed to install uv${RESET}\n"; exit 1; }; \
	fi
	# verify task is installed
	./bin/task --version

install: install-task ## install
	./bin/task build:install

clean: install-task ## clean
	./bin/task cleanup:clean

##@ Development and Testing
test: install ## run all tests
	./bin/task docs:test

marimo: install ## fire up Marimo server
	./bin/task docs:marimo

##@ Documentation
book: test ## compile the companion book
	./bin/task docs:docs
	./bin/task docs:marimushka
	./bin/task docs:book

fmt: install ## check the pre-commit hooks and the linting
	./bin/task quality:fmt
	./bin/task quality:lint

deptry: install  ## check deptry
	./bin/task quality:deptry

all: fmt deptry test book ## Run everything
	echo "Run fmt, deptry, test and book"

##@ Meta
help: ## Display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
