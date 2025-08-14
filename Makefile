# Colors for pretty output
BLUE := \033[36m
BOLD := \033[1m
GREEN := \033[32m
RESET := \033[0m

.DEFAULT_GOAL := help

.PHONY: build clean book check

install: ## install
	task build:install

clean: ## clean
	task cleanup:clean

test: install ## run all tests
	task docs:test

book: test ## compile the companion book
	task docs:docs
	task docs:marimushka
	task docs:book

check: install ## check the pre-commit hooks, the linting and deptry
	task quality:check

help: ## Display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
