# Colors for pretty output
BLUE := \033[36m
BOLD := \033[1m
GREEN := \033[32m
RESET := \033[0m

.DEFAULT_GOAL := help

.PHONY: build clean book check

install: ## install
	task build:install -s

clean: ## clean
	task cleanup:clean -s

test: install ## run all tests
	task docs:test -s

book: test ## compile the companion book
	task docs:docs -s
	task docs:marimushka -s
	task docs:book -s

check: install ## check the pre-commit hooks, the linting and deptry
	task quality:check -s

help: ## Display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
