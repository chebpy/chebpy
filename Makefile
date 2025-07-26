# This file is part of the tschm/.config-templates repository
# (https://github.com/tschm/.config-templates).
#
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
SOURCE_FOLDER ?= src
TESTS_FOLDER ?= tests
MARIMO_FOLDER ?= book/marimo

.DEFAULT_GOAL := help

.PHONY: uv help

##@ Development Setup

uv: ## Install uv and uvx
	@printf "$(BLUE)Creating virtual environment...$(RESET)\n"
	@curl -LsSf https://astral.sh/uv/install.sh | sh

##@ Help

help: ## Display this help message
	@printf "$(BOLD)Usage:$(RESET)\n"
	@printf "  make $(BLUE)<target>$(RESET)\n\n"
	@printf "$(BOLD)Targets:$(RESET)\n"
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)

