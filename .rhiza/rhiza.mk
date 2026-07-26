## Makefile for jebel-quant/rhiza
# (https://github.com/jebel-quant/rhiza)
#
# Purpose: Developer tasks using uv/uvx (install, test, book).
# Lines with `##` after a target are parsed into help text,
# and lines starting with `##@` create section headers in the help output.
#
# Require GNU Make (MAKE_VERSION is unset in BSD make)
ifndef MAKE_VERSION
$(error GNU Make is required. macOS ships BSD make — install GNU Make with: brew install make)
endif

# Require a working POSIX shell.
# Recipes use POSIX shell syntax (mkdir -p, printf, [ ... ], curl | sh). GNU Make
# runs recipes with sh when it finds one on PATH -- even when make itself was
# launched from PowerShell or cmd -- so e.g. CI on windows-latest (which ships
# Git's sh.exe) works fine. Only when no POSIX shell is found does make fall back
# to cmd.exe, where the recipes fail on the first non-cmd line with errors like:
#   process_begin: CreateProcess(NULL, # Ensure ... folder exists, ...) failed.
# So probe the shell make actually uses rather than guessing from the OS: run a
# POSIX command and require its output. Guarded by $(OS) so the probe only runs
# on Windows (no subprocess cost on Linux/macOS/WSL, where the shell is POSIX).
ifeq ($(OS),Windows_NT)
ifneq ($(shell printf POSIX),POSIX)
define RHIZA_WINDOWS_SHELL_ERROR

  This project's Makefile requires a POSIX shell, but make could not find one
  and fell back to cmd.exe, which is not supported.

  Run 'make' from an environment that provides a POSIX shell, e.g.:
    - WSL (recommended):  https://learn.microsoft.com/windows/wsl/install
    - Git Bash:           bundled with Git for Windows (https://git-scm.com/download/win)

  Tip: if 'sh.exe' (from Git for Windows) is on your PATH, make will use it
  automatically even from PowerShell or cmd.

  See README.md > Prerequisites for details.

endef
$(error $(RHIZA_WINDOWS_SHELL_ERROR))
endif
endif

# Colours for pretty output in help messages
BLUE := \033[36m
BOLD := \033[1m
GREEN := \033[32m
RED := \033[31m
YELLOW := \033[33m
RESET := \033[0m

# Default goal when running `make` with no target
.DEFAULT_GOAL := help

# Declare phony targets (they don't produce files)
.PHONY: \
	help \
	post-bump \
	post-install \
	post-release \
	post-sync \
	post-validate \
	pre-bump \
	pre-install \
	pre-release \
	pre-sync \
	pre-validate \
	print-logo \
	readme \
	summarise-sync \
	sync \
	validate \
	version-matrix \
	ci-os-matrix

# we need absolute paths!
INSTALL_DIR ?= $(abspath ./bin)
UV_BIN ?= $(shell command -v uv 2>/dev/null || echo ${INSTALL_DIR}/uv)
UVX_BIN ?= $(shell command -v uvx 2>/dev/null || echo ${INSTALL_DIR}/uvx)
VENV ?= .venv

# Read Python version from .python-version (single source of truth)
PYTHON_VERSION ?= $(strip $(shell cat .python-version 2>/dev/null || echo "3.13"))
export PYTHON_VERSION

# Read Rhiza version from .rhiza/.rhiza-version (single source of truth for rhiza-tools)
RHIZA_VERSION ?= $(shell cat .rhiza/.rhiza-version 2>/dev/null || echo "0.10.2")
export RHIZA_VERSION

# Default sync schedule (cron expression for GitHub Actions sync workflow)
# Override in your root Makefile to customise when sync runs.
# Example: RHIZA_SYNC_SCHEDULE = 0 9 * * 1-5  (weekdays at 9 AM UTC)
RHIZA_SYNC_SCHEDULE ?= 0 0 * * 1

export UV_NO_MODIFY_PATH := 1
export UV_VENV_CLEAR := 1

# Unset VIRTUAL_ENV to prevent uv from warning about path mismatches
# when a virtual environment is already activated in the shell
unexport VIRTUAL_ENV

# Load .rhiza/.env (if present) and export its variables so recipes see them.
# This file is optional — sensible defaults are defined below.
-include .rhiza/.env

# ---------------------------------------------------------------------------
# Default values for variables that may be set in .rhiza/.env.
# These ?= assignments are skipped when the variable is already defined by
# the included file, by an environment variable, or by the root Makefile.
# ---------------------------------------------------------------------------

# Directory that holds the project's Python source package(s).
SOURCE_FOLDER ?= src

# Directory that holds Marimo notebooks (used by marimo.mk and book.mk).
MARIMO_FOLDER ?= docs/notebooks

# JSON array of GitHub Actions runner OS labels used by the CI matrix.
# Override in .rhiza/.env or your root Makefile to add more platforms.
RHIZA_CI_OS_MATRIX ?= ["ubuntu-latest"]

# ==============================================================================
# Rhiza Core
# ==============================================================================

# RHIZA_LOGO definition
define RHIZA_LOGO
  ____  _     _
 |  _ \| |__ (_)______ _
 | |_) | '_ \| |_  / _\`|
 |  _ <| | | | |/ / (_| |
 |_| \_\_| |_|_/___\__,_|

endef
export RHIZA_LOGO

# Declare phony targets for Rhiza Core
.PHONY: print-logo sync sync-experimental materialize validate readme pre-sync post-sync pre-validate post-validate _apply-sync-schedule

# Hook targets (double-colon rules allow multiple definitions)
# Note: pre-install/post-install are defined in bootstrap.mk
# Note: pre-bump/post-bump/pre-release/post-release are defined in releasing.mk
pre-sync:: ; @:
post-sync:: ; @:
pre-validate:: ; @:
post-validate:: ; @:

# Detected once at parse time: non-empty when origin is the jebel-quant/rhiza
# mother repository, which by design has no template.yml. The sync/summarise/
# validate targets are no-ops there. Evaluated a single time and reused so the
# origin regex lives in exactly one place.
IS_MOTHER_REPO := $(shell git remote get-url origin 2>/dev/null | grep -iqE 'jebel-quant/rhiza(\.git)?$$' && echo 1)

##@ Rhiza Workflows

print-logo:
	@printf "${BLUE}$$RHIZA_LOGO${RESET}\n"


sync: pre-sync ## sync with template repository as defined in .rhiza/template.yml
	@if [ -n "$(IS_MOTHER_REPO)" ]; then \
		printf "${BLUE}[INFO] Skipping sync in rhiza repository (no template.yml by design)${RESET}\n"; \
	else \
		$(MAKE) install-uv && \
		${UVX_BIN} "rhiza==$(RHIZA_VERSION)" sync . && \
		$(MAKE) _apply-sync-schedule; \
	fi
	@$(MAKE) post-sync

_apply-sync-schedule: ## (internal) apply RHIZA_SYNC_SCHEDULE override to GitHub Actions sync workflow
	@if [ "$(RHIZA_SYNC_SCHEDULE)" != "0 0 * * 1" ] && [ -f .github/workflows/rhiza_sync.yml ]; then \
		sed -i.bak "s|cron: '[^']*'|cron: '$(RHIZA_SYNC_SCHEDULE)'|" .github/workflows/rhiza_sync.yml && rm -f .github/workflows/rhiza_sync.yml.bak; \
		printf "${BLUE}[INFO] Applied custom sync schedule: $(RHIZA_SYNC_SCHEDULE)${RESET}\n"; \
	fi

materialize: ## [DEPRECATED] use 'make sync' instead — materialize --force is now sync
	@printf "${YELLOW}[WARN] 'make materialize' is deprecated and will be removed in a future release.${RESET}\n"
	@printf "${YELLOW}[WARN] Please use 'make sync' instead (e.g. 'materialize --force' is now 'make sync').${RESET}\n"
	@$(MAKE) sync

summarise-sync: install-uv ## summarise differences created by sync with template repository
	@if [ -n "$(IS_MOTHER_REPO)" ]; then \
		printf "${BLUE}[INFO] Skipping summarise-sync in rhiza repository (no template.yml by design)${RESET}\n"; \
	else \
		$(MAKE) install-uv; \
		${UVX_BIN} "rhiza==$(RHIZA_VERSION)" summarise .; \
	fi

rhiza-test: install ## run rhiza's own tests (if any)
	@if [ -d ".rhiza/tests" ]; then \
		${UV_BIN} run --with pytest --with pytest-timeout --with python-dotenv --with packaging pytest .rhiza/tests; \
	else \
		printf "${YELLOW}[WARN] No .rhiza/tests directory found, skipping rhiza-tests${RESET}\n"; \
	fi

validate: pre-validate rhiza-test ## validate project structure against template repository as defined in .rhiza/template.yml
	@if [ -n "$(IS_MOTHER_REPO)" ]; then \
		printf "${BLUE}[INFO] Skipping validate in rhiza repository (no template.yml by design)${RESET}\n"; \
	else \
		$(MAKE) install-uv; \
		${UVX_BIN} "rhiza==$(RHIZA_VERSION)" validate .; \
	fi
	@$(MAKE) post-validate

readme: install-uv ## update README.md with current Makefile help output
	@${UVX_BIN} "rhiza-tools>=0.2.0" update-readme

##@ Meta

help: print-logo ## Display this help message
	+@printf "$(BOLD)Usage:$(RESET)\n"
	+@printf "  make $(BLUE)<target>$(RESET)\n\n"
	+@printf "$(BOLD)Targets:$(RESET)\n"
	+@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	+@printf "\n"

version-matrix: install-uv ## Emit the list of supported Python versions from pyproject.toml
	@${UVX_BIN} "rhiza-tools>=0.2.2" version-matrix

ci-os-matrix: ## Emit GitHub CI OSes (RHIZA_CI_OS_MATRIX as JSON array, default ["ubuntu-latest"])
	@$(info $(or $(RHIZA_CI_OS_MATRIX),["ubuntu-latest"]))

print-% : ## print the value of a variable (usage: make print-VARIABLE)
	@printf "${BLUE}[INFO] Printing value of variable '$*':${RESET}\n"
	@printf "${BOLD}Value of $*:${RESET}\n"
	@printf "${GREEN}"
	@printf "%s\n" "$($*)"
	@printf "${RESET}"
	@printf "${BLUE}[INFO] End of value for '$*'${RESET}\n"

# Optional: repo extensions (committed)
-include .rhiza/make.d/*.mk
