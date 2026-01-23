## Makefile for jebel-quant/rhiza
# (https://github.com/jebel-quant/rhiza)
#
# Purpose: Developer tasks using uv/uvx (install, test, docs, marimushka, book).
# Lines with `##` after a target are parsed into help text,
# and lines starting with `##@` create section headers in the help output.
#
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
	bump \
	clean \
	deptry \
	fmt \
	mypy \
	help \
	install \
	install-uv \
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
	release \
	sync \
	summarise-sync \
	update-readme \
	validate \
	version-matrix

# we need absolute paths!
INSTALL_DIR ?= $(abspath ./bin)
UV_BIN ?= $(shell command -v uv 2>/dev/null || echo ${INSTALL_DIR}/uv)
UVX_BIN ?= $(shell command -v uvx 2>/dev/null || echo ${INSTALL_DIR}/uvx)
VENV ?= .venv

# Read Python version from .python-version (single source of truth)
PYTHON_VERSION ?= $(shell cat .python-version 2>/dev/null || echo "3.13")
export PYTHON_VERSION

# Read Rhiza version from .rhiza/.rhiza-version (single source of truth for rhiza-tools)
RHIZA_VERSION ?= $(shell cat .rhiza/.rhiza-version 2>/dev/null || echo "0.9.0")
export RHIZA_VERSION

export UV_NO_MODIFY_PATH := 1
export UV_VENV_CLEAR := 1

# Load .rhiza/.env (if present) and export its variables so recipes see them.
-include .rhiza/.env

# Include split Makefiles
-include tests/tests.mk
-include book/book.mk
-include book/marimo/marimo.mk
-include presentation/presentation.mk
-include docker/docker.mk
-include .github/agents/agentic.mk
# .rhiza/rhiza.mk is INLINED below
-include .github/github.mk



# ==============================================================================
# Rhiza Core Actions (formerly .rhiza/rhiza.mk)
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
.PHONY: print-logo sync validate readme pre-sync post-sync pre-validate post-validate

# Hook targets (double-colon rules allow multiple definitions)
pre-sync:: ; @:
post-sync:: ; @:
pre-validate:: ; @:
post-validate:: ; @:
pre-install:: ; @:
post-install:: ; @:
pre-release:: ; @:
post-release:: ; @:
pre-bump:: ; @:
post-bump:: ; @:

##@ Rhiza Workflows

print-logo:
	@printf "${BLUE}$$RHIZA_LOGO${RESET}\n"


sync: pre-sync ## sync with template repository as defined in .rhiza/template.yml
	@if git remote get-url origin 2>/dev/null | grep -iqE 'jebel-quant/rhiza(\.git)?$$'; then \
		printf "${BLUE}[INFO] Skipping sync in rhiza repository (no template.yml by design)${RESET}\n"; \
	else \
		$(MAKE) install-uv; \
		${UVX_BIN} "rhiza>=$(RHIZA_VERSION)" materialize --force .; \
	fi
	@$(MAKE) post-sync

summarise-sync: install-uv ## summarise differences created by sync with template repository
	@if git remote get-url origin 2>/dev/null | grep -iqE 'jebel-quant/rhiza(\.git)?$$'; then \
		printf "${BLUE}[INFO] Skipping summarise-sync in rhiza repository (no template.yml by design)${RESET}\n"; \
	else \
		$(MAKE) install-uv; \
		${UVX_BIN} "rhiza>=$(RHIZA_VERSION)" summarise .; \
	fi

validate: pre-validate ## validate project structure against template repository as defined in .rhiza/template.yml
	@if git remote get-url origin 2>/dev/null | grep -iqE 'jebel-quant/rhiza(\.git)?$$'; then \
		printf "${BLUE}[INFO] Skipping validate in rhiza repository (no template.yml by design)${RESET}\n"; \
	else \
		$(MAKE) install-uv; \
		${UVX_BIN} "rhiza>=$(RHIZA_VERSION)" validate .; \
	fi
	@$(MAKE) post-validate

readme: install-uv ## update README.md with current Makefile help output
	@${UVX_BIN} "rhiza-tools>=0.2.0" update-readme

# ==============================================================================
# End Rhiza Core Actions
# ==============================================================================

##@ Bootstrap
install-uv: ## ensure uv/uvx is installed
	# Ensure the ${INSTALL_DIR} folder exists
	@mkdir -p ${INSTALL_DIR}

	# Install uv/uvx only if they are not already present in PATH or in the install dir
	@if command -v uv >/dev/null 2>&1 && command -v uvx >/dev/null 2>&1; then \
	  :; \
	elif [ -x "${INSTALL_DIR}/uv" ] && [ -x "${INSTALL_DIR}/uvx" ]; then \
	  printf "${BLUE}[INFO] uv and uvx already installed in ${INSTALL_DIR}, skipping.${RESET}\n"; \
	else \
	  printf "${BLUE}[INFO] Installing uv and uvx into ${INSTALL_DIR}...${RESET}\n"; \
	  if ! curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="${INSTALL_DIR}" sh >/dev/null 2>&1; then \
	    printf "${RED}[ERROR] Failed to install uv${RESET}\n"; \
	    exit 1; \
	  fi; \
	fi

install: pre-install install-uv ## install
	# Create the virtual environment only if it doesn't exist
	@if [ ! -d "${VENV}" ]; then \
	  ${UV_BIN} venv $(if $(PYTHON_VERSION),--python $(PYTHON_VERSION)) ${VENV} || { printf "${RED}[ERROR] Failed to create virtual environment${RESET}\n"; exit 1; }; \
	else \
	  printf "${BLUE}[INFO] Using existing virtual environment at ${VENV}, skipping creation${RESET}\n"; \
	fi

	# Install the dependencies from pyproject.toml (if it exists)
	@if [ -f "pyproject.toml" ]; then \
	  if [ -f "uv.lock" ]; then \
	    printf "${BLUE}[INFO] Installing dependencies from lock file${RESET}\n"; \
	    ${UV_BIN} sync --all-extras --all-groups --frozen || { printf "${RED}[ERROR] Failed to install dependencies${RESET}\n"; exit 1; }; \
	  else \
	    printf "${YELLOW}[WARN] uv.lock not found. Generating lock file and installing dependencies...${RESET}\n"; \
	    ${UV_BIN} sync --all-extras || { printf "${RED}[ERROR] Failed to install dependencies${RESET}\n"; exit 1; }; \
	  fi; \
	else \
	  printf "${YELLOW}[WARN] No pyproject.toml found, skipping install${RESET}\n"; \
	fi

	# Install dev dependencies from .rhiza/requirements/*.txt files
	@if [ -d ".rhiza/requirements" ] && ls .rhiza/requirements/*.txt >/dev/null 2>&1; then \
	  for req_file in .rhiza/requirements/*.txt; do \
	    if [ -f "$$req_file" ]; then \
	      printf "${BLUE}[INFO] Installing requirements from $$req_file${RESET}\n"; \
	      ${UV_BIN} pip install -r "$$req_file" || { printf "${RED}[ERROR] Failed to install requirements from $$req_file${RESET}\n"; exit 1; }; \
	    fi; \
	  done; \
	fi

	# Check if there is requirements.txt file in the tests folder (legacy support)
	@if [ -f "tests/requirements.txt" ]; then \
	  printf "${BLUE}[INFO] Installing requirements from tests/requirements.txt${RESET}\n"; \
	  ${UV_BIN} pip install -r tests/requirements.txt || { printf "${RED}[ERROR] Failed to install test requirements${RESET}\n"; exit 1; }; \
	fi
	@$(MAKE) post-install

clean: ## Clean project artifacts and stale local branches
	@printf "%bCleaning project...%b\n" "$(BLUE)" "$(RESET)"

	# Remove ignored files/directories, but keep .env files, tested with futures project
	@git clean -d -X -f \
		-e '!.env' \
		-e '!.env.*'

	# Remove build & test artifacts
	@rm -rf \
		dist \
		build \
		*.egg-info \
		.coverage \
		.pytest_cache \
		.benchmarks

	@printf "%bRemoving local branches with no remote counterpart...%b\n" "$(BLUE)" "$(RESET)"

	@git fetch --prune

	@git branch -vv | awk '/: gone]/{print $$1}' | xargs -r git branch -D

##@ Quality and Formatting
deptry: install-uv ## Run deptry
	@if [ -d ${SOURCE_FOLDER} ]; then \
		$(UVX_BIN) -p ${PYTHON_VERSION} deptry ${SOURCE_FOLDER}; \
	fi

	@if [ -d ${MARIMO_FOLDER} ]; then \
		if [ -d ${SOURCE_FOLDER} ]; then \
			$(UVX_BIN) -p ${PYTHON_VERSION} deptry ${MARIMO_FOLDER} ${SOURCE_FOLDER} --ignore DEP004; \
		else \
		  	$(UVX_BIN) -p ${PYTHON_VERSION} deptry ${MARIMO_FOLDER} --ignore DEP004; \
		fi \
	fi

fmt: install-uv ## check the pre-commit hooks and the linting
	@${UVX_BIN} -p ${PYTHON_VERSION} pre-commit run --all-files

mypy: install-uv ## run mypy analysis
	@if [ -d ${SOURCE_FOLDER} ]; then \
		${UVX_BIN} -p ${PYTHON_VERSION} mypy ${SOURCE_FOLDER} --strict --config-file=pyproject.toml; \
	fi

##@ Releasing and Versioning
bump: pre-bump ## bump version
	@if [ -f "pyproject.toml" ]; then \
		$(MAKE) install; \
		${UVX_BIN} "rhiza[tools]>=0.8.6" tools bump; \
		printf "${BLUE}[INFO] Updating uv.lock file...${RESET}\n"; \
		${UV_BIN} lock; \
	else \
		printf "${YELLOW}[WARN] No pyproject.toml found, skipping bump${RESET}\n"; \
	fi
	@$(MAKE) post-bump

release: pre-release install-uv ## create tag and push to remote with prompts
	@UV_BIN="${UV_BIN}" /bin/sh ".rhiza/scripts/release.sh"
	@$(MAKE) post-release


##@ Meta

help: print-logo ## Display this help message
	+@printf "$(BOLD)Usage:$(RESET)\n"
	+@printf "  make $(BLUE)<target>$(RESET)\n\n"
	+@printf "$(BOLD)Targets:$(RESET)\n"
	+@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	+@printf "\n"

version-matrix: install-uv ## Emit the list of supported Python versions from pyproject.toml
	@${UV_BIN} run .rhiza/utils/version_matrix.py

print-% : ## print the value of a variable (usage: make print-VARIABLE)
	@printf "${BLUE}[INFO] Printing value of variable '$*':${RESET}\n"
	@printf "${BOLD}Value of $*:${RESET}\n"
	@printf "${GREEN}"
	@printf "%s\n" "$($*)"
	@printf "${RESET}"
	@printf "${BLUE}[INFO] End of value for '$*'${RESET}\n"

# Optional: repo extensions (committed)
-include .rhiza/make.d/*.mk

