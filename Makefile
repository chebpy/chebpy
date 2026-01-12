## Makefile for jebel-quant/rhiza
# (https://github.com/jebel-quant/rhiza)
#
# Purpose: Developer tasks using uv/uvx (install, test, docs, marimushka, book).
# Lines with `##` after a target are parsed into help text,
# and lines starting with `##@` create section headers in the help output.
#
# Colors for pretty output in help messages
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
	customisations \
	deptry \
	fmt \
	help \
	install \
	install-extras \
	install-uv \
	marimo \
	post-release \
	release \
	sync \
	update-readme \
	validate \
	version-matrix

INSTALL_DIR ?= ./bin
UV_BIN ?= $(shell command -v uv 2>/dev/null || echo ${INSTALL_DIR}/uv)
UVX_BIN ?= $(shell command -v uvx 2>/dev/null || echo ${INSTALL_DIR}/uvx)
VENV ?= .venv

# Read Python version from .python-version (single source of truth)
PYTHON_VERSION ?= $(shell cat .python-version 2>/dev/null || echo "3.12")
export PYTHON_VERSION

export UV_NO_MODIFY_PATH := 1
export UV_VENV_CLEAR := 1

# Load .rhiza/.env (if present) and export its variables so recipes see them.
-include .rhiza/.env

# Include split Makefiles
-include tests/Makefile.tests
-include book/Makefile.book
-include presentation/Makefile.presentation
-include docker/Makefile.docker
-include .rhiza/customisations/Makefile.customisations
-include .rhiza/agentic/Makefile.agentic
-include .rhiza/Makefile.rhiza
-include .github/Makefile.gh

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

install-extras:: ## run custom build script (if exists)
	@:


install: install-uv install-extras ## install
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

##@ Tools
marimo: install ## fire up Marimo server
	@if [ ! -d "${MARIMO_FOLDER}" ]; then \
	  printf " ${YELLOW}[WARN] Marimo folder '${MARIMO_FOLDER}' not found, skipping start${RESET}\n"; \
	else \
	  ${UV_BIN} run --with marimo marimo edit --no-token --headless "${MARIMO_FOLDER}"; \
	fi

##@ Quality and Formatting
deptry: install-uv ## Run deptry
	@if [ -d ${SOURCE_FOLDER} ]; then \
		$(UVX_BIN) deptry ${SOURCE_FOLDER}; \
	fi

	@if [ -d ${MARIMO_FOLDER} ]; then \
		if [ -d ${SOURCE_FOLDER} ]; then \
			$(UVX_BIN) deptry ${MARIMO_FOLDER} ${SOURCE_FOLDER} --ignore DEP004; \
		else \
		  	$(UVX_BIN) deptry ${MARIMO_FOLDER} --ignore DEP004; \
		fi \
	fi

fmt: install ## check the pre-commit hooks and the linting
	@${UV_BIN} run pre-commit run --all-files

##@ Releasing and Versioning
bump: ## bump version
	@if [ -f "pyproject.toml" ]; then \
		$(MAKE) install; \
		${UVX_BIN} "rhiza[tools]>=0.8.6" tools bump; \
		printf "${BLUE}[INFO] Updating uv.lock file...${RESET}\n"; \
		${UV_BIN} lock; \
	else \
		printf "${YELLOW}[WARN] No pyproject.toml found, skipping bump${RESET}\n"; \
	fi

release: install-uv ## create tag and push to remote with prompts
	@UV_BIN="${UV_BIN}" /bin/sh "${SCRIPTS_FOLDER}/release.sh"

post-release:: install-uv ## perform post-release tasks
	@:


##@ Meta

help: print-logo ## Display this help message
	+@printf "$(BOLD)Usage:$(RESET)\n"
	+@printf "  make $(BLUE)<target>$(RESET)\n\n"
	+@printf "$(BOLD)Targets:$(RESET)\n"
	+@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)
	+@printf "\n"

customisations: ## list available customisation scripts
	@printf "${BLUE}${BOLD}Customisation scripts available in ${CUSTOM_SCRIPTS_FOLDER}:$(RESET)\n"
	@if [ -d "${CUSTOM_SCRIPTS_FOLDER}" ]; then \
		ls -1 "${CUSTOM_SCRIPTS_FOLDER}"/*.sh 2>/dev/null || printf "  (none)\n"; \
	else \
		printf "${YELLOW}[INFO] No customisations found in ${CUSTOM_SCRIPTS_FOLDER}${RESET}\n"; \
	fi

update-readme: ## update README.md with current Makefile help output
	@/bin/sh "${SCRIPTS_FOLDER}/update-readme-help.sh"

version-matrix: install-uv ## Emit the list of supported Python versions from pyproject.toml
	@${UV_BIN} run .rhiza/utils/version_matrix.py

print-% : ## print the value of a variable (usage: make print-VARIABLE)
	@printf "${BLUE}[INFO] Printing value of variable '$*':${RESET}\n"
	@printf "${BOLD}Value of $*:${RESET}\n"
	@printf "${GREEN}"
	@printf "%s\n" "$($*)"
	@printf "${RESET}"
	@printf "${BLUE}[INFO] End of value for '$*'${RESET}\n"
