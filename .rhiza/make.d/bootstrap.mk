## .rhiza/make.d/bootstrap.mk - Bootstrap and Installation
# This file provides targets for setting up the development environment,
# installing dependencies, and cleaning project artifacts.

# Declare phony targets (they don't produce files)
.PHONY: install-uv install clean pre-install post-install

UV_SYNC_ARGS ?= --all-extras --all-groups

# Hook targets (double-colon rules allow multiple definitions)
pre-install:: ; @:
post-install:: ; @:

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

	# Install the dependencies from pyproject.toml (if it exists).
	# --inexact leaves packages uv did not manage in place instead of pruning them each
	# run, so repeated 'make' targets don't churn the environment. Per-target tooling
	# (pytest, interrogate, mutmut, ...) is provisioned on the fly via `uv run --with`
	# in the individual targets, so there is no separate dependency-install step here.
	@if [ -f "pyproject.toml" ]; then \
	  if [ -f "uv.lock" ]; then \
	    if ! ${UV_BIN} lock --check >/dev/null 2>&1; then \
	      printf "${YELLOW}[WARN] uv.lock is out of sync with pyproject.toml${RESET}\n"; \
	      printf "${YELLOW}       Run 'uv sync' to update your lock file and environment${RESET}\n"; \
	      printf "${YELLOW}       Or run 'uv lock' to update only the lock file${RESET}\n"; \
	      exit 1; \
	    fi; \
	    printf "${BLUE}[INFO] Installing dependencies from lock file${RESET}\n"; \
	    ${UV_BIN} sync $(UV_SYNC_ARGS) --inexact --frozen || { printf "${RED}[ERROR] Failed to install dependencies${RESET}\n"; exit 1; }; \
	  else \
	    printf "${YELLOW}[WARN] uv.lock not found. Generating lock file and installing dependencies...${RESET}\n"; \
	    ${UV_BIN} sync $(UV_SYNC_ARGS) --inexact || { printf "${RED}[ERROR] Failed to install dependencies${RESET}\n"; exit 1; }; \
	  fi; \
	else \
	  printf "${YELLOW}[WARN] No pyproject.toml found, skipping install${RESET}\n"; \
	fi

	# Install pre-commit hooks (skip when core.hooksPath is set, e.g. by an
	# external hook manager — pre-commit refuses to install in that case)
	@if [ -f ".pre-commit-config.yaml" ]; then \
	  if [ -n "$$(git config --get core.hooksPath 2>/dev/null)" ]; then \
	    printf "${BLUE}[INFO] Skipping pre-commit hook install: core.hooksPath is set${RESET}\n"; \
	  else \
	    printf "${BLUE}[INFO] Installing pre-commit hooks...${RESET}\n"; \
	    ${UVX_BIN} -p ${PYTHON_VERSION} pre-commit install || { printf "${YELLOW}[WARN] Failed to install pre-commit hooks${RESET}\n"; }; \
	  fi; \
	fi

	@$(MAKE) post-install
	
	# Display success message with activation instructions
	@printf "\n${GREEN}[SUCCESS] Installation complete!${RESET}\n\n"
	@printf "${BLUE}To activate the virtual environment, run:${RESET}\n"
	@printf "${YELLOW}  source ${VENV}/bin/activate${RESET}\n\n"

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

	@git branch -vv | awk '/: gone]/ && $$1 != "*" && $$1 != "+" {print $$1}' | xargs -r git branch -D
