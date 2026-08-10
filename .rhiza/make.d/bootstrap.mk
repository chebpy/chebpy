## .rhiza/make.d/bootstrap.mk - Bootstrap and Installation
# This file provides the language-neutral half of setup: it puts uv/uvx on the
# path (rhiza runs pre-commit, mkdocs and semgrep through uvx whatever the
# project is written in), declares the install hooks, and cleans artifacts.
#
# `install` itself belongs to the language layer, because what it means differs:
# python.mk (bundle python-core) creates a virtualenv and runs `uv sync`.

# Declare phony targets (they don't produce files)
.PHONY: install-uv clean pre-install post-install

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
