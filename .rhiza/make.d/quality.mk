## .rhiza/make.d/quality.mk - Quality and Formatting
# This file provides targets for code quality checks, linting, and formatting.

# Configurable list of licenses that fail the compliance scan (semicolon-separated)
LICENSE_FAIL_ON ?= GPL;LGPL;AGPL

# Declare phony targets (they don't produce files)
.PHONY: all deptry fmt license todos semgrep

##@ Quality and Formatting
all: fmt deptry test docs-coverage security license typecheck rhiza-test ## run all CI targets locally

# deptry scans one or more folders for dependency issues. Each feature bundle
# contributes the folders it owns to DEPTRY_FOLDERS (and any per-folder ignores
# to DEPTRY_IGNORE), so this core target never needs to know which bundles are
# present. Core itself contributes SOURCE_FOLDER when it exists; see e.g.
# marimo.mk for a bundle that appends its own folder. Rhiza's own test folder
# (.rhiza/tests) is deliberately excluded: its tooling is provisioned on the fly
# via `uv run --with` in the individual targets, not declared in the project's
# pyproject, so deptry (which validates against pyproject) would only emit noise
# for it.
DEPTRY_FOLDERS ?=
DEPTRY_IGNORE ?=
ifneq ($(wildcard $(SOURCE_FOLDER)),)
DEPTRY_FOLDERS += $(SOURCE_FOLDER)
endif

deptry: install-uv ## Run deptry over the folders contributed by each bundle
	@if [ -n "$(strip $(DEPTRY_FOLDERS))" ]; then \
		printf "${BLUE}[INFO] Running deptry on:${RESET} $(strip $(DEPTRY_FOLDERS))\n"; \
		$(UVX_BIN) -p ${PYTHON_VERSION} deptry $(strip $(DEPTRY_FOLDERS) $(DEPTRY_IGNORE)); \
	else \
		printf "${YELLOW}[WARN] no deptry folders found, skipping.${RESET}\n"; \
	fi

fmt: install-uv ## check the pre-commit hooks and the linting
	@${UVX_BIN} -p ${PYTHON_VERSION} pre-commit run --all-files

todos: ## search and report all TODO/FIXME/HACK comments in the codebase
	@printf "${BLUE}[INFO] Searching for TODO, FIXME, and HACK comments...${RESET}\n"
	@printf "${BOLD}Found the following items:${RESET}\n\n"
	@find . -type f \( -name "*.py" -o -name "*.mk" -o -name "*.sh" -o -name "*.md" -o -name "*.yml" -o -name "*.yaml" \) \
		-not -path "./.venv/*" \
		-not -path "./.git/*" \
		-not -path "./node_modules/*" \
		-not -path "./.tox/*" \
		-not -path "./build/*" \
		-not -path "./dist/*" \
		-print0 | xargs -0 grep -nHE "(TODO|FIXME|HACK):" 2>/dev/null | \
		grep -v "make todos" | \
		awk -F: '{ printf "${YELLOW}%s${RESET}:${GREEN}%s${RESET}: %s\n", $$1, $$2, substr($$0, index($$0,$$3)) }' || \
		printf "${GREEN}[SUCCESS] No TODO/FIXME/HACK comments found!${RESET}\n"
	@printf "\n${BLUE}[INFO] Search complete.${RESET}\n"

semgrep: install ## run Semgrep static analysis
	@printf "${BLUE}[INFO] Running Semgrep...${RESET}\n"
	@if [ -d ${SOURCE_FOLDER} ]; then \
		${UVX_BIN} semgrep --config .rhiza/semgrep.yml ${SOURCE_FOLDER}; \
	else \
		printf "${YELLOW}[WARN] SOURCE_FOLDER '${SOURCE_FOLDER}' not found, skipping semgrep.${RESET}\n"; \
	fi

license: install ## run license compliance scan (fail on GPL, LGPL, AGPL)
	@printf "${BLUE}[INFO] Running license compliance scan...${RESET}\n"
	@${UV_BIN} run --with pip-licenses pip-licenses --fail-on="${LICENSE_FAIL_ON}"
