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
	install-uv \
	install \
	install-extras \
	clean \
	marimo \
	fmt \
	deptry \
	bump \
	release \
	release-dry-run \
	post-release \
	sync \
	help \
	update-readme

UV_INSTALL_DIR ?= ./bin
UV_BIN := ${UV_INSTALL_DIR}/uv
UVX_BIN := ${UV_INSTALL_DIR}/uvx
MARIMO_FOLDER := book/marimo
SOURCE_FOLDER := src
SCRIPTS_FOLDER := .github/rhiza/scripts
CUSTOM_SCRIPTS_FOLDER := .github/rhiza/scripts/customisations

export UV_NO_MODIFY_PATH := 1
export UV_VENV_CLEAR := 1

# Include split Makefiles
-include tests/Makefile.tests
-include book/Makefile.book
-include presentation/Makefile.presentation

##@ Bootstrap
install-uv: ## ensure uv/uvx is installed
	# Ensure the ${UV_INSTALL_DIR} folder exists
	@mkdir -p ${UV_INSTALL_DIR}

	# Install uv/uvx only if they are not already present
	@if [ -x "${UV_INSTALL_DIR}/uv" ] && [ -x "${UV_INSTALL_DIR}/uvx" ]; then \
	  printf "${BLUE}[INFO] uv and uvx already installed in ${UV_INSTALL_DIR}, skipping.${RESET}\n"; \
	else \
	  printf "${BLUE}[INFO] Installing uv and uvx...${RESET}\n"; \
	  if ! curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="${UV_INSTALL_DIR}" sh >/dev/null 2>&1; then \
	    printf "${RED}[ERROR] Failed to install uv${RESET}\n"; \
	    exit 1; \
	  fi; \
	fi

install-extras: ## run custom build script (if exists)
	@if [ -x "${CUSTOM_SCRIPTS_FOLDER}/build-extras.sh" ]; then \
		printf "${BLUE}[INFO] Running custom build script from customisations folder...${RESET}\n"; \
		"${CUSTOM_SCRIPTS_FOLDER}"/build-extras.sh; \
	elif [ -f "${CUSTOM_SCRIPTS_FOLDER}/build-extras.sh" ]; then \
		printf "${BLUE}[INFO] Running custom build script from customisations folder...${RESET}\n"; \
		/bin/sh "${CUSTOM_SCRIPTS_FOLDER}/build-extras.sh"; \
	else \
		printf "${BLUE}[INFO] No custom build script in ${CUSTOM_SCRIPTS_FOLDER}, skipping...${RESET}\n"; \
	fi

install: install-uv install-extras ## install
	# Create the virtual environment only if it doesn't exist
	@if [ ! -d ".venv" ]; then \
	  ${UV_BIN} venv --python 3.12 || { printf "${RED}[ERROR] Failed to create virtual environment${RESET}\n"; exit 1; }; \
	else \
	  printf "${BLUE}[INFO] Using existing virtual environment at .venv, skipping creation${RESET}\n"; \
	fi

	# Check if there is requirements.txt file in the tests folder
	@if [ -f "tests/requirements.txt" ]; then \
	  ${UV_BIN} pip install -r tests/requirements.txt || { printf "${RED}[ERROR] Failed to install test requirements${RESET}\n"; exit 1; }; \
	fi

	# Install the dependencies from pyproject.toml (if it exists)
	@if [ -f "pyproject.toml" ]; then \
	  printf "${BLUE}[INFO] Installing dependencies${RESET}\n"; \
	  ${UV_BIN} sync --all-extras --frozen || { printf "${RED}[ERROR] Failed to install dependencies${RESET}\n"; exit 1; }; \
	else \
	  printf "${YELLOW}[WARN] No pyproject.toml found, skipping install${RESET}\n"; \
	fi

sync: install-uv ## sync with template repository as defined in .github/rhiza/template.yml
	@${UVX_BIN} "rhiza>=0.7.1" materialize --force .

validate: install-uv ## validate project structure against template repository as defined in .github/rhiza/template.yml
	@${UVX_BIN} "rhiza>=0.7.1" validate .

clean: ## clean
	@printf "${BLUE}Cleaning project...${RESET}\n"
	# do not clean .env files
	@git clean -d -X -f -e .env -e '.env.*'
	@rm -rf dist build *.egg-info .coverage .pytest_cache
	@printf "${BLUE}Removing local branches with no remote counterpart...${RESET}\n"
	@git fetch --prune
	@git branch -vv \
	  | grep ': gone]' \
	  | awk '{print $1}' \
	  | xargs -r git branch -D 2>/dev/null || true

##@ Tools
marimo: install ## fire up Marimo server
	@if [ ! -d "${MARIMO_FOLDER}" ]; then \
	  printf " ${YELLOW}[WARN] Marimo folder '${MARIMO_FOLDER}' not found, skipping start${RESET}\n"; \
	else \
	  ${UV_BIN} run --with marimo marimo edit --no-token --headless "${MARIMO_FOLDER}"; \
	fi

##@ Quality and Formatting
deptry: install-uv ## run deptry if pyproject.toml exists
	@if [ -f "pyproject.toml" ]; then \
	  ${UVX_BIN} deptry "${SOURCE_FOLDER}"; \
	else \
	  printf "${YELLOW} No pyproject.toml found, skipping deptry${RESET}\n"; \
	fi

fmt: install-uv ## check the pre-commit hooks and the linting
	@${UVX_BIN} pre-commit run --all-files

##@ Releasing and Versioning
bump: install-uv ## bump version
	@UV_BIN="${UV_BIN}" /bin/sh "${SCRIPTS_FOLDER}/bump.sh"

release: install-uv ## create tag and push to remote with prompts
	@UV_BIN="${UV_BIN}" /bin/sh "${SCRIPTS_FOLDER}/release.sh"

post-release: install-uv ## perform post-release tasks
	@if [ -x "${CUSTOM_SCRIPTS_FOLDER}/post-release.sh" ]; then \
		printf "${BLUE}[INFO] Running post-release script from customisations folder...${RESET}\n"; \
		"${CUSTOM_SCRIPTS_FOLDER}"/post-release.sh; \
	elif [ -f "${CUSTOM_SCRIPTS_FOLDER}/post-release.sh" ]; then \
		printf "${BLUE}[INFO] Running post-release script from customisations folder...${RESET}\n"; \
		/bin/sh "${CUSTOM_SCRIPTS_FOLDER}/post-release.sh"; \
	else \
		printf "${BLUE}[INFO] No post-release script in ${CUSTOM_SCRIPTS_FOLDER}, skipping...${RESET}\n"; \
	fi

##@ Meta

help: ## Display this help message
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

# debugger tools
custom-%: ## run a custom script (usage: make custom-scriptname)
	@SCRIPT="${CUSTOM_SCRIPTS_FOLDER}/$*.sh"; \
	if [ -x "$$SCRIPT" ]; then \
		printf "${BLUE}[INFO] Running custom script $$SCRIPT...${RESET}\n"; \
		"$$SCRIPT"; \
	elif [ -f "$$SCRIPT" ]; then \
		printf "${BLUE}[INFO] Running custom script $$SCRIPT with /bin/sh...${RESET}\n"; \
		/bin/sh "$$SCRIPT"; \
	else \
		printf "${RED}[ERROR] Custom script '$$SCRIPT' not found.${RESET}\n"; \
		printf "Available scripts:\n"; \
		ls -1 "${CUSTOM_SCRIPTS_FOLDER}"/*.sh 2>/dev/null | xargs -n1 basename | sed 's/\.sh$$//' | sed 's/^/  - /'; \
		exit 1; \
	fi

print-% : ## print the value of a variable (usage: make print-VARIABLE)
	@printf "${BLUE}[INFO] Printing value of variable '$*':${RESET}\n"
	@printf "${BOLD}Value of $*:${RESET}\n"
	@printf "${GREEN}"
	@printf "%s\n" "$($*)"
	@printf "${RESET}"
	@printf "${BLUE}[INFO] End of value for '$*'${RESET}\n"
