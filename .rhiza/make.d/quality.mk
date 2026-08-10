## .rhiza/make.d/quality.mk - Quality and Formatting
# The language-neutral gates: pre-commit, the TODO sweep, semgrep, and the runner for
# the template's own test suite. Everything that needs to know how the project declares
# its dependencies — `deptry`, the licence-compliance scan — and the `all` aggregate that
# names the per-language gates live in the language layer (python.mk, from the
# python-core bundle).

# Declare phony targets (they don't produce files)
.PHONY: fmt todos semgrep rhiza-test

##@ Quality and Formatting
# prek rather than pre-commit: a Rust reimplementation that reads the same
# `.pre-commit-config.yaml` and needs no Python of its own. Two consequences, one
# gained and one that has to be asked for.
#
# Gained: the `-p ${PYTHON_VERSION}` this recipe used to carry is gone. That flag
# existed because `uvx pre-commit` had to choose an interpreter to run pre-commit
# *itself* on, and a Rust or Go project ships no `.python-version` — so the whole
# language-neutral half of the template rested on rhiza.mk's fallback resolving to
# something real. prek is a binary and provisions each hook's toolchain itself, so the
# coupling is removed rather than merely satisfied.
#
# Asked for: `--config`. By default prek treats every directory below the root that
# holds a `.pre-commit-config.yaml` as a separate *project* and runs each one's hooks —
# useful in a monorepo, surprising anywhere else, and wrong in rhiza's own repo, where
# `bundles/{python,rust,go}-core` each ship one as template content. (go-core's hooks
# then run `go vet ./...` in a directory with no `go.mod` and fail.) Naming the config
# explicitly disables that discovery, so `make fmt` means exactly what it meant under
# pre-commit: this repo's config, once. A consumer who wants the monorepo behaviour
# drops the flag. `.prekignore` is documented for the same job but is not honoured by
# prek 0.4.12, so it is not what this relies on.
fmt: install-uv ## check the pre-commit hooks and the linting
	@${UVX_BIN} prek run --all-files --config .pre-commit-config.yaml

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

# Owned by core, not by a language layer, because nothing about it is language-specific:
# `.rhiza/tests` validates the *template* — READMEs, release config, YAML — and runs
# under pytest whatever the project is written in. `uv` lives in core for the same
# reason. Each layer contributes its own modules to the suite (test_pyproject.py,
# test_cargo_toml.py, test_go_module.py) while the runner stays here, so the recipe
# exists once rather than identically in all three (#1471).
#
# `install` is a deliberate exception to core's rule of never naming a layer-owned
# target. It is a *prerequisite* here, not a definition, and the suite needs it: the
# shipped test_docstrings.py imports the project's own packages to run their doctests,
# which requires the dependencies installed. Make resolves it because every fragment is
# included into one namespace, and every profile selects exactly one language layer —
# `test_a_profile_never_selects_two_bundles_from_one_layer` and
# `test_every_profile_selects_a_language_layer` bracket that from both sides. A
# core-only tree is not a shipped configuration; there, this fails naming `install`.
rhiza-test: install ## run rhiza's own tests (if any)
	@if [ -d ".rhiza/tests" ]; then \
		${UV_BIN} run --with pytest --with pytest-timeout --with python-dotenv --with packaging pytest .rhiza/tests; \
	else \
		printf "${YELLOW}[WARN] No .rhiza/tests directory found, skipping rhiza-tests${RESET}\n"; \
	fi
