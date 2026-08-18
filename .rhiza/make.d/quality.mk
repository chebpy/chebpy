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

# semgrep takes its folder list from an accumulator, exactly as `typecheck`, `security`,
# `docs-coverage` and `deps` do since #1505. It was left on the old `[ -d $(SOURCE_FOLDER) ]`
# form by that change because it is the one static gate owned by *core* rather than by
# python.mk, so it sat outside the file being edited — and outside the guard, whose
# `_SCOPED_GATES` list named only the four (#1511).
#
# The consequence was the same one #1505 existed to remove: on a repo with no `src/` the
# recipe printed a warning and exited 0 having analysed nothing. That is a silent pass in
# the mother repo, where `.github/workflows/rhiza_weekly.yml` runs `make semgrep` on a
# schedule, and in any downstream project keeping Python outside its source root.
#
# The accumulator is declared here, in core, and seeded from SOURCE_FOLDER when that
# folder exists — SOURCE_FOLDER is a core variable (rhiza.mk), not a Python-layer one, so
# nothing about this reaches across the language-layer boundary. A project with no `src/`
# contributes its folders by appending, the way .rhiza/make.d/bundles.mk contributes
# `utils` here.
SEMGREP_FOLDERS ?=
ifneq ($(wildcard $(SOURCE_FOLDER)),)
SEMGREP_FOLDERS += $(SOURCE_FOLDER)
endif

semgrep: install ## run Semgrep static analysis
	@semgrep_paths="$(strip $(SEMGREP_FOLDERS))"; \
	if [ -n "$${semgrep_paths}" ]; then \
		printf "${BLUE}[INFO] Running Semgrep in:$${semgrep_paths}${RESET}\n"; \
		${UVX_BIN} semgrep --config .rhiza/semgrep.yml $${semgrep_paths}; \
	else \
		printf "${YELLOW}[WARN] No semgrep folders found (SEMGREP_FOLDERS is empty and SOURCE_FOLDER='${SOURCE_FOLDER}' does not exist), skipping semgrep.${RESET}\n"; \
	fi

# The rhiza repository checks — README fences, release tags, the manifest and its
# bump-my-version wiring — arrive as a *dependency* rather than as files (#1540). Until
# now the template synced them into `.rhiza/tests/`: seven modules plus a `conftest.py`
# nobody downstream may edit, one per bundle that owns an assertion. Five costs came with
# the copy, and all five go away here:
#
#   - seven template-owned files in every consumer's tree, plus `pythonpath = .rhiza/tests`
#     in pytest.ini so the synced suite could import itself;
#   - `.rhiza/tests` appended to `docs-coverage`'s interrogate paths, holding *template*
#     code to the project's 100% docstring bar;
#   - `--with pytest-timeout --with python-dotenv --with packaging` spelled out in this
#     recipe, because a copied file carries no dependency metadata;
#   - the template's own meta-tests (`TestSkipFlag`, asserting rhiza's fence helper against
#     itself) re-running in every project that syncs;
#   - and the duplication the copy imposed: `SKIP_FLAG`/`_should_skip` existed twice,
#     because bundles are copied independently so a shared helper had no third home. One
#     distribution is that home.
#
# pytest-rhiza is those modules installed. The fixtures (`root`, `logger`, `latest_tag`)
# arrive through its `pytest11` entry point; the checks have to be named explicitly with
# `--pyargs`, because an entry point can contribute plugins but not *tests*.
#
# Ownership of the selection does not move — it stays with the bundle that owns the
# assertion, resolved at sync time. So RHIZA_CHECKS is an accumulator like DEPTRY_FOLDERS
# and SEMGREP_FOLDERS above: core names the two language-neutral modules, and python-core,
# rust-core, go-core and `tests` each append their own. One `+=` line per bundle replaces
# one synced file per bundle. Nothing sniffs the manifest at runtime to decide what
# applies, so a misconfigured repo still goes red rather than quietly skipping a check.
#
# The version is pinned, and the pin is what keeps the second cost of file-copy delivery
# from simply moving: a repo synced at one template release ran that release's assertions
# forever, with no signal that newer ones existed. The pin travels *in* the template, so
# the checks and the template still move as one number — bumped here, delivered by the
# next sync. A consumer who wants to lead or lag overrides RHIZA_CHECKS_VERSION.
RHIZA_CHECKS_VERSION ?= 0.2.1

# `?=` empty and then `+=` the seed, matching DEPTRY_FOLDERS and the rest — and here the
# shape is load-bearing rather than merely conventional. `.rhiza/rhiza.mk` includes
# `make.d/*.mk` alphabetically, so `go.mk` and `python.mk` are both read *before* this
# file. In make, `+=` on an undefined variable defines it, which would leave a bare
# `RHIZA_CHECKS ?= <core's two modules>` here as a no-op on exactly those two layers: a
# Python or Go project would run its own checks and silently lose the neutral README and
# release-tag ones. Seeding by append cannot be ordered out.
RHIZA_CHECKS ?=
RHIZA_CHECKS += pytest_rhiza.checks.test_readme pytest_rhiza.checks.test_release_tags

# `install` is a deliberate exception to core's rule of never naming a layer-owned
# target. It is a *prerequisite* here, not a definition, and the checks need it: the
# docstring check imports the project's own packages to run their doctests, which
# requires the dependencies installed. Make resolves it because every fragment is
# included into one namespace, and every profile selects exactly one language layer —
# `test_a_profile_never_selects_two_bundles_from_one_layer` and
# `test_every_profile_selects_a_language_layer` bracket that from both sides. A
# core-only tree is not a shipped configuration; there, this fails naming `install`.
#
# RHIZA_DOCTEST_FOLDERS carries the doctest scope to the docstring check, which had
# resolved `src` and nothing else — so a project keeping Python outside its source root
# had its docstring examples skipped rather than checked (#1517). DOCSTRING_FOLDERS is the
# same accumulator `make docs-coverage` reads, so "has a docstring" and "the example in it
# still works" cannot end up scoped differently.
#
# Naming a python-core variable from core is safe in the way `install` above is: on a Rust
# or Go layer the variable is simply undefined, the value is empty, and the check falls
# back to SOURCE_FOLDER from `.rhiza/.env` exactly as before.
#
# The resolved check list is printed rather than left implicit, for the reason the other
# gates print their folder lists: under `--pyargs` pytest reports node ids with no file
# name at all, so the run itself is no evidence of *which* checks ran.
rhiza-test: install ## run the rhiza repository checks
	@checks="$(strip $(RHIZA_CHECKS))"; \
	if [ -z "$${checks}" ]; then \
		printf "${YELLOW}[WARN] RHIZA_CHECKS is empty, skipping the rhiza checks${RESET}\n"; \
		exit 0; \
	fi; \
	if [ -d ".rhiza/tests" ]; then \
		printf "${YELLOW}[WARN] .rhiza/tests/ is a leftover from before the checks became a dependency (#1540); nothing runs it. Delete it: git rm -r .rhiza/tests${RESET}\n"; \
	fi; \
	printf "${BLUE}[INFO] Running the rhiza checks: $${checks}${RESET}\n"; \
	RHIZA_DOCTEST_FOLDERS="$(strip $(DOCSTRING_FOLDERS))" \
	${UV_BIN} run --with 'pytest-rhiza==$(RHIZA_CHECKS_VERSION)' pytest --pyargs $${checks}
