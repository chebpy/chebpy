## .rhiza/make.d/python.mk - the Python language layer (bundle: python-core)
#
# Everything in rhiza that only makes sense because the project is written in
# Python. `core` provides the make framework and uv/uvx as a tool runner; this
# file turns that into a Python project: the virtualenv, the `install` that syncs
# it, dependency and licence analysis of the declared dependencies, and the `all`
# aggregate naming the gates.
#
# A sibling language layer (rust.mk, from a rust-core bundle) ships the same
# *target names* — install, all — with different recipes. That contract is the
# reason book.mk and the CI workflows can call `make install` without knowing the
# language. Only one language layer is ever synced into a repo.
#
# Every gate named in `all` below is defined *here*, which is what makes this file
# comparable to rust.mk and go.mk. Until #1475 four of them — test, typecheck,
# security, docs-coverage — lived in the `tests` bundle's test.mk while `all` named
# them, so a project syncing `core + python-core` without `tests` had an `all` that
# could not run. The `tests` bundle now carries only the genuinely optional extras
# (benchmark, hypothesis-test, stress, mutation).

# Declare phony targets (they don't produce files)
.PHONY: all deps deptry docs-coverage install license security test test-pyproject typecheck

# The project virtualenv, and the interpreter that fills it. PYTHON_VERSION is
# declared in rhiza.mk (core needs a Python to run its own tooling on); here
# `.python-version` — which this bundle ships — makes it the project's version too.
VENV ?= .venv
UV_SYNC_ARGS ?= --all-extras --all-groups

export UV_VENV_CLEAR := 1

# Configurable list of licenses that fail the compliance scan (semicolon-separated).
#
# These are matched as *substrings* of the reported licence -- see `--partial-match`
# in the `license` recipe. Without that flag pip-licenses compares against the whole
# licence string, and `GPL` never equals a real classifier such as "GNU General
# Public License v2 or later (GPLv2+)", so the gate passed with a GPL package
# installed.
LICENSE_FAIL_ON ?= GPL;LGPL;AGPL

# Packages exempted from the scan by name, space-separated. Empty by default: an
# exemption should be a deliberate, reviewable act -- in a project's own Makefile, or
# in the bundle that owns the dependency -- not something this layer grants globally.
# Two cases are legitimate, and `?=` plus `+=` makes it an accumulator for both:
#
#   1. A copyleft *development* dependency -- a reference implementation a differential
#      test runs against -- never imported by the shipped package and never redistributed.
#   2. A package offered under several licences at the reader's choice. pip-licenses
#      reports the classifier list joined with "; " and has no notion of *or*, so
#      `--partial-match` fires on the copyleft option even where a permissive one is
#      taken. marimo.mk exempts docutils ("BSD License; GNU General Public License
#      (GPL); Public Domain") for exactly this reason.
#
# Case 2 is the one to be careful with: check that a permissive option really is on
# offer, rather than that the string merely looks long.
LICENSE_IGNORE_PACKAGES ?=

# Default directory for tests. Also read by the `tests` bundle's test.mk, which
# requires this layer, so the two never disagree about where tests live.
TESTS_FOLDER := tests

# Minimum coverage percent for tests to pass
# (Can be overridden in local.mk or via environment variable)
COVERAGE_FAIL_UNDER ?= 90

# Which static type checker(s) the 'typecheck' target runs: ty, mypy, or both.
# Running both is the default for backward compatibility, but ty and mypy
# occasionally disagree (e.g. one accepts a suppression the other still flags),
# forcing duplicate `# type: ignore` / `# ty: ignore` comments. Set this to
# 'ty' or 'mypy' in local.mk or .rhiza/.env to run a single checker instead.
TYPECHECKER ?= both

##@ Python
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
	#
	# `-c` for the reason quality.mk's `fmt` passes `--config`, and it must be repeated
	# here: prek bakes the flag into the generated shim, so without it the commit-time
	# gate rediscovers nested projects and stops meaning what `make fmt` means. A
	# consumer who wants prek's monorepo behaviour drops the flag from both places.
	@if [ -f ".pre-commit-config.yaml" ]; then \
	  if [ -n "$$(git config --get core.hooksPath 2>/dev/null)" ]; then \
	    printf "${BLUE}[INFO] Skipping pre-commit hook install: core.hooksPath is set${RESET}\n"; \
	  else \
	    printf "${BLUE}[INFO] Installing pre-commit hooks...${RESET}\n"; \
	    ${UVX_BIN} prek install -c .pre-commit-config.yaml || { printf "${YELLOW}[WARN] Failed to install pre-commit hooks${RESET}\n"; }; \
	  fi; \
	fi

	@$(MAKE) post-install
	
	# Display success message with activation instructions
	@printf "\n${GREEN}[SUCCESS] Installation complete!${RESET}\n\n"
	@printf "${BLUE}To activate the virtual environment, run:${RESET}\n"
	@printf "${YELLOW}  source ${VENV}/bin/activate${RESET}\n\n"

all: fmt deps test docs-coverage security license typecheck rhiza-test ## run all CI targets locally

# deptry scans one or more folders for dependency issues. Each feature bundle
# contributes the folders it owns to DEPTRY_FOLDERS (and any per-folder ignores
# to DEPTRY_IGNORE), so this target never needs to know which bundles are
# present. The language layer itself contributes SOURCE_FOLDER when it exists; see e.g.
# marimo.mk for a bundle that appends its own folder. There is no `.rhiza/tests` carve-out
# any more: the rhiza checks are a pinned dependency of the `rhiza-test` gate rather than
# files in the tree (#1540), so there is nothing for deptry to resolve against pyproject
# and nothing to exclude.
DEPTRY_FOLDERS ?=
DEPTRY_IGNORE ?=
DEPTRY_FOLDERS += $(wildcard $(SOURCE_FOLDER))

# The seed is deferred, and that is the whole point of the `$(wildcard ...)` rather than
# the `ifneq ($(wildcard $(SOURCE_FOLDER)),)` guard that used to wrap this line (#1534).
#
# `?=` (and `+=` on a not-yet-defined variable) creates a *recursive* variable, so the
# text appended here is expanded when a gate uses it — after every makefile has been
# read. An `ifneq` is not: it is evaluated where it is written, which is while python.mk
# itself is being parsed.
#
# That difference was load-bearing, because the root Makefile reads `local.mk` *after*
# `include .rhiza/rhiza.mk`. A project whose source root is not `src/` and which set
# `SOURCE_FOLDER` there — the file CLAUDE.md points developers at — had the conditional
# already decided against it on the `?= src` default, so `deps`, `typecheck`, `security`,
# `docs-coverage` and `test` each fell through to their empty-list branch and exited 0
# having measured nothing. Silently: an empty folder list warns rather than fails, which
# is the failure mode #1505, #1511 and #1516 each closed one gate at a time.
#
# Deferring it makes the accumulators independent of *where* SOURCE_FOLDER is set —
# `.rhiza/.env`, the environment, the root Makefile above the include, the command line,
# or `local.mk` below it all now agree. The non-existent case is unchanged: `$(wildcard)`
# expands to nothing, so a project without the folder still gets an empty seed.

# The same accumulator shape, for the other four path-scoped gates. `deps` has had
# one since the bundle model began; `typecheck`, `security` and `docs-coverage` each
# hard-coded SOURCE_FOLDER instead, so any Python a project keeps *outside* its source
# root was unreachable by three of the four static gates (#1505). The mother repo is
# the extreme case — it has no `src/` at all, so those three exited 0 having measured
# nothing — but a downstream project with a `scripts/` or `tools/` directory has the
# identical hole.
#
# COVERAGE_FOLDERS joined them in #1516, and it is the same bug one gate later: `test`
# still passed `--cov=$(SOURCE_FOLDER)` behind a `[ -d ... ]`, so on a project with no
# `src/` the suite ran and measured no coverage at all — silently, since a missing
# source folder is a warning rather than a failure. `utils/` in this very repo was the
# proof: reachable by four gates, invisible to the fifth.
#
# It is a *separate* accumulator from the other four rather than a reuse of one, because
# the questions differ. A folder can be worth type-checking or scanning without being
# worth a coverage percentage (generated code, a vendored tree), and DEPTRY_FOLDERS in
# particular already carries folders — marimo.mk's notebooks — that no test imports.
#
# Each seeds itself from SOURCE_FOLDER when that folder exists, so a project that never
# touches these variables gets precisely the previous behaviour. A bundle or a consuming
# Makefile contributes a folder by appending, exactly as marimo.mk does for DEPTRY_FOLDERS.
#
# The folder list is computed here, at make time, rather than by the `[ -d ... ]` tests
# that used to live in each recipe. That is a deliberate change in what `make -n` prints:
# the shell form emitted its assignment whether or not the folder existed, so a dry run
# reported a scope the real run would not use.
# One asymmetry here is deliberate rather than an oversight, and is recorded because it
# reads as one (#1518): `docs-coverage` folds in the test folders on top of this list,
# while `typecheck` does not. So a repo's tests are held to the docstring bar and not to
# the typing bar.
#
# The reason is that `typecheck` runs mypy in --strict mode, and strict mode's
# no-untyped-def is a poor fit for a pytest suite: fixtures arrive untyped from pytest,
# parametrize decorators erase signatures, and monkeypatch stand-ins must match a
# signature they cannot import. Measured on this repo, extending the scope reports 627
# errors in 66 files, none of which is a defect. A consumer who does want its tests
# type-checked appends them: `TYPECHECK_FOLDERS += tests` in the root Makefile or
# local.mk, which is the accumulator's whole purpose.
TYPECHECK_FOLDERS ?=
BANDIT_FOLDERS ?=
DOCSTRING_FOLDERS ?=
COVERAGE_FOLDERS ?=
# Deferred for the reason spelled out at DEPTRY_FOLDERS above (#1534): expanded at use,
# so it no longer matters whether SOURCE_FOLDER was set above or below the include.
TYPECHECK_FOLDERS += $(wildcard $(SOURCE_FOLDER))
BANDIT_FOLDERS += $(wildcard $(SOURCE_FOLDER))
DOCSTRING_FOLDERS += $(wildcard $(SOURCE_FOLDER))
COVERAGE_FOLDERS += $(wildcard $(SOURCE_FOLDER))

# The two rhiza checks this layer owns, appended to core's accumulator (#1540). They were
# two synced files — `.rhiza/tests/test_pyproject.py` and `.rhiza/tests/test_docstrings.py`
# — and are now two module names in pytest-rhiza, which core's `rhiza-test` names with
# `--pyargs`. Selection stays with the bundle that owns the assertion, resolved at sync
# time: a Rust project appends `test_cargo_toml` and never sees these two, exactly as it
# never received the files.
RHIZA_CHECKS += pytest_rhiza.checks.test_pyproject pytest_rhiza.checks.test_docstrings

# Named `deps`, matching rust.mk and go.mk. This was the one gate whose *target name*
# differed by language (#1474): `deptry` names the tool, which nothing else in the
# contract does — pytest is `test`, mypy is `typecheck`, bandit is `security`. So
# `make deps` used to fail on a Python project and `make deptry` on the other two, and
# nothing language-neutral could invoke the gate without knowing the language first.
#
# The variables stay `DEPTRY_*`: they name the tool's *arguments*, honestly so —
# marimo.mk appends `--ignore DEP004`, a deptry rule code — and they are the accumulator
# interface a downstream local.mk writes to, so renaming them would break consumers for
# nothing.
deps: install-uv ## Run deptry over the folders contributed by each bundle
	@if [ -n "$(strip $(DEPTRY_FOLDERS))" ]; then \
		printf "${BLUE}[INFO] Running deptry on:${RESET} $(strip $(DEPTRY_FOLDERS))\n"; \
		$(UVX_BIN) -p ${PYTHON_VERSION} deptry $(strip $(DEPTRY_FOLDERS) $(DEPTRY_IGNORE)); \
	else \
		printf "${YELLOW}[WARN] no deptry folders found, skipping.${RESET}\n"; \
	fi

# Deprecated alias. A downstream local.mk, a hand-written CI job or a contributor's
# muscle memory all still call `make deptry`; a hard rename would break them at the
# point where the gate simply stops existing. Warns rather than failing, and can go in
# a future release once consumers have moved.
deptry: ## deprecated alias for `deps`
	@printf "${YELLOW}[WARN] \`make deptry\` is deprecated and will be removed; use \`make deps\`.${RESET}\n"
	@$(MAKE) --no-print-directory deps

# --ignore-packages takes one or more names and errors on a bare flag, so it is
# omitted entirely when nothing is exempted.
LICENSE_IGNORE_FLAG = $(if $(strip $(LICENSE_IGNORE_PACKAGES)),--ignore-packages $(strip $(LICENSE_IGNORE_PACKAGES)),)

license: install ## run license compliance scan (fail on GPL, LGPL, AGPL)
	@printf "${BLUE}[INFO] Running license compliance scan...${RESET}\n"
	@${UV_BIN} run --with pip-licenses pip-licenses \
		--fail-on="${LICENSE_FAIL_ON}" --partial-match ${LICENSE_IGNORE_FLAG}

##@ Development and Testing

# The 'test' target runs the complete test suite.
# 1. Cleans up any previous test results in _tests/ and stale coverage data.
# 2. Creates directories for HTML coverage and test reports.
# 3. Invokes pytest via the local virtual environment.
# 4. Generates terminal output, HTML coverage, JSON coverage, and HTML test reports.
#
# Parallel (pytest-xdist) runs occasionally crash *during worker/session
# teardown* even though every test passed — e.g. the xdist
# `worker_workerfinished` KeyError or a pytest-html report-write race. pytest
# signals these runner-internal crashes with exit code 3 (INTERNALERROR),
# which is distinct from real test failures (1), interruptions (2) and usage
# errors (4). We therefore retry the suite once on exit code 3 only, so a
# teardown race no longer flips a green run red, while genuine failures still
# fail immediately. Stale `.coverage*` data is removed before each attempt so a
# previously crashed run cannot leave a corrupt data file that reports a false
# 0% coverage on the next run.
test:: install ## run all tests
	@rm -rf _tests
	@coverage_paths="$(strip $(COVERAGE_FOLDERS))"; \
	if [ -z "$$(find ${TESTS_FOLDER} -name 'test_*.py' -o -name '*_test.py' 2>/dev/null)" ]; then \
	  printf "${YELLOW}[WARN] No test files found in ${TESTS_FOLDER}, skipping tests.${RESET}\n"; \
	  exit 0; \
	fi; \
	set -- -n auto \
	  --ignore=${TESTS_FOLDER}/benchmarks \
	  --ignore=${TESTS_FOLDER}/stress; \
	if [ -n "$${coverage_paths}" ]; then \
	  printf "${BLUE}[INFO] Measuring coverage in:$${coverage_paths}${RESET}\n"; \
	  for coverage_path in $${coverage_paths}; do \
	    set -- "$$@" --cov="$${coverage_path}"; \
	  done; \
	  set -- "$$@" \
	    --cov-report=term \
	    --cov-report=html:_tests/html-coverage \
	    --cov-fail-under=$(COVERAGE_FAIL_UNDER) \
	    --cov-report=json:_tests/coverage.json \
	    --cov-report=xml:_tests/coverage.xml; \
	else \
	  printf "${YELLOW}[WARN] No coverage folders found (COVERAGE_FOLDERS is empty and SOURCE_FOLDER='${SOURCE_FOLDER}' does not exist), running tests without coverage${RESET}\n"; \
	fi; \
	set -- "$$@" --html=_tests/html-report/report.html; \
	attempt=1; max_attempts=2; \
	while :; do \
	  rm -f .coverage .coverage.* _tests/coverage.xml _tests/coverage.json 2>/dev/null || true; \
	  mkdir -p _tests/html-coverage _tests/html-report; \
	  ${UV_BIN} run --with pytest --with pytest-cov --with pytest-xdist --with pytest-html --with pytest-timeout --with pytest-mock pytest "$$@"; status=$$?; \
	  if [ $$status -ne 3 ]; then exit $$status; fi; \
	  if [ $$attempt -ge $$max_attempts ]; then \
	    printf "${RED}[ERROR] pytest reported an internal (teardown) error after %s attempts; failing.${RESET}\n" "$$attempt"; \
	    exit $$status; \
	  fi; \
	  printf "${YELLOW}[WARN] pytest exited 3 (xdist/teardown internal error, all tests may have passed); retrying suite (attempt %s/%s)...${RESET}\n" "$$((attempt + 1))" "$$max_attempts"; \
	  attempt=$$((attempt + 1)); \
	done

# The 'typecheck' target runs static type analysis using ty and/or mypy.
# 1. Takes the folder list from TYPECHECK_FOLDERS (see the accumulator block above).
# 2. Depending on TYPECHECKER (ty|mypy|both, default: both), runs ty,
#    mypy in strict mode, or both in sequence as a cross-check.
typecheck: install ## run ty and/or mypy type checking (TYPECHECKER=ty|mypy|both, default: both)
	@typecheck_paths="$(strip $(TYPECHECK_FOLDERS))"; \
	if [ -z "$${typecheck_paths}" ]; then \
	  printf "${YELLOW}[WARN] No typecheck folders found (TYPECHECK_FOLDERS is empty and SOURCE_FOLDER='${SOURCE_FOLDER}' does not exist), skipping typecheck${RESET}\n"; \
	  exit 0; \
	fi; \
	case "${TYPECHECKER}" in \
	  ty) \
	    printf "${BLUE}[INFO] Running ty type checking in:$${typecheck_paths}${RESET}\n"; \
	    ${UV_BIN} run --with ty ty check $${typecheck_paths} \
	    ;; \
	  mypy) \
	    printf "${BLUE}[INFO] Running mypy strict type checking in:$${typecheck_paths}${RESET}\n"; \
	    ${UV_BIN} run --with mypy mypy --strict $${typecheck_paths} \
	    ;; \
	  both) \
	    printf "${BLUE}[INFO] Running ty type checking in:$${typecheck_paths}${RESET}\n"; \
	    ${UV_BIN} run --with ty ty check $${typecheck_paths} && \
	    printf "${BLUE}[INFO] Running mypy strict type checking in:$${typecheck_paths}${RESET}\n"; \
	    ${UV_BIN} run --with mypy mypy --strict $${typecheck_paths} \
	    ;; \
	  *) \
	    printf "${RED}[ERROR] Invalid TYPECHECKER='${TYPECHECKER}' (expected: ty, mypy, or both)${RESET}\n"; \
	    exit 1 \
	    ;; \
	esac

# The 'security' target runs bandit over the folders in BANDIT_FOLDERS (see the
# accumulator block above). Scope *within* those folders is .bandit's job, not this
# target's — see that file for why it is the single source of truth (#1493).
security: install ## run security scans (bandit)
	@bandit_paths="$(strip $(BANDIT_FOLDERS))"; \
	if [ -n "$${bandit_paths}" ]; then \
	  printf "${BLUE}[INFO] Running bandit security scan in:$${bandit_paths}${RESET}\n"; \
	  ${UVX_BIN} bandit -r $${bandit_paths} -ll -q --ini .bandit; \
	else \
	  printf "${YELLOW}[WARN] No bandit scan folders found (BANDIT_FOLDERS is empty and SOURCE_FOLDER='${SOURCE_FOLDER}' does not exist), skipping bandit${RESET}\n"; \
	fi

# The 'docs-coverage' target checks documentation coverage using interrogate.
# 1. Takes DOCSTRING_FOLDERS (see the accumulator block above) as the base list.
# 2. Adds the project's test folder, which is checked wherever it exists and is not part
#    of the accumulator: a consumer contributing a folder means source, not tests.
# 3. Runs interrogate with verbose output against the result.
#
# `.rhiza/tests` used to be folded in as a third path, and that was a cost of delivering
# the rhiza checks by file-copy: template code nobody downstream may edit was held to the
# project's own 100% docstring bar. The checks are a dependency now (#1540), so the path
# is gone rather than merely skipped.
docs-coverage: install ## check documentation coverage with interrogate
	@docstring_paths="$(strip $(DOCSTRING_FOLDERS))"; \
	if [ -d "tests" ]; then \
	  docstring_paths="$${docstring_paths} tests"; \
	fi; \
	if [ -n "$${docstring_paths}" ]; then \
	  printf "${BLUE}[INFO] Checking documentation coverage in:$${docstring_paths}${RESET}\n"; \
	  ${UV_BIN} run --with interrogate interrogate -vv --fail-under 100 --ignore-init-method --ignore-magic $${docstring_paths}; \
	else \
	  printf "${YELLOW}[WARN] No docs-coverage folders found (DOCSTRING_FOLDERS is empty, SOURCE_FOLDER='${SOURCE_FOLDER}' does not exist, and there are no test folders), skipping docs-coverage${RESET}\n"; \
	fi

# The single-check shortcut for the gate `rhiza-test` runs in full. It names the module
# rather than a path for the same reason that gate does: since #1540 the check is installed
# from pytest-rhiza, not synced into the tree, so there is no file here to point pytest at.
test-pyproject: install ## run pyproject.toml structure tests
	@${UV_BIN} run --with 'pytest-rhiza==$(RHIZA_CHECKS_VERSION)' pytest --pyargs pytest_rhiza.checks.test_pyproject \
		-v \
		--tb=long \
		--showlocals \
		-rA \
		--durations=0 \
		--no-header
