## Makefile.tests - Testing and benchmarking targets
# This file is included by the main Makefile.
# It provides targets for running the test suite with coverage and
# executing performance benchmarks.

# Declare phony targets (they don't produce files)
.PHONY: test benchmark typecheck security docs-coverage hypothesis-test coverage-badge stress test-pyproject mutation

# Default directory for tests
TESTS_FOLDER := tests

# Minimum coverage percent for tests to pass
# (Can be overridden in local.mk or via environment variable)
COVERAGE_FAIL_UNDER ?= 90

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
	@if [ -z "$$(find ${TESTS_FOLDER} -name 'test_*.py' -o -name '*_test.py' 2>/dev/null)" ]; then \
	  printf "${YELLOW}[WARN] No test files found in ${TESTS_FOLDER}, skipping tests.${RESET}\n"; \
	  exit 0; \
	fi; \
	if [ -d ${SOURCE_FOLDER} ]; then \
	  set -- -n auto \
	    --ignore=${TESTS_FOLDER}/benchmarks \
	    --ignore=${TESTS_FOLDER}/stress \
	    --cov=${SOURCE_FOLDER} \
	    --cov-report=term \
	    --cov-report=html:_tests/html-coverage \
	    --cov-fail-under=$(COVERAGE_FAIL_UNDER) \
	    --cov-report=json:_tests/coverage.json \
	    --cov-report=xml:_tests/coverage.xml \
	    --html=_tests/html-report/report.html; \
	else \
	  printf "${YELLOW}[WARN] Source folder ${SOURCE_FOLDER} not found, running tests without coverage${RESET}\n"; \
	  set -- -n auto \
	    --ignore=${TESTS_FOLDER}/benchmarks \
	    --ignore=${TESTS_FOLDER}/stress \
	    --html=_tests/html-report/report.html; \
	fi; \
	attempt=1; max_attempts=2; \
	while :; do \
	  rm -f .coverage .coverage.* _tests/coverage.xml _tests/coverage.json 2>/dev/null || true; \
	  mkdir -p _tests/html-coverage _tests/html-report; \
	  ${UV_BIN} run pytest "$$@"; status=$$?; \
	  if [ $$status -ne 3 ]; then exit $$status; fi; \
	  if [ $$attempt -ge $$max_attempts ]; then \
	    printf "${RED}[ERROR] pytest reported an internal (teardown) error after %s attempts; failing.${RESET}\n" "$$attempt"; \
	    exit $$status; \
	  fi; \
	  printf "${YELLOW}[WARN] pytest exited 3 (xdist/teardown internal error, all tests may have passed); retrying suite (attempt %s/%s)...${RESET}\n" "$$((attempt + 1))" "$$max_attempts"; \
	  attempt=$$((attempt + 1)); \
	done

# The 'typecheck' target runs static type analysis using ty and mypy.
# 1. Builds a list of existing Python source folders to check.
# 2. Runs ty on those folders.
# 3. Runs mypy in strict mode on those folders as a cross-check.
typecheck: install ## run ty and mypy type checking
	@typecheck_paths=""; \
	if [ -d "${SOURCE_FOLDER}" ]; then \
	  typecheck_paths="${SOURCE_FOLDER}"; \
	fi; \
	if [ -d ".rhiza/utils" ]; then \
	  typecheck_paths="$${typecheck_paths} .rhiza/utils"; \
	fi; \
	if [ -n "$${typecheck_paths}" ]; then \
	  printf "${BLUE}[INFO] Running ty type checking in:$${typecheck_paths}${RESET}\n"; \
	  ${UV_BIN} run ty check $${typecheck_paths} && \
	  printf "${BLUE}[INFO] Running mypy strict type checking in:$${typecheck_paths}${RESET}\n"; \
	  ${UV_BIN} run mypy --strict $${typecheck_paths}; \
	else \
	  printf "${YELLOW}[WARN] No typecheck folders found (SOURCE_FOLDER='${SOURCE_FOLDER}', .rhiza/utils missing), skipping typecheck${RESET}\n"; \
	fi

# Extra flags forwarded to pip-audit (e.g. --ignore-vuln CVE-XXXX-YYYY)
PIP_AUDIT_ARGS ?=

# The 'security' target performs security vulnerability scans.
# 1. Runs pip-audit via pip_audit_policy.py: fails on runtime dep CVEs, warns on tooling (pip/setuptools/wheel).
# 2. Runs bandit to find common security issues in Python source folders that exist.
security: install ## run security scans (pip-audit and bandit)
	@printf "${BLUE}[INFO] Running pip-audit for dependency vulnerabilities...${RESET}\n"
	@${UV_BIN} run python .rhiza/utils/pip_audit_policy.py ${PIP_AUDIT_ARGS}
	@bandit_paths=""; \
	if [ -d "${SOURCE_FOLDER}" ]; then \
	  bandit_paths="${SOURCE_FOLDER}"; \
	fi; \
	if [ -d ".rhiza/utils" ]; then \
	  bandit_paths="$${bandit_paths} .rhiza/utils"; \
	fi; \
	if [ -n "$${bandit_paths}" ]; then \
	  printf "${BLUE}[INFO] Running bandit security scan in:$${bandit_paths}${RESET}\n"; \
	  ${UVX_BIN} bandit -r $${bandit_paths} -ll -q --ini .bandit; \
	else \
	  printf "${YELLOW}[WARN] No bandit scan folders found (SOURCE_FOLDER='${SOURCE_FOLDER}', .rhiza/utils missing), skipping bandit${RESET}\n"; \
	fi

# The 'benchmark' target runs performance benchmarks using pytest-benchmark.
# 1. Installs benchmarking dependencies (pytest-benchmark, pygal).
# 2. Executes benchmarks found in the benchmarks/ subfolder.
# 3. Generates histograms and JSON results.
# 4. Runs a post-analysis script to process the results.
benchmark:: install ## run performance benchmarks
	@if [ -d "${TESTS_FOLDER}/benchmarks" ]; then \
	  printf "${BLUE}[INFO] Running performance benchmarks...${RESET}\n"; \
	  ${UV_BIN} pip install pytest-benchmark==5.2.3 pygal==3.1.0; \
	  mkdir -p _tests/benchmarks; \
	  ${UV_BIN} run pytest "${TESTS_FOLDER}/benchmarks/" \
	  		--benchmark-only \
			--benchmark-histogram=_tests/benchmarks/histogram \
			--benchmark-json=_tests/benchmarks/results.json; \
	  ${UVX_BIN} "rhiza-tools>=0.2.3" analyze-benchmarks --benchmarks-json _tests/benchmarks/results.json --output-html _tests/benchmarks/report.html; \
	else \
	  printf "${YELLOW}[WARN] Benchmarks folder not found, skipping benchmarks${RESET}\n"; \
	fi

# The 'docs-coverage' target checks documentation coverage using interrogate.
# 1. Builds a list of existing Python source folders to check.
# 2. Runs interrogate with verbose output against those folders.
docs-coverage: install ## check documentation coverage with interrogate
	@docstring_paths=""; \
	if [ -d "${SOURCE_FOLDER}" ]; then \
	  docstring_paths="${SOURCE_FOLDER}"; \
	fi; \
	if [ -d ".rhiza/utils" ]; then \
	  docstring_paths="$${docstring_paths} .rhiza/utils"; \
	fi; \
	if [ -d "tests" ]; then \
	  docstring_paths="$${docstring_paths} tests"; \
	fi; \
	if [ -d ".rhiza/tests" ]; then \
	  docstring_paths="$${docstring_paths} .rhiza/tests"; \
	fi; \
	if [ -n "$${docstring_paths}" ]; then \
	  printf "${BLUE}[INFO] Checking documentation coverage in:$${docstring_paths}${RESET}\n"; \
	  ${UV_BIN} run interrogate -vv --fail-under 100 --ignore-init-method --ignore-magic $${docstring_paths}; \
	else \
	  printf "${YELLOW}[WARN] No docs-coverage folders found (SOURCE_FOLDER='${SOURCE_FOLDER}', .rhiza/utils missing), skipping docs-coverage${RESET}\n"; \
	fi

# The 'hypothesis-test' target runs property-based tests using Hypothesis.
# 1. Checks if hypothesis tests exist in the tests directory.
# 2. Runs pytest with hypothesis-specific settings and statistics.
# 3. Generates detailed hypothesis examples and statistics.
hypothesis-test:: install ## run property-based tests with Hypothesis
	@if [ -z "$$(find ${TESTS_FOLDER} -name 'test_*.py' -o -name '*_test.py' 2>/dev/null)" ]; then \
	  printf "${YELLOW}[WARN] No test files found in ${TESTS_FOLDER}, skipping hypothesis tests.${RESET}\n"; \
	  exit 0; \
	fi; \
	printf "${BLUE}[INFO] Running Hypothesis property-based tests...${RESET}\n"; \
	mkdir -p _tests/hypothesis; \
	PYTEST_HTML_TITLE="Hypothesis tests" ${UV_BIN} run pytest \
	  --ignore=${TESTS_FOLDER}/benchmarks \
	  -v \
	  --hypothesis-show-statistics \
	  --hypothesis-seed=0 \
	  -m "hypothesis or property" \
	  --tb=short \
	  --html=_tests/hypothesis/report.html; \
	exit_code=$$?; \
	if [ $$exit_code -eq 5 ]; then \
	  printf "${YELLOW}[WARN] No hypothesis/property tests collected, skipping.${RESET}\n"; \
	  exit 0; \
	fi; \
	exit $$exit_code

# The 'stress' target runs stress/load tests.
# 1. Checks if stress tests exist in the tests/stress directory.
# 2. Runs pytest with the stress marker to execute only stress tests.
# 3. Generates an HTML report of stress test results.
stress:: install ## run stress/load tests
	@if [ ! -d "${TESTS_FOLDER}/stress" ]; then \
	  printf "${YELLOW}[WARN] Stress tests folder not found, skipping stress tests.${RESET}\n"; \
	  exit 0; \
	fi; \
	printf "${BLUE}[INFO] Running stress/load tests...${RESET}\n"; \
	mkdir -p _tests/stress; \
	${UV_BIN} run pytest \
	  -v \
	  -m stress \
	  --tb=short \
	  --html=_tests/stress/report.html

mutation: install ## run mutation tests with mutmut
	@if [ ! -d ${SOURCE_FOLDER} ]; then \
	  printf "${YELLOW}[WARN] Source folder ${SOURCE_FOLDER} not found, skipping mutation tests.${RESET}\n"; \
	  exit 0; \
	fi; \
	printf "${BLUE}[INFO] Running mutation tests on ${SOURCE_FOLDER}...${RESET}\n"; \
	mkdir -p _tests/mutation; \
	run_status=0; \
	${UV_BIN} run mutmut run \
	  --paths-to-mutate="${SOURCE_FOLDER}" \
	  --tests-dir="${TESTS_FOLDER}" || run_status=$$?; \
	${UV_BIN} run mutmut html || exit $$?; \
	rm -rf _tests/mutation/html; \
	mv html _tests/mutation/html || exit $$?; \
	${UV_BIN} run mutmut results || exit $$?; \
	exit $$run_status

test-pyproject: install ## run pyproject.toml structure tests
	@${UV_BIN} run pytest .rhiza/tests/structure/test_pyproject.py \
		-v \
		--tb=long \
		--showlocals \
		-rA \
		--durations=0 \
		--no-header
