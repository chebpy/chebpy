## Makefile.tests - Testing and benchmarking targets
# This file is included by the main Makefile.
# It provides targets for running the test suite with coverage and
# executing performance benchmarks.

# Declare phony targets (they don't produce files)
.PHONY: test benchmark typecheck security docs-coverage

# Default directory for tests
TESTS_FOLDER := tests

# Minimum coverage percent for tests to pass
# (Can be overridden in local.mk or via environment variable)
COVERAGE_FAIL_UNDER ?= 90

##@ Development and Testing

# The 'test' target runs the complete test suite.
# 1. Cleans up any previous test results in _tests/.
# 2. Creates directories for HTML coverage and test reports.
# 3. Invokes pytest via the local virtual environment.
# 4. Generates terminal output, HTML coverage, JSON coverage, and HTML test reports.
test: install ## run all tests
	@rm -rf _tests;

	@mkdir -p _tests/html-coverage _tests/html-report; \
	if [ -d ${SOURCE_FOLDER} ]; then \
	  ${VENV}/bin/python -m pytest \
	  --ignore=${TESTS_FOLDER}/benchmarks \
	  --cov=${SOURCE_FOLDER} \
	  --cov-report=term \
	  --cov-report=html:_tests/html-coverage \
	  --cov-fail-under=$(COVERAGE_FAIL_UNDER) \
	  --cov-report=json:_tests/coverage.json \
	  --html=_tests/html-report/report.html; \
	else \
	  printf "${YELLOW}[WARN] Source folder ${SOURCE_FOLDER} not found, running tests without coverage${RESET}\n"; \
	  ${VENV}/bin/python -m pytest \
	  --ignore=${TESTS_FOLDER}/benchmarks \
	  --html=_tests/html-report/report.html; \
	fi

# The 'typecheck' target runs static type analysis using mypy.
# 1. Checks if the source directory exists.
# 2. Runs mypy on the source folder using the configuration in pyproject.toml.
typecheck: install ## run mypy type checking
	@if [ -d ${SOURCE_FOLDER} ]; then \
	  printf "${BLUE}[INFO] Running mypy type checking...${RESET}\n"; \
	  ${UV_BIN} run mypy ${SOURCE_FOLDER} --config-file pyproject.toml; \
	else \
	  printf "${YELLOW}[WARN] Source folder ${SOURCE_FOLDER} not found, skipping typecheck${RESET}\n"; \
	fi

# The 'security' target performs security vulnerability scans.
# 1. Runs pip-audit to check for known vulnerabilities in dependencies.
# 2. Runs bandit to find common security issues in the source code.
security: install ## run security scans (pip-audit and bandit)
	@printf "${BLUE}[INFO] Running pip-audit for dependency vulnerabilities...${RESET}\n"
	@${UVX_BIN} pip-audit
	@printf "${BLUE}[INFO] Running bandit security scan...${RESET}\n"
	@${UVX_BIN} bandit -r ${SOURCE_FOLDER} -ll -q

# The 'benchmark' target runs performance benchmarks using pytest-benchmark.
# 1. Installs benchmarking dependencies (pytest-benchmark, pygal).
# 2. Executes benchmarks found in the benchmarks/ subfolder.
# 3. Generates histograms and JSON results.
# 4. Runs a post-analysis script to process the results.
benchmark: install ## run performance benchmarks
	@if [ -d "${TESTS_FOLDER}/benchmarks" ]; then \
	  printf "${BLUE}[INFO] Running performance benchmarks...${RESET}\n"; \
	  ${UV_BIN} pip install pytest-benchmark==5.2.3 pygal==3.1.0; \
	  ${VENV}/bin/python -m pytest "${TESTS_FOLDER}/benchmarks/" \
	  		--benchmark-only \
			--benchmark-histogram=tests/test_rhiza/benchmarks/benchmarks \
			--benchmark-json=tests/test_rhiza/benchmarks/benchmarks.json; \
	  ${VENV}/bin/python tests/test_rhiza/benchmarks/analyze_benchmarks.py ; \
	else \
	  printf "${YELLOW}[WARN] Benchmarks folder not found, skipping benchmarks${RESET}\n"; \
	fi

# The 'docs-coverage' target checks documentation coverage using interrogate.
# 1. Checks if SOURCE_FOLDER exists.
# 2. Runs interrogate on the source folder with verbose output.
docs-coverage: install ## check documentation coverage with interrogate
	@if [ -d "${SOURCE_FOLDER}" ]; then \
	  printf "${BLUE}[INFO] Checking documentation coverage in ${SOURCE_FOLDER}...${RESET}\n"; \
	  ${VENV}/bin/python -m interrogate -vv ${SOURCE_FOLDER}; \
	else \
	  printf "${YELLOW}[WARN] Source folder ${SOURCE_FOLDER} not found, skipping docs-coverage${RESET}\n"; \
	fi

