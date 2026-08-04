## .rhiza/make.d/test.mk - optional Python testing extras (bundle: tests)
# This file is included by the main Makefile.
#
# The gates the language contract names — `test`, `typecheck`, `security`,
# `docs-coverage` — moved into `python.mk` in #1475, because `python.mk`'s own `all`
# named them while this bundle owned them: a project syncing `core + python-core`
# without `tests` got an `all` that could not run. The Rust and Go layers never had
# that split, and now neither does Python.
#
# What is left here is what genuinely *is* optional — four gates no layer's `all`
# depends on, each of which needs its own tool and its own folder convention:
# benchmarks, Hypothesis property tests, stress/load tests and mutation testing.
# Keeping them out of the layer is what lets a project take the Python gate set
# without also declaring an opinion on mutation testing.
#
# `TESTS_FOLDER`, and the shared pytest config in `pytest.ini`, come from
# `python.mk` / `python-core`. This bundle requires `python-core`, so both are always
# present alongside it.

# Declare phony targets (they don't produce files)
.PHONY: benchmark hypothesis-test stress mutation

##@ Development and Testing (extras)

# The 'benchmark' target runs performance benchmarks using pytest-benchmark.
# 1. Installs benchmarking dependencies (pytest-benchmark, pygal).
# 2. Executes benchmarks found in the benchmarks/ subfolder.
# 3. Generates histograms and JSON results.
benchmark:: install ## run performance benchmarks
	@if [ -d "${TESTS_FOLDER}/benchmarks" ]; then \
	  printf "${BLUE}[INFO] Running performance benchmarks...${RESET}\n"; \
	  mkdir -p _tests/benchmarks; \
	  ${UV_BIN} run --with pytest --with pytest-benchmark==5.2.3 --with pygal==3.1.0 pytest "${TESTS_FOLDER}/benchmarks/" \
	  		--benchmark-only \
			--benchmark-histogram=_tests/benchmarks/histogram \
			--benchmark-json=_tests/benchmarks/results.json; \
	else \
	  printf "${YELLOW}[WARN] Benchmarks folder not found, skipping benchmarks${RESET}\n"; \
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
	PYTEST_HTML_TITLE="Hypothesis tests" ${UV_BIN} run --with pytest --with hypothesis --with pytest-html pytest \
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
	${UV_BIN} run --with pytest --with pytest-html pytest \
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
	${UV_BIN} run --with mutmut mutmut run \
	  --paths-to-mutate="${SOURCE_FOLDER}" \
	  --tests-dir="${TESTS_FOLDER}" || run_status=$$?; \
	${UV_BIN} run --with mutmut mutmut html || exit $$?; \
	rm -rf _tests/mutation/html; \
	mv html _tests/mutation/html || exit $$?; \
	${UV_BIN} run --with mutmut mutmut results || exit $$?; \
	exit $$run_status
