"""Integration tests for .rhiza/make.d/test.mk.

Covers the missing-test-files short circuit plus the parallel-teardown
robustness behaviour (retry on pytest exit code 3, no retry on real failures,
stale coverage cleanup) added for issues #1256 and #1257.
"""

from pathlib import Path

from test_utils import run_make

# A stand-in for `uv` that records each invocation and exits with the next code
# from FAKE_EXIT_CODES (space-separated, last value repeats). Lets the tests
# drive the test.mk retry loop deterministically without invoking real pytest.
FAKE_UV_SCRIPT = r"""#!/bin/sh
count_file="$FAKE_COUNT_FILE"
n=$(cat "$count_file" 2>/dev/null || echo 0)
n=$((n + 1))
printf '%s' "$n" > "$count_file"
i=0
code=0
for c in $FAKE_EXIT_CODES; do
  i=$((i + 1))
  code=$c
  [ "$i" -ge "$n" ] && break
done
echo "fake-uv invocation $n -> exit $code (args: $*)"
exit "$code"
"""

# Minimal Makefile that exercises the `test` target in isolation: real test.mk,
# mocked install, fake uv, and the colour/path variables the recipe expects.
_TEST_MAKEFILE = r"""
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

TESTS_FOLDER := tests
SOURCE_FOLDER := src
UV_BIN := {uv_bin}

install:
	@echo "Mock install"

include .rhiza/make.d/test.mk
"""


def _setup_test_target(git_repo: Path) -> Path:
    """Wire up the fake uv and a non-empty tests folder; return the invocation-count file."""
    uv_bin = git_repo / "fake-uv"
    uv_bin.write_text(FAKE_UV_SCRIPT, encoding="utf-8")
    uv_bin.chmod(0o755)

    (git_repo / "Makefile").write_text(_TEST_MAKEFILE.format(uv_bin=uv_bin), encoding="utf-8")

    tests_dir = git_repo / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test_placeholder.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    count_file = git_repo / "uv_invocations"
    return count_file


def _run_test_target(logger, count_file: Path, exit_codes: str):
    """Run `make test` with the fake uv configured to return ``exit_codes``."""
    import os

    env = os.environ.copy()
    env["FAKE_EXIT_CODES"] = exit_codes
    env["FAKE_COUNT_FILE"] = str(count_file)
    return run_make(logger, ["test"], check=False, dry_run=False, env=env)


def test_retry_on_internal_error_then_passes(git_repo, logger):
    """Exit code 3 (xdist/teardown internal error) is retried once and a clean rerun passes (#1256)."""
    count_file = _setup_test_target(git_repo)
    result = _run_test_target(logger, count_file, "3 0")

    assert result.returncode == 0, f"retry should recover a teardown crash; got {result.returncode}\n{result.stdout}"
    assert count_file.read_text() == "2", "pytest should have been invoked exactly twice (initial + one retry)"
    assert "retrying suite" in result.stdout


def test_persistent_internal_error_fails(git_repo, logger):
    """A repeated exit code 3 fails after the bounded retry rather than looping forever.

    make collapses any recipe failure to its own exit code 2, so the propagated
    pytest exit code is asserted via the ``Error 3`` line make prints to stderr.
    """
    count_file = _setup_test_target(git_repo)
    result = _run_test_target(logger, count_file, "3 3 3")

    assert result.returncode != 0, "persistent internal error must fail the target"
    assert "Error 3" in result.stderr, f"recipe should propagate pytest exit 3; stderr:\n{result.stderr}"
    assert count_file.read_text() == "2", "retry must be bounded to two attempts total"
    assert "internal (teardown) error" in result.stdout


def test_real_failure_is_not_retried(git_repo, logger):
    """A genuine test failure (exit 1) fails immediately without a retry, so flaky tests are not masked."""
    count_file = _setup_test_target(git_repo)
    result = _run_test_target(logger, count_file, "1 0")

    assert result.returncode != 0, "a real test failure must fail the target"
    assert "Error 1" in result.stderr, f"recipe should propagate pytest exit 1 unretried; stderr:\n{result.stderr}"
    assert count_file.read_text() == "1", "pytest should have been invoked exactly once on a real failure"
    assert "retrying suite" not in result.stdout


def test_stale_coverage_is_removed_before_run(git_repo, logger):
    """A corrupt .coverage left by a previous crashed run is cleaned before pytest runs (#1257)."""
    count_file = _setup_test_target(git_repo)
    stale = git_repo / ".coverage"
    stale.write_text("corrupt-not-a-sqlite-db", encoding="utf-8")

    result = _run_test_target(logger, count_file, "0")

    assert result.returncode == 0, result.stdout
    assert not stale.exists(), "stale .coverage must be removed before the run to avoid a false 0% coverage"


def test_missing_tests_warning(git_repo, logger):
    """Test that missing tests trigger a warning but do not fail (exit 0)."""
    # 1. Setup a minimal Makefile in the test repo
    # We include .rhiza/make.d/test.mk but mock the 'install' dependency
    # and provide color variables used in the script.
    makefile_content = r"""
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Define folders expected by test.mk
TESTS_FOLDER := tests
SOURCE_FOLDER := src
VENV := .venv

# Mock install to avoid actual installation in test
install:
	@echo "Mock install"

# Include the target under test
include .rhiza/make.d/test.mk
"""
    (git_repo / "Makefile").write_text(makefile_content, encoding="utf-8")

    # 2. Ensure 'tests' folder exists but is empty/has no python test files
    tests_dir = git_repo / "tests"
    if tests_dir.exists():
        import shutil

        shutil.rmtree(tests_dir)
    tests_dir.mkdir()

    # 3. Run 'make test'
    # We use dry_run=False so the shell commands in the recipe actually execute.
    # The 'check=False' allows us to assert the return code ourselves,
    # though we expect 0 now.
    result = run_make(logger, ["test"], check=False, dry_run=False)

    # 4. output for debugging
    logger.info("make stdout: %s", result.stdout)
    logger.info("make stderr: %s", result.stderr)

    # 5. Verify results
    assert result.returncode == 0, "make test should exit with 0 when no tests found"

    # The warning message matches what we put in test.mk
    # "No test files found in {TESTS_FOLDER}, skipping tests"
    assert "No test files found in tests, skipping tests" in result.stdout
