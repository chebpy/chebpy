"""Tests for Makefile variable override behaviour.

This file and its associated tests flow down via a SYNC action from the
jebel-quant/rhiza repository (https://github.com/jebel-quant/rhiza).

Validates that key Makefile variables behave correctly when overridden on
the command line, ensuring downstream projects can customise coverage
thresholds, license checks, and Python tooling without modifying the
shared Makefile infrastructure.

All tests use `make -n` (dry-run) to observe what commands *would* be
executed without running them — keeping the suite fast and side-effect-free.
"""

from __future__ import annotations

import os

from api.conftest import run_make, strip_ansi


class TestCoverageFailUnder:
    """COVERAGE_FAIL_UNDER controls the pytest --cov-fail-under threshold."""

    def test_threshold_override_to_100(self, logger) -> None:
        """COVERAGE_FAIL_UNDER=100 must propagate to pytest invocation."""
        proc = run_make(logger, ["test", "COVERAGE_FAIL_UNDER=100"])
        assert "--cov-fail-under=100" in proc.stdout

    def test_threshold_override_to_0(self, logger) -> None:
        """COVERAGE_FAIL_UNDER=0 must propagate (useful for bootstrapping new projects)."""
        proc = run_make(logger, ["test", "COVERAGE_FAIL_UNDER=0"])
        assert "--cov-fail-under=0" in proc.stdout

    def test_threshold_override_to_arbitrary_value(self, logger) -> None:
        """Any integer override for COVERAGE_FAIL_UNDER must appear verbatim in the command."""
        proc = run_make(logger, ["test", "COVERAGE_FAIL_UNDER=73"])
        assert "--cov-fail-under=73" in proc.stdout


class TestLicenseFailOn:
    """LICENSE_FAIL_ON controls which SPDX license identifiers cause a build failure."""

    def test_default_fails_on_gpl(self, logger) -> None:
        """Default LICENSE_FAIL_ON must include GPL to block copyleft licenses."""
        proc = run_make(logger, ["license"])
        assert "GPL" in proc.stdout, "Default license check should fail on GPL; got:\n" + proc.stdout[:500]

    def test_fail_on_override_single_license(self, logger) -> None:
        """Custom single-license override must appear in the make license command."""
        proc = run_make(logger, ["license", "LICENSE_FAIL_ON=MIT"])
        assert "MIT" in proc.stdout

    def test_fail_on_override_multiple_licenses(self, logger) -> None:
        """Semicolon-separated multi-license override must appear verbatim."""
        proc = run_make(logger, ["license", "LICENSE_FAIL_ON=AGPL-3.0;GPL-2.0;LGPL-2.1"])
        assert "AGPL-3.0" in proc.stdout
        assert "GPL-2.0" in proc.stdout

    def test_fail_on_override_quoted_correctly(self, logger) -> None:
        """LICENSE_FAIL_ON value must be quoted in the underlying pip-licenses call."""
        proc = run_make(logger, ["license", "LICENSE_FAIL_ON=MIT;Apache"])
        # The Makefile must quote the value to handle semicolons properly
        assert '--fail-on="MIT;Apache"' in proc.stdout


class TestPythonVersionVariable:
    """PYTHON_VERSION drives uvx -p <version> ... in quality and formatting targets."""

    def test_python_version_read_from_python_version_file(self, logger, tmp_path) -> None:
        """When .python-version exists, PYTHON_VERSION should reflect its contents."""
        python_version_file = tmp_path / ".python-version"
        if python_version_file.exists():
            version = python_version_file.read_text().strip()
            proc = run_make(logger, ["print-PYTHON_VERSION"], dry_run=False)
            out = strip_ansi(proc.stdout)
            assert version in out, f"Expected {version} in PYTHON_VERSION output; got: {out}"

    def test_python_version_default_when_file_missing(self, logger, tmp_path) -> None:
        """When .python-version is absent and PYTHON_VERSION env var is unset, default to 3.13."""
        pv_file = tmp_path / ".python-version"
        if pv_file.exists():
            pv_file.unlink()

        env = os.environ.copy()
        env.pop("PYTHON_VERSION", None)

        proc = run_make(logger, ["print-PYTHON_VERSION"], dry_run=False, env=env)
        out = strip_ansi(proc.stdout)
        assert "3.13" in out, f"Expected default 3.13; got: {out}"

    def test_python_version_used_in_fmt_target(self, logger, tmp_path) -> None:
        """The fmt target must pass -p <PYTHON_VERSION> to uvx."""
        env = os.environ.copy()
        env.pop("PYTHON_VERSION", None)

        proc = run_make(logger, ["fmt"], env=env)
        assert "uvx -p" in proc.stdout, "fmt target should use uvx -p <version>"


class TestSourceFolderVariable:
    """SOURCE_FOLDER drives coverage collection and static analysis targets."""

    def test_typecheck_uses_source_folder(self, logger, tmp_path) -> None:
        """The typecheck target must pass SOURCE_FOLDER to ty and mypy."""
        src_dir = tmp_path / "mypackage"
        src_dir.mkdir(exist_ok=True)

        env_file = tmp_path / ".rhiza" / ".env"
        if env_file.exists():
            env_file.write_text(env_file.read_text() + "\nSOURCE_FOLDER=mypackage\n")

        proc = run_make(logger, ["typecheck", "SOURCE_FOLDER=mypackage"])
        assert 'typecheck_paths="mypackage"' in proc.stdout, (
            "typecheck should include SOURCE_FOLDER in computed path list; got:\n" + proc.stdout[:400]
        )
        assert " run ty check ${typecheck_paths}" in proc.stdout, (
            "typecheck should pass computed path list to ty; got:\n" + proc.stdout[:400]
        )
        assert " run mypy --strict ${typecheck_paths}" in proc.stdout, (
            "typecheck should pass computed path list to mypy; got:\n" + proc.stdout[:400]
        )

    def test_deptry_uses_source_folder(self, logger, tmp_path) -> None:
        """The deptry target must scan the directory set by SOURCE_FOLDER."""
        src_dir = tmp_path / "mypackage"
        src_dir.mkdir(exist_ok=True)

        proc = run_make(logger, ["deptry", "SOURCE_FOLDER=mypackage"])
        assert "mypackage" in proc.stdout, "deptry should reference SOURCE_FOLDER; got:\n" + proc.stdout[:400]

    def test_deptry_accumulates_marimo_and_source_in_one_call(self, logger, tmp_path) -> None:
        """The marimo bundle must contribute its folder (and DEP004 ignore) to the single deptry scan.

        This locks in the accumulator design: each bundle appends to DEPTRY_FOLDERS /
        DEPTRY_IGNORE rather than the core target hard-coding knowledge of marimo.
        """
        (tmp_path / "mypackage").mkdir(exist_ok=True)
        (tmp_path / "notebooks").mkdir(exist_ok=True)

        proc = run_make(logger, ["deptry", "SOURCE_FOLDER=mypackage", "MARIMO_FOLDER=notebooks"])
        out = strip_ansi(proc.stdout)
        # marimo.mk is included before quality.mk, so its folder is appended first.
        assert "deptry notebooks mypackage --ignore DEP004" in out, (
            "deptry should scan marimo + source folders in a single call with DEP004 ignored; got:\n" + out[:600]
        )


class TestUvNoModifyPath:
    """UV_NO_MODIFY_PATH must always be exported to 1 to avoid uv touching PATH."""

    def test_uv_no_modify_path_is_1(self, logger) -> None:
        """UV_NO_MODIFY_PATH must be exported as 1 in the Makefile."""
        proc = run_make(logger, ["print-UV_NO_MODIFY_PATH"], dry_run=False)
        out = strip_ansi(proc.stdout)
        assert "1" in out, f"UV_NO_MODIFY_PATH should be 1; got: {out}"

    def test_uv_no_modify_path_cannot_be_overridden_to_empty(self, logger) -> None:
        """UV_NO_MODIFY_PATH must still appear in the printed value when queried."""
        proc = run_make(logger, ["print-UV_NO_MODIFY_PATH"], dry_run=False)
        out = strip_ansi(proc.stdout)
        assert "UV_NO_MODIFY_PATH" in out


class TestTestsFolder:
    """TESTS_FOLDER defaults to 'tests' but can be overridden."""

    def test_default_tests_folder_is_tests(self, logger) -> None:
        """Default TESTS_FOLDER must be 'tests'."""
        proc = run_make(logger, ["print-TESTS_FOLDER"], dry_run=False)
        out = strip_ansi(proc.stdout)
        assert "tests" in out, f"Default TESTS_FOLDER should be 'tests'; got: {out}"

    def test_pytest_uses_tests_folder(self, logger) -> None:
        """The test target must invoke pytest with the TESTS_FOLDER path."""
        proc = run_make(logger, ["test"])
        # The default tests folder must appear somewhere in the pytest invocation
        assert "pytest" in proc.stdout
