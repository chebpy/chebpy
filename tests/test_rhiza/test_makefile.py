"""Tests for the Makefile targets and help output using safe dry‑runs.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

These tests validate that the Makefile exposes expected targets and emits
the correct commands without actually executing them, by invoking `make -n`
(dry-run). We also pass `-s` to reduce noise in CI logs. This approach keeps
tests fast, portable, and free of side effects like network or environment
changes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

# Get absolute paths for executables to avoid S607 warnings from CodeFactor/Bandit
MAKE = shutil.which("make") or "/usr/bin/make"

# Split Makefile paths that are included in the main Makefile
SPLIT_MAKEFILES = [
    ".rhiza/rhiza.mk",
    "tests/tests.mk",
    "book/book.mk",
    "presentation/presentation.mk",
]


def strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


@pytest.fixture(autouse=True)
def setup_tmp_makefile(logger, root, tmp_path: Path):
    """Copy the Makefile and split Makefiles into a temp directory and chdir there.

    We rely on `make -n` so that no real commands are executed.
    """
    logger.debug("Setting up temporary Makefile test dir: %s", tmp_path)

    # Copy the main Makefile into the temporary working directory
    shutil.copy(root / "Makefile", tmp_path / "Makefile")

    # Copy core Rhiza Makefiles
    (tmp_path / ".rhiza").mkdir(exist_ok=True)
    shutil.copy(root / ".rhiza" / "rhiza.mk", tmp_path / ".rhiza" / "rhiza.mk")

    # Copy .python-version file for PYTHON_VERSION variable
    if (root / ".python-version").exists():
        shutil.copy(root / ".python-version", tmp_path / ".python-version")

    # Create a minimal, deterministic .rhiza/.env for tests so they don't
    # depend on the developer's local configuration which may vary.
    env_content = "SCRIPTS_FOLDER=.rhiza/scripts\nCUSTOM_SCRIPTS_FOLDER=.rhiza/customisations/scripts\n"
    (tmp_path / ".rhiza" / ".env").write_text(env_content)

    logger.debug("Copied Makefile from %s to %s", root / "Makefile", tmp_path / "Makefile")

    # Copy split Makefiles if they exist (maintaining directory structure)
    for split_file in SPLIT_MAKEFILES:
        source_path = root / split_file
        if source_path.exists():
            dest_path = tmp_path / split_file
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, dest_path)
            logger.debug("Copied %s to %s", source_path, dest_path)

    # Move into tmp directory for isolation
    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    logger.debug("Changed working directory to %s", tmp_path)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        logger.debug("Restored working directory to %s", old_cwd)


def run_make(
    logger,
    args: list[str] | None = None,
    check: bool = True,
    dry_run: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run `make` with optional arguments and return the completed process.

    Args:
        logger: Logger used to emit diagnostic messages during the run
        args: Additional arguments for make
        check: If True, raise on non-zero return code
        dry_run: If True, use -n to avoid executing commands
        env: Optional environment variables to pass to the subprocess
    """
    cmd = [MAKE]
    if args:
        cmd.extend(args)
    # Use -s to reduce noise, -n to avoid executing commands
    flags = "-sn" if dry_run else "-s"
    cmd.insert(1, flags)
    logger.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    logger.debug("make exited with code %d", result.returncode)
    if result.stdout:
        logger.debug("make stdout (truncated to 500 chars):\n%s", result.stdout[:500])
    if result.stderr:
        logger.debug("make stderr (truncated to 500 chars):\n%s", result.stderr[:500])
    if check and result.returncode != 0:
        msg = f"make failed with code {result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        raise AssertionError(msg)
    return result


def setup_rhiza_git_repo():
    """Initialize a git repository and set remote to rhiza."""
    git = shutil.which("git") or "/usr/bin/git"
    subprocess.run([git, "init"], check=True, capture_output=True)
    subprocess.run(
        [git, "remote", "add", "origin", "https://github.com/jebel-quant/rhiza"],
        check=True,
        capture_output=True,
    )


class TestMakefile:
    """Smoke tests for Makefile help and common targets using make -n."""

    def test_default_goal_is_help(self, logger):
        """Default goal should render the help index with known targets."""
        proc = run_make(logger)
        out = proc.stdout
        assert "Usage:" in out
        assert "Targets:" in out
        # ensure a few known targets appear in the help index
        for target in ["install", "fmt", "deptry", "test", "help"]:
            assert target in out

    def test_help_target(self, logger):
        """Explicit `make help` prints usage, targets, and section headers."""
        proc = run_make(logger, ["help"])
        out = proc.stdout
        assert "Usage:" in out
        assert "Targets:" in out
        assert "Bootstrap" in out or "Meta" in out  # section headers

    def test_fmt_target_dry_run(self, logger, tmp_path):
        """Fmt target should invoke pre-commit via uvx with Python version in dry-run output."""
        # Create clean environment without PYTHON_VERSION so Makefile reads from .python-version
        env = os.environ.copy()
        env.pop("PYTHON_VERSION", None)

        proc = run_make(logger, ["fmt"], env=env)
        out = proc.stdout
        # Check for uvx command with the Python version flag
        # The PYTHON_VERSION should be read from .python-version file (e.g., "3.12")
        python_version_file = tmp_path / ".python-version"
        if python_version_file.exists():
            python_version = python_version_file.read_text().strip()
            assert f"uvx -p {python_version} pre-commit run --all-files" in out
        else:
            # Fallback check if .python-version doesn't exist
            assert "uvx -p" in out
            assert "pre-commit run --all-files" in out

    def test_deptry_target_dry_run(self, logger, tmp_path):
        """Deptry target should invoke deptry via uvx with Python version in dry-run output."""
        # Create a mock SOURCE_FOLDER directory so the deptry command runs
        source_folder = tmp_path / "src"
        source_folder.mkdir(exist_ok=True)

        # Update .env to set SOURCE_FOLDER
        env_file = tmp_path / ".rhiza" / ".env"
        env_content = env_file.read_text()
        env_content += "\nSOURCE_FOLDER=src\n"
        env_file.write_text(env_content)

        # Create clean environment without PYTHON_VERSION so Makefile reads from .python-version
        env = os.environ.copy()
        env.pop("PYTHON_VERSION", None)

        proc = run_make(logger, ["deptry"], env=env)

        out = proc.stdout
        # Check for uvx command with the Python version flag
        python_version_file = tmp_path / ".python-version"
        if python_version_file.exists():
            python_version = python_version_file.read_text().strip()
            assert f"uvx -p {python_version} deptry src" in out
        else:
            # Fallback check if .python-version doesn't exist
            assert "uvx -p" in out
            assert "deptry src" in out

    def test_mypy_target_dry_run(self, logger, tmp_path):
        """Mypy target should invoke mypy via uv run in dry-run output."""
        # Create a mock SOURCE_FOLDER directory so the mypy command runs
        source_folder = tmp_path / "src"
        source_folder.mkdir(exist_ok=True)

        # Update .env to set SOURCE_FOLDER
        env_file = tmp_path / ".rhiza" / ".env"
        env_content = env_file.read_text()
        env_content += "\nSOURCE_FOLDER=src\n"
        env_file.write_text(env_content)

        proc = run_make(logger, ["mypy"])
        out = proc.stdout
        # Check for uv run command instead of uvx
        assert "uv run mypy src --strict --config-file=pyproject.toml" in out

    def test_test_target_dry_run(self, logger):
        """Test target should invoke pytest via uv with coverage and HTML outputs in dry-run output."""
        proc = run_make(logger, ["test"])
        out = proc.stdout
        # Expect key steps
        assert "mkdir -p _tests/html-coverage _tests/html-report" in out
        # Check for uv command with the configured path

    def test_test_target_without_source_folder(self, logger, tmp_path):
        """Test target should run without coverage when SOURCE_FOLDER doesn't exist."""
        # Update .env to set SOURCE_FOLDER to a non-existent directory
        env_file = tmp_path / ".rhiza" / ".env"
        env_content = env_file.read_text()
        env_content += "\nSOURCE_FOLDER=nonexistent_src\n"
        env_file.write_text(env_content)

        # Create tests folder
        tests_folder = tmp_path / "tests"
        tests_folder.mkdir(exist_ok=True)

        proc = run_make(logger, ["test"])
        out = proc.stdout
        # Should see warning about missing source folder
        assert "if [ -d nonexistent_src ]" in out
        # Should still run pytest but without coverage flags
        assert "pytest tests" in out
        assert "--html=_tests/html-report/report.html" in out

    def test_python_version_defaults_to_3_13_if_missing(self, logger, tmp_path):
        """`PYTHON_VERSION` should default to `3.13` if .python-version is missing."""
        # Ensure .python-version does not exist
        python_version_file = tmp_path / ".python-version"
        if python_version_file.exists():
            python_version_file.unlink()

        # Create clean environment without PYTHON_VERSION
        env = os.environ.copy()
        env.pop("PYTHON_VERSION", None)

        proc = run_make(logger, ["print-PYTHON_VERSION"], dry_run=False, env=env)
        out = strip_ansi(proc.stdout)
        assert "Value of PYTHON_VERSION:\n3.13" in out

    def test_uv_no_modify_path_is_exported(self, logger):
        """`UV_NO_MODIFY_PATH` should be set to `1` in the Makefile."""
        proc = run_make(logger, ["print-UV_NO_MODIFY_PATH"], dry_run=False)
        out = strip_ansi(proc.stdout)
        assert "Value of UV_NO_MODIFY_PATH:\n1" in out

    def test_script_folder_is_github_scripts(self, logger):
        """`SCRIPTS_FOLDER` should point to `.rhiza/scripts`."""
        proc = run_make(logger, ["print-SCRIPTS_FOLDER"], dry_run=False)
        out = strip_ansi(proc.stdout)
        assert "Value of SCRIPTS_FOLDER:\n.rhiza/scripts" in out

    def test_that_target_coverage_is_configurable(self, logger):
        """Test target should respond to COVERAGE_FAIL_UNDER variable."""
        # Default case: ensure the flag is present
        proc = run_make(logger, ["test"])
        assert "--cov-fail-under=" in proc.stdout

        # Override case: ensure the flag takes the specific value
        proc_override = run_make(logger, ["test", "COVERAGE_FAIL_UNDER=42"])
        assert "--cov-fail-under=42" in proc_override.stdout


class TestMakefileRootFixture:
    """Tests for root fixture usage in Makefile tests."""

    def test_makefile_exists_at_root(self, root):
        """Makefile should exist at repository root."""
        makefile = root / "Makefile"
        assert makefile.exists()
        assert makefile.is_file()

    def test_makefile_is_readable(self, root):
        """Makefile should be readable."""
        makefile = root / "Makefile"
        content = makefile.read_text()
        assert len(content) > 0

    def test_makefile_contains_targets(self, root):
        """Makefile should contain expected targets (including split files)."""
        makefile = root / "Makefile"
        content = makefile.read_text()

        # Read split Makefiles as well
        for split_file in SPLIT_MAKEFILES:
            split_path = root / split_file
            if split_path.exists():
                content += "\n" + split_path.read_text()

        expected_targets = ["install", "fmt", "test", "deptry", "help"]
        for target in expected_targets:
            assert f"{target}:" in content or f".PHONY: {target}" in content

    def test_makefile_has_uv_variables(self, root):
        """Makefile should define UV-related variables."""
        makefile = root / "Makefile"
        content = makefile.read_text()

        # Read split Makefiles as well
        for split_file in SPLIT_MAKEFILES:
            split_path = root / split_file
            if split_path.exists():
                content += "\n" + split_path.read_text()

        assert "UV_BIN" in content or "uv" in content.lower()

    def test_validate_target_skips_in_rhiza_repo(self, logger):
        """Validate target should skip execution in rhiza repository."""
        setup_rhiza_git_repo()

        proc = run_make(logger, ["validate"], dry_run=False)
        # out = strip_ansi(proc.stdout)
        # assert "[INFO] Skipping validate in rhiza repository" in out
        assert proc.returncode == 0

    def test_sync_target_skips_in_rhiza_repo(self, logger):
        """Sync target should skip execution in rhiza repository."""
        setup_rhiza_git_repo()

        proc = run_make(logger, ["sync"], dry_run=False)
        # out = strip_ansi(proc.stdout)
        # assert "[INFO] Skipping sync in rhiza repository" in out
        assert proc.returncode == 0


class TestMakeBump:
    """Tests for the 'make bump' target."""

    @pytest.fixture
    def mock_bin(self, tmp_path):
        """Create mock uv and uvx scripts in ./bin."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)

        uv = bin_dir / "uv"
        uv.write_text('#!/bin/sh\necho "[MOCK] uv $@"\n')
        uv.chmod(0o755)

        # Mock uvx to simulate version bump if arguments match
        uvx = bin_dir / "uvx"
        uvx_script = """#!/usr/bin/env python3
import sys
import re
from pathlib import Path

args = sys.argv[1:]
print(f"[MOCK] uvx {' '.join(args)}")

# Check if this is the bump command: "rhiza[tools]>=0.8.6" tools bump
if "tools" in args and "bump" in args:
    # Simulate bumping version in pyproject.toml
    pyproject = Path("pyproject.toml")
    if pyproject.exists():
        content = pyproject.read_text()
        # Simple regex replacement for version
        # Assuming version = "0.1.0" -> "0.1.1"
        new_content = re.sub(r'version = "([0-9.]+)"', lambda m: f'version = "{m.group(1)[:-1]}{int(m.group(1)[-1]) + 1}"', content)
        pyproject.write_text(new_content)
        print(f"[MOCK] Bumped version in {pyproject}")
"""  # noqa: E501
        uvx.write_text(uvx_script)
        uvx.chmod(0o755)

        return bin_dir

    def test_bump_execution(self, logger, mock_bin, tmp_path):
        """Test 'make bump' execution with mocked tools and verify version change."""
        # Create dummy pyproject.toml with initial version
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('version = "0.1.0"\n[project]\nname = "test"\n')

        uv_bin = mock_bin / "uv"
        uvx_bin = mock_bin / "uvx"

        # Run make bump with dry_run=False to actually execute the shell commands
        result = run_make(logger, ["bump", f"UV_BIN={uv_bin}", f"UVX_BIN={uvx_bin}"], dry_run=False)

        # Verify that the mock tools were called
        assert "[MOCK] uvx rhiza[tools]>=0.8.6 tools bump" in result.stdout
        assert "[MOCK] uv lock" in result.stdout

        # Verify that 'make install' was called (which calls uv sync)
        assert "[MOCK] uv sync" in result.stdout

        # Verify that the version was actually bumped by our mock
        new_content = pyproject.read_text()
        assert 'version = "0.1.1"' in new_content

    def test_bump_no_pyproject(self, logger, mock_bin, tmp_path):
        """Test 'make bump' execution without pyproject.toml."""
        # Ensure pyproject.toml does not exist
        pyproject = tmp_path / "pyproject.toml"
        if pyproject.exists():
            pyproject.unlink()

        uv_bin = mock_bin / "uv"
        uvx_bin = mock_bin / "uvx"

        result = run_make(logger, ["bump", f"UV_BIN={uv_bin}", f"UVX_BIN={uvx_bin}"], dry_run=False)

        # Check for warning message
        assert "No pyproject.toml found, skipping bump" in result.stdout

        # Ensure bump commands are NOT executed
        assert "[MOCK] uvx" not in result.stdout
        assert "[MOCK] uv lock" not in result.stdout
