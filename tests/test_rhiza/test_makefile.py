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


def strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


@pytest.fixture
def expected_uv_install_dir() -> str:
    """Get the expected UV_INSTALL_DIR from environment or default to ./bin."""
    return os.environ.get("UV_INSTALL_DIR", "./bin")


@pytest.fixture(autouse=True)
def setup_tmp_makefile(logger, root, tmp_path: Path):
    """Copy only the Makefile into a temp directory and chdir there.

    We rely on `make -n` so that no real commands are executed.
    """
    logger.debug("Setting up temporary Makefile test dir: %s", tmp_path)

    # Copy the Makefile into the temporary working directory
    shutil.copy(root / "Makefile", tmp_path / "Makefile")

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
    logger, args: list[str] | None = None, check: bool = True, dry_run: bool = True
) -> subprocess.CompletedProcess:
    """Run `make` with optional arguments and return the completed process.

    Args:
        logger: Logger used to emit diagnostic messages during the run
        args: Additional arguments for make
        check: If True, raise on non-zero return code
        dry_run: If True, use -n to avoid executing commands
    """
    cmd = ["make"]
    if args:
        cmd.extend(args)
    # Use -s to reduce noise, -n to avoid executing commands
    flags = "-sn" if dry_run else "-s"
    cmd.insert(1, flags)
    logger.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    logger.debug("make exited with code %d", result.returncode)
    if result.stdout:
        logger.debug("make stdout (truncated to 500 chars):\n%s", result.stdout[:500])
    if result.stderr:
        logger.debug("make stderr (truncated to 500 chars):\n%s", result.stderr[:500])
    if check and result.returncode != 0:
        msg = f"make failed with code {result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        raise AssertionError(msg)
    return result


class TestMakefile:
    """Smoke tests for Makefile help and common targets using make -n."""

    def test_default_goal_is_help(self, logger):
        """Default goal should render the help index with known targets."""
        proc = run_make(logger)
        out = proc.stdout
        assert "Usage:" in out
        assert "Targets:" in out
        # ensure a few known targets appear in the help index
        for target in ["install", "fmt", "deptry", "test", "book", "help"]:
            assert target in out

    def test_help_target(self, logger):
        """Explicit `make help` prints usage, targets, and section headers."""
        proc = run_make(logger, ["help"])
        out = proc.stdout
        assert "Usage:" in out
        assert "Targets:" in out
        assert "Bootstrap" in out or "Meta" in out  # section headers

    def test_fmt_target_dry_run(self, logger, expected_uv_install_dir):
        """Fmt target should invoke pre-commit via uvx in dry-run output."""
        proc = run_make(logger, ["fmt"])
        out = proc.stdout
        # Check for uvx command with the configured path
        expected_uvx = f"{expected_uv_install_dir}/uvx"
        assert f"{expected_uvx} pre-commit run --all-files" in out

    def test_deptry_target_dry_run(self, logger, expected_uv_install_dir):
        """Deptry target should invoke deptry via uvx in dry-run output."""
        proc = run_make(logger, ["deptry"])
        out = proc.stdout
        # Check for uvx command with the configured path
        expected_uvx = f"{expected_uv_install_dir}/uvx"
        assert f'{expected_uvx} deptry "src"' in out

    def test_test_target_dry_run(self, logger, expected_uv_install_dir):
        """Test target should invoke pytest via uv with coverage and HTML outputs in dry-run output."""
        proc = run_make(logger, ["test"])
        out = proc.stdout
        # Expect key steps
        assert "mkdir -p _tests/html-coverage _tests/html-report" in out
        # Check for uv command with the configured path
        expected_uv = f"{expected_uv_install_dir}/uv"
        assert f"{expected_uv} run pytest" in out

    def test_book_target_dry_run(self, logger, expected_uv_install_dir):
        """Book target should run inline commands to assemble the book without go-task."""
        proc = run_make(logger, ["book"])
        out = proc.stdout
        # Expect marimushka export to install marimo and minibook to be invoked
        # Check for uvx command with the configured path
        expected_uvx = f"{expected_uv_install_dir}/uvx"
        assert f"{expected_uvx} minibook" in out

    def test_all_target_dry_run(self, logger):
        """All target echoes a composite message in dry-run output."""
        proc = run_make(logger, ["all"])
        out = proc.stdout
        # The composite target should echo a message
        assert "Run fmt, deptry, test and book" in out

    def test_uv_no_modify_path_is_exported(self, logger):
        """`UV_NO_MODIFY_PATH` should be set to `1` in the Makefile."""
        proc = run_make(logger, ["print-UV_NO_MODIFY_PATH"], dry_run=False)
        out = strip_ansi(proc.stdout)
        assert "Value of UV_NO_MODIFY_PATH:\n1" in out

    def test_uv_install_dir_is_bin(self, logger, expected_uv_install_dir):
        """`UV_INSTALL_DIR` can be configured via environment variable or defaults to ./bin."""
        proc = run_make(logger, ["print-UV_INSTALL_DIR"], dry_run=False)
        out = strip_ansi(proc.stdout)
        # Check if UV_INSTALL_DIR is set in environment, otherwise expect default ./bin
        assert f"Value of UV_INSTALL_DIR:\n{expected_uv_install_dir}" in out

    def test_uv_bin_is_bin_uv(self, logger, expected_uv_install_dir):
        """`UV_BIN` is derived from UV_INSTALL_DIR environment variable or defaults to ./bin/uv."""
        proc = run_make(logger, ["print-UV_BIN"], dry_run=False)
        out = strip_ansi(proc.stdout)
        # Check if UV_INSTALL_DIR is set in environment, otherwise expect default ./bin
        expected_bin = f"{expected_uv_install_dir}/uv"
        assert f"Value of UV_BIN:\n{expected_bin}" in out

    def test_uvx_bin_is_bin_uvx(self, logger, expected_uv_install_dir):
        """`UVX_BIN` is derived from UV_INSTALL_DIR environment variable or defaults to ./bin/uvx."""
        proc = run_make(logger, ["print-UVX_BIN"], dry_run=False)
        out = strip_ansi(proc.stdout)
        # Check if UV_INSTALL_DIR is set in environment, otherwise expect default ./bin
        expected_bin = f"{expected_uv_install_dir}/uvx"
        assert f"Value of UVX_BIN:\n{expected_bin}" in out

    def test_script_folder_is_github_scripts(self, logger):
        """`SCRIPTS_FOLDER` should point to `.github/scripts`."""
        proc = run_make(logger, ["print-SCRIPTS_FOLDER"], dry_run=False)
        out = strip_ansi(proc.stdout)
        assert "Value of SCRIPTS_FOLDER:\n.github/scripts" in out

    def test_custom_scripts_folder_is_set(self, logger):
        """`CUSTOM_SCRIPTS_FOLDER` should point to `.github/scripts/customisations`."""
        proc = run_make(logger, ["print-CUSTOM_SCRIPTS_FOLDER"], dry_run=False)
        out = strip_ansi(proc.stdout)
        assert "Value of CUSTOM_SCRIPTS_FOLDER:\n.github/scripts/customisations" in out


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
        """Makefile should contain expected targets."""
        makefile = root / "Makefile"
        content = makefile.read_text()

        expected_targets = ["install", "fmt", "test", "deptry", "book", "help"]
        for target in expected_targets:
            assert f"{target}:" in content or f".PHONY: {target}" in content

    def test_makefile_has_uv_variables(self, root):
        """Makefile should define UV-related variables."""
        makefile = root / "Makefile"
        content = makefile.read_text()

        assert "UV_BIN" in content or "uv" in content.lower()
