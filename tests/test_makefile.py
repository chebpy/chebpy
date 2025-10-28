"""Tests for the Makefile targets and help output.

These tests validate that the Makefile exposes expected targets and
emits the correct commands without actually executing them by using
`make -n` (dry-run). This keeps tests fast and avoids network calls.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def setup_tmp_makefile(tmp_path: Path):
    """Copy only the Makefile into a temp directory and chdir there.

    We rely on `make -n` so that no real commands are executed.
    """
    project_root = Path(os.getcwd())

    # Copy the Makefile into the temporary working directory
    shutil.copy(project_root / "Makefile", tmp_path / "Makefile")

    # Move into tmp directory for isolation
    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def run_make(args: list[str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run `make` with optional arguments and return the completed process.

    Args:
        args: Additional arguments for make
        check: If True, raise on non-zero return code
    """
    cmd = ["make"]
    if args:
        cmd.extend(args)
    # Use -s to reduce noise, -n to avoid executing commands
    cmd.insert(1, "-sn")
    result = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    if check and result.returncode != 0:
        msg = f"make failed with code {result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        raise AssertionError(msg)
    return result


class TestMakefile:
    """Smoke tests for Makefile help and common targets using make -n."""

    def test_default_goal_is_help(self):
        """Default goal should render the help index with known targets."""
        proc = run_make()
        out = proc.stdout
        assert "Usage:" in out
        assert "Targets:" in out
        # ensure a few known targets appear in the help index
        for target in ["install", "fmt", "deptry", "test", "book", "help"]:
            assert target in out

    def test_help_target(self):
        """Explicit `make help` prints usage, targets, and section headers."""
        proc = run_make(["help"])
        out = proc.stdout
        assert "Usage:" in out
        assert "Targets:" in out
        assert "Bootstrap" in out or "Meta" in out  # section headers

    def test_fmt_target_dry_run(self):
        """Fmt target should call the quality:lint task in dry-run output."""
        proc = run_make(["fmt"])
        out = proc.stdout
        assert "./bin/task quality:lint" in out

    def test_deptry_target_dry_run(self):
        """Deptry target should call the quality:deptry task in dry-run."""
        proc = run_make(["deptry"])
        out = proc.stdout
        assert "./bin/task quality:deptry" in out

    def test_test_target_dry_run(self):
        """Test target should invoke docs:test in dry-run output."""
        proc = run_make(["test"])
        out = proc.stdout
        assert "./bin/task docs:test" in out

    def test_book_target_dry_run(self):
        """Book target should run all three docs-related commands in order."""
        proc = run_make(["book"])
        out = proc.stdout
        # The composite target should run all dependent commands
        for expected_cmd in [
            "./bin/task docs:docs",
            "./bin/task docs:marimushka",
            "./bin/task docs:book",
        ]:
            assert expected_cmd in out

    def test_all_target_dry_run(self):
        """All target echoes a composite message in dry-run output."""
        proc = run_make(["all"])
        out = proc.stdout
        # The composite target should echo a message
        assert "Run fmt, deptry, test and book" in out

    def test_install_task_dry_run_shows_expected_commands(self):
        """install-task target should show task installer and version check."""
        proc = run_make(["install-task"])
        out = proc.stdout
        # ensure key steps of install are present in the dry run output
        assert "curl --location https://taskfile.dev/install.sh" in out
