"""Tests for the Makefile targets and help output using safe dry‑runs.

These tests validate that the Makefile exposes expected targets and emits
the correct commands without actually executing them, by invoking `make -n`
(dry-run). We also pass `-s` to reduce noise in CI logs. This approach keeps
tests fast, portable, and free of side effects like network or environment
changes.
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
    project_root = Path(__file__).parent.parent

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
        """Fmt target should invoke pre-commit via uvx in dry-run output."""
        proc = run_make(["fmt"])
        out = proc.stdout
        assert "./bin/uvx pre-commit run --all-files" in out

    def test_deptry_target_dry_run(self):
        """Deptry target should invoke deptry via uvx in dry-run output."""
        proc = run_make(["deptry"])
        out = proc.stdout
        assert './bin/uvx deptry "src"' in out

    def test_test_target_dry_run(self):
        """Test target should invoke pytest via uv with coverage and HTML outputs in dry-run output."""
        proc = run_make(["test"])
        out = proc.stdout
        # Expect key steps
        assert "mkdir -p _tests/html-coverage _tests/html-report" in out
        assert "./bin/uv run pytest" in out

    def test_book_target_dry_run(self):
        """Book target should run inline commands to assemble the book without go-task."""
        proc = run_make(["book"])
        out = proc.stdout
        # Expect marimushka export to install marimo and minibook to be invoked
        assert "./bin/uvx minibook" in out

    def test_all_target_dry_run(self):
        """All target echoes a composite message in dry-run output."""
        proc = run_make(["all"])
        out = proc.stdout
        # The composite target should echo a message
        assert "Run fmt, deptry, test and book" in out

    def test_uv_no_modify_path_is_exported(self):
        """`UV_NO_MODIFY_PATH` should be set to `1` in the Makefile."""
        proc = run_make(["print-UV_NO_MODIFY_PATH"])
        out = proc.stdout
        assert "UV_NO_MODIFY_PATH = 1" in out

    def test_uv_install_dir_is_bin(self):
        """`UV_INSTALL_DIR` should point to `./bin`."""
        proc = run_make(["print-UV_INSTALL_DIR"])
        out = proc.stdout
        assert "UV_INSTALL_DIR = ./bin" in out

    def test_uv_bin_is_bin_uv(self):
        """`UV_BIN` should point to `./bin/uv`."""
        proc = run_make(["print-UV_BIN"])
        out = proc.stdout
        assert "UV_BIN = ./bin/uv" in out

    def test_uvx_bin_is_bin_uvx(self):
        """`UVX_BIN` should point to `./bin/uvx`."""
        proc = run_make(["print-UVX_BIN"])
        out = proc.stdout
        assert "UVX_BIN = ./bin/uvx" in out
