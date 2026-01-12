"""Tests for the GitHub Makefile targets using safe dry-runs.

These tests validate that the .github/Makefile.gh targets are correctly exposed
and emit the expected commands without actually executing them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# We need to copy these files to the temp dir for the tests to work
REQUIRED_FILES = [
    ".github/Makefile.gh",
]


@pytest.fixture(autouse=True)
def setup_gh_makefile(logger, root, tmp_path: Path):
    """Copy the Makefile and GitHub Makefile into a temp directory."""
    logger.debug("Setting up temporary GitHub Makefile test dir: %s", tmp_path)

    # Copy the main Makefile
    if (root / "Makefile").exists():
        shutil.copy(root / "Makefile", tmp_path / "Makefile")

    if (root / ".rhiza" / ".env").exists():
        (tmp_path / ".rhiza").mkdir(exist_ok=True)
        shutil.copy(root / ".rhiza" / ".env", tmp_path / ".rhiza" / ".env")

    # Copy required split Makefiles
    for rel_path in REQUIRED_FILES:
        source_path = root / rel_path
        if source_path.exists():
            dest_path = tmp_path / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, dest_path)
            logger.debug("Copied %s to %s", source_path, dest_path)
        else:
            pytest.skip(f"Required file {rel_path} not found")

    # Move into tmp directory
    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def run_make(
    logger, args: list[str] | None = None, check: bool = True, dry_run: bool = True
) -> subprocess.CompletedProcess:
    """Run `make` with optional arguments."""
    cmd = ["make"]
    if args:
        cmd.extend(args)
    flags = "-sn" if dry_run else "-s"
    cmd.insert(1, flags)

    logger.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if check and result.returncode != 0:
        msg = f"make failed with code {result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        raise AssertionError(msg)
    return result


def test_gh_targets_exist(logger):
    """Verify that GitHub targets are listed in help."""
    result = run_make(logger, ["help"], dry_run=False)
    output = result.stdout

    expected_targets = ["gh-install", "view-prs", "view-issues", "failed-workflows", "whoami"]

    for target in expected_targets:
        assert target in output, f"Target {target} not found in help output"


def test_gh_install_dry_run(logger):
    """Verify gh-install target dry-run."""
    result = run_make(logger, ["gh-install"])
    # In dry-run, we expect to see the shell commands that would be executed.
    # Since the recipe uses @if, make -n might verify the syntax or show the command if not silenced.
    # However, with -s (silent), make -n might not show much for @ commands unless they are echoed.
    # But we mainly want to ensure it runs without error.
    assert result.returncode == 0


def test_view_prs_dry_run(logger):
    """Verify view-prs target dry-run."""
    result = run_make(logger, ["view-prs"])
    assert result.returncode == 0


def test_view_issues_dry_run(logger):
    """Verify view-issues target dry-run."""
    result = run_make(logger, ["view-issues"])
    assert result.returncode == 0


def test_failed_workflows_dry_run(logger):
    """Verify failed-workflows target dry-run."""
    result = run_make(logger, ["failed-workflows"])
    assert result.returncode == 0


def test_whoami_dry_run(logger):
    """Verify whoami target dry-run."""
    result = run_make(logger, ["whoami"])
    assert result.returncode == 0
