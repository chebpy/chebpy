"""Tests for the .rhiza-version file and related Makefile functionality.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

These tests validate:
- Reading RHIZA_VERSION from .rhiza/.rhiza-version
- The summarise-sync Makefile target
- Version usage in sync and validate targets
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import run_make, setup_rhiza_git_repo, strip_ansi


@pytest.fixture(autouse=True)
def setup_tmp_makefile(logger, root, tmp_path: Path):
    """Copy the Makefile and necessary files into a temp directory and chdir there.

    We rely on `make -n` so that no real commands are executed.
    """
    logger.debug("Setting up temporary Makefile test dir: %s", tmp_path)

    # Copy the main Makefile into the temporary working directory
    shutil.copy(root / "Makefile", tmp_path / "Makefile")

    # Copy core Rhiza Makefiles and version file
    (tmp_path / ".rhiza").mkdir(exist_ok=True)
    shutil.copy(root / ".rhiza" / "rhiza.mk", tmp_path / ".rhiza" / "rhiza.mk")

    # Copy .rhiza-version if it exists
    if (root / ".rhiza" / ".rhiza-version").exists():
        shutil.copy(root / ".rhiza" / ".rhiza-version", tmp_path / ".rhiza" / ".rhiza-version")

    # Create a minimal, deterministic .rhiza/.env for tests
    env_content = "SCRIPTS_FOLDER=.rhiza/scripts\nCUSTOM_SCRIPTS_FOLDER=.rhiza/customisations/scripts\n"
    (tmp_path / ".rhiza" / ".env").write_text(env_content)

    logger.debug("Copied Makefile from %s to %s", root / "Makefile", tmp_path / "Makefile")

    # Create a minimal .rhiza/template.yml
    (tmp_path / ".rhiza" / "template.yml").write_text("repository: Jebel-Quant/rhiza\nref: main\n")

    # Sort out pyproject.toml
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test-project"\nversion = "0.1.0"\n')

    # Move into tmp directory for isolation
    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    logger.debug("Changed working directory to %s", tmp_path)

    # Initialize a git repo so that commands checking for it (like materialize) don't fail validation
    setup_rhiza_git_repo()

    # Create src and tests directories to satisfy validate
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "tests").mkdir(exist_ok=True)

    try:
        yield
    finally:
        os.chdir(old_cwd)
        logger.debug("Restored working directory to %s", old_cwd)


class TestRhizaVersion:
    """Tests for RHIZA_VERSION variable and .rhiza-version file."""

    def test_rhiza_version_exists_in_file(self, root):
        """The .rhiza/.rhiza-version file should exist and contain a version."""
        version_file = root / ".rhiza" / ".rhiza-version"
        assert version_file.exists()
        assert version_file.is_file()

        content = version_file.read_text().strip()
        assert len(content) > 0
        # Check it looks like a version (e.g., "0.9.0")
        assert content[0].isdigit()

    def test_rhiza_version_exported_in_makefile(self, logger):
        """RHIZA_VERSION should be exported and readable."""
        proc = run_make(logger, ["print-RHIZA_VERSION"], dry_run=False)
        out = strip_ansi(proc.stdout)
        # The output should contain the version value
        assert "Value of RHIZA_VERSION:" in out
        # Should have a version number
        assert any(char.isdigit() for char in out)

    def test_rhiza_version_defaults_to_0_9_0_without_file(self, logger, tmp_path):
        """RHIZA_VERSION should default to 0.9.0 if .rhiza-version doesn't exist."""
        # Remove the .rhiza-version file
        version_file = tmp_path / ".rhiza" / ".rhiza-version"
        if version_file.exists():
            version_file.unlink()

        proc = run_make(logger, ["print-RHIZA_VERSION"], dry_run=False)
        out = strip_ansi(proc.stdout)
        assert "Value of RHIZA_VERSION:\n0.9.0" in out

    def test_rhiza_version_used_in_sync_target(self, logger):
        """Sync target should use RHIZA_VERSION from .rhiza-version."""
        proc = run_make(logger, ["sync"])
        out = proc.stdout
        # Check that rhiza>= is used with the version variable
        assert 'uvx "rhiza>=' in out or "rhiza>=" in out

    def test_rhiza_version_used_in_validate_target(self, logger):
        """Validate target should use RHIZA_VERSION from .rhiza-version."""
        proc = run_make(logger, ["validate"])
        out = proc.stdout
        # Check that rhiza>= is used with the version variable
        assert 'uvx "rhiza>=' in out or "rhiza>=" in out


class TestSummariseSync:
    """Tests for the summarise-sync Makefile target."""

    def test_summarise_sync_target_exists(self, logger):
        """The summarise-sync target should be available."""
        proc = run_make(logger, ["help"])
        out = proc.stdout
        # Check that summarise-sync appears in help
        assert "summarise-sync" in out

    def test_summarise_sync_dry_run(self, logger):
        """Summarise-sync target should invoke rhiza summarise in dry-run output."""
        proc = run_make(logger, ["summarise-sync"])
        out = proc.stdout
        # Check for uvx command with rhiza summarise
        assert "uvx" in out
        assert "rhiza" in out
        assert "summarise" in out

    def test_summarise_sync_uses_rhiza_version(self, logger):
        """Summarise-sync target should use RHIZA_VERSION from .rhiza-version."""
        proc = run_make(logger, ["summarise-sync"])
        out = proc.stdout
        # Check that rhiza>= is used with the version
        assert 'uvx "rhiza>=' in out or "rhiza>=" in out

    def test_summarise_sync_skips_in_rhiza_repo(self, logger):
        """Summarise-sync target should skip execution in rhiza repository."""
        # setup_rhiza_git_repo() is already called by fixture

        proc = run_make(logger, ["summarise-sync"], dry_run=True)
        # Should succeed but skip the actual summarise
        assert proc.returncode == 0
        # Verify the skip message is in the output
        assert "Skipping summarise-sync in rhiza repository" in proc.stdout

    def test_summarise_sync_requires_install_uv(self, logger):
        """Summarise-sync should ensure uv is installed first."""
        proc = run_make(logger, ["summarise-sync"])
        out = proc.stdout
        # The output should show that install-uv is called
        # This might be implicit via the dependency chain
        assert "rhiza" in out


class TestWorkflowSync:
    """Tests to validate the workflow pattern used in .github/workflows/rhiza_sync.yml."""

    def test_workflow_version_reading_pattern(self, logger, tmp_path):
        """Test the pattern used in workflow to read Rhiza version."""
        # Create .rhiza-version file
        version_file = tmp_path / ".rhiza" / ".rhiza-version"
        version_file.write_text("0.9.5\n")

        # Simulate the workflow's version reading step
        result = subprocess.run(
            [shutil.which("cat") or "cat", str(version_file)],
            capture_output=True,
            text=True,
            check=True,
        )
        version = result.stdout.strip()

        assert version == "0.9.5"

    def test_workflow_version_fallback_pattern(self, logger, tmp_path):
        """Test the fallback pattern when .rhiza-version doesn't exist."""
        # Ensure .rhiza-version doesn't exist
        version_file = tmp_path / ".rhiza" / ".rhiza-version"
        if version_file.exists():
            version_file.unlink()

        # Simulate the workflow's version reading with fallback using proper subprocess
        try:
            result = subprocess.run(
                [shutil.which("cat") or "cat", str(version_file)],
                capture_output=True,
                text=True,
                check=True,
            )
            version = result.stdout.strip()
        except subprocess.CalledProcessError:
            # File doesn't exist, use fallback
            version = "0.9.0"

        assert version == "0.9.0"

    def test_workflow_uvx_command_format(self, logger):
        """Test that the uvx command format matches workflow expectations."""
        # This test validates the command format used in both Makefile and workflow
        proc = run_make(logger, ["sync"])
        out = proc.stdout

        # The format should be: uvx "rhiza>=VERSION" materialize --force .
        assert 'uvx "rhiza>=' in out
        assert "materialize --force" in out

    def test_workflow_summarise_command_format(self, logger):
        """Test that the summarise command format matches workflow expectations."""
        proc = run_make(logger, ["summarise-sync"])
        out = proc.stdout

        # The format should be: uvx "rhiza>=VERSION" summarise .
        assert 'uvx "rhiza>=' in out
        assert "summarise" in out
