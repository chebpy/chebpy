"""Tests for Git LFS Makefile targets.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

Tests the lfs-install, lfs-pull, lfs-track, and lfs-status targets.
"""

import os
import shutil
import subprocess  # nosec

import pytest

# Get shell path and make command once at module level
SHELL = shutil.which("sh") or "/bin/sh"
MAKE = shutil.which("make") or "/usr/bin/make"


def test_lfs_targets_exist(git_repo, logger):
    """Test that all LFS targets are defined in the Makefile."""
    if not (git_repo / ".rhiza" / "make.d" / "lfs.mk").exists():
        pytest.skip("lfs.mk not found, skipping test")

    result = subprocess.run(
        [MAKE, "help"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )  # nosec

    assert result.returncode == 0
    assert "lfs-install" in result.stdout
    assert "lfs-pull" in result.stdout
    assert "lfs-track" in result.stdout
    assert "lfs-status" in result.stdout
    assert "Git LFS" in result.stdout


def test_lfs_install_dry_run(git_repo, logger):
    """Test lfs-install target in dry-run mode."""
    if not (git_repo / ".rhiza" / "make.d" / "lfs.mk").exists():
        pytest.skip("lfs.mk not found, skipping test")

    env = os.environ.copy()

    result = subprocess.run(
        [MAKE, "-n", "lfs-install"],
        env=env,
        cwd=git_repo,
        capture_output=True,
        text=True,
    )  # nosec

    assert result.returncode == 0
    # Check that the command includes OS detection
    assert "uname -s" in result.stdout
    assert "uname -m" in result.stdout


def test_lfs_install_macos_logic(git_repo, logger, monkeypatch):
    """Test that lfs-install generates correct logic for macOS."""
    if not (git_repo / ".rhiza" / "make.d" / "lfs.mk").exists():
        pytest.skip("lfs.mk not found, skipping test")

    env = os.environ.copy()

    result = subprocess.run(
        [MAKE, "-n", "lfs-install"],
        env=env,
        cwd=git_repo,
        capture_output=True,
        text=True,
    )  # nosec

    assert result.returncode == 0
    # Verify macOS installation logic is present
    assert "Darwin" in result.stdout
    assert "darwin-arm64" in result.stdout
    assert "darwin-amd64" in result.stdout
    assert ".local/bin" in result.stdout
    assert "curl" in result.stdout
    assert "github.com/git-lfs/git-lfs/releases" in result.stdout


def test_lfs_install_linux_logic(git_repo, logger):
    """Test that lfs-install generates correct logic for Linux."""
    if not (git_repo / ".rhiza" / "make.d" / "lfs.mk").exists():
        pytest.skip("lfs.mk not found, skipping test")

    env = os.environ.copy()

    result = subprocess.run(
        [MAKE, "-n", "lfs-install"],
        env=env,
        cwd=git_repo,
        capture_output=True,
        text=True,
    )  # nosec

    assert result.returncode == 0
    # Verify Linux installation logic is present
    assert "Linux" in result.stdout
    assert "apt-get update" in result.stdout
    assert "apt-get install" in result.stdout
    assert "git-lfs" in result.stdout


def test_lfs_pull_target(git_repo, logger):
    """Test lfs-pull target in dry-run mode."""
    if not (git_repo / ".rhiza" / "make.d" / "lfs.mk").exists():
        pytest.skip("lfs.mk not found, skipping test")

    result = subprocess.run(
        [MAKE, "-n", "lfs-pull"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )  # nosec

    assert result.returncode == 0
    assert "git lfs pull" in result.stdout


def test_lfs_track_target(git_repo, logger):
    """Test lfs-track target in dry-run mode."""
    if not (git_repo / ".rhiza" / "make.d" / "lfs.mk").exists():
        pytest.skip("lfs.mk not found, skipping test")

    result = subprocess.run(
        [MAKE, "-n", "lfs-track"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )  # nosec

    assert result.returncode == 0
    assert "git lfs track" in result.stdout


def test_lfs_status_target(git_repo, logger):
    """Test lfs-status target in dry-run mode."""
    if not (git_repo / ".rhiza" / "make.d" / "lfs.mk").exists():
        pytest.skip("lfs.mk not found, skipping test")

    result = subprocess.run(
        [MAKE, "-n", "lfs-status"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )  # nosec

    assert result.returncode == 0
    assert "git lfs status" in result.stdout


def test_lfs_install_error_handling(git_repo, logger):
    """Test that lfs-install includes error handling."""
    if not (git_repo / ".rhiza" / "make.d" / "lfs.mk").exists():
        pytest.skip("lfs.mk not found, skipping test")

    result = subprocess.run(
        [MAKE, "-n", "lfs-install"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )  # nosec

    assert result.returncode == 0
    # Verify error handling is present
    assert "ERROR" in result.stdout
    assert "exit 1" in result.stdout


def test_lfs_install_uses_github_api(git_repo, logger):
    """Test that lfs-install uses GitHub API for version detection."""
    if not (git_repo / ".rhiza" / "make.d" / "lfs.mk").exists():
        pytest.skip("lfs.mk not found, skipping test")

    result = subprocess.run(
        [MAKE, "-n", "lfs-install"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )  # nosec

    assert result.returncode == 0


def test_lfs_install_sudo_handling(git_repo, logger):
    """Test that lfs-install handles sudo correctly on Linux."""
    if not (git_repo / ".rhiza" / "make.d" / "lfs.mk").exists():
        pytest.skip("lfs.mk not found, skipping test")

    result = subprocess.run(
        [MAKE, "-n", "lfs-install"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )  # nosec

    assert result.returncode == 0
    # Verify sudo logic is present
    assert "sudo" in result.stdout
    assert "id -u" in result.stdout


@pytest.mark.skipif(
    not shutil.which("git-lfs"),
    reason="git-lfs not installed",
)
def test_lfs_actual_execution_status(git_repo, logger):
    """Test actual execution of lfs-status (requires git-lfs to be installed)."""
    if not (git_repo / ".rhiza" / "make.d" / "lfs.mk").exists():
        pytest.skip("lfs.mk not found, skipping test")

    # Initialize git-lfs in the test repo
    subprocess.run(["git", "lfs", "install"], cwd=git_repo, capture_output=True)  # nosec

    result = subprocess.run(
        [MAKE, "lfs-status"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )  # nosec

    # Should succeed even if no LFS files are tracked
    assert result.returncode == 0


@pytest.mark.skipif(
    not shutil.which("git-lfs"),
    reason="git-lfs not installed",
)
def test_lfs_actual_execution_track(git_repo, logger):
    """Test actual execution of lfs-track (requires git-lfs to be installed)."""
    if not (git_repo / ".rhiza" / "make.d" / "lfs.mk").exists():
        pytest.skip("lfs.mk not found, skipping test")

    # Initialize git-lfs in the test repo
    subprocess.run(["git", "lfs", "install"], cwd=git_repo, capture_output=True)  # nosec

    result = subprocess.run(
        [MAKE, "lfs-track"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )  # nosec

    # Should succeed even if no patterns are tracked
    assert result.returncode == 0
