"""Tests for the git_repo pytest fixture that creates a mock Git repository.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

This module validates the temporary repository structure, git initialization,
mocked tool executables, environment variables, and basic git configuration the
fixture is expected to provide for integration-style tests.
"""

import os
import subprocess
from pathlib import Path


class TestGitRepoFixture:
    """Tests for the git_repo fixture that sets up a mock git repository."""

    def test_git_repo_creates_temporary_directory(self, git_repo):
        """Git repo fixture should create a temporary directory."""
        assert git_repo.exists()
        assert git_repo.is_dir()

    def test_git_repo_contains_pyproject_toml(self, git_repo):
        """Git repo should contain a pyproject.toml file."""
        pyproject = git_repo / "pyproject.toml"
        assert pyproject.exists()
        content = pyproject.read_text()
        assert 'name = "test-project"' in content
        assert 'version = "0.1.0"' in content

    def test_git_repo_contains_uv_lock(self, git_repo):
        """Git repo should contain a uv.lock file."""
        assert (git_repo / "uv.lock").exists()

    def test_git_repo_has_bin_directory_with_mocks(self, git_repo):
        """Git repo should have bin directory with mock tools."""
        bin_dir = git_repo / "bin"
        assert bin_dir.exists()
        assert (bin_dir / "uv").exists()

    def test_git_repo_mock_tools_are_executable(self, git_repo):
        """Mock tools should be executable."""
        for tool in ["uv"]:
            tool_path = git_repo / "bin" / tool
            assert os.access(tool_path, os.X_OK), f"{tool} is not executable"

    def test_git_repo_has_github_scripts_directory(self, git_repo):
        """Git repo should have .github/rhiza/scripts directory."""
        scripts_dir = git_repo / ".rhiza" / "scripts"
        assert scripts_dir.exists()
        assert (scripts_dir / "release.sh").exists()
        assert (scripts_dir / "bump.sh").exists()

    def test_git_repo_scripts_are_executable(self, git_repo):
        """GitHub scripts should be executable."""
        for script in ["release.sh", "bump.sh"]:
            script_path = git_repo / ".rhiza" / "scripts" / script
            assert os.access(script_path, os.X_OK), f"{script} is not executable"

    def test_git_repo_is_initialized(self, git_repo):
        """Git repo should be properly initialized."""
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert ".git" in result.stdout

    def test_git_repo_has_master_branch(self, git_repo):
        """Git repo should be on master branch."""
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == "master"

    def test_git_repo_has_initial_commit(self, git_repo):
        """Git repo should have an initial commit."""
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Initial commit" in result.stdout

    def test_git_repo_has_remote_configured(self, git_repo):
        """Git repo should have origin remote configured."""
        result = subprocess.run(
            ["git", "remote", "-v"],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "origin" in result.stdout

    def test_git_repo_user_config_is_set(self, git_repo):
        """Git repo should have user.email and user.name configured."""
        email = subprocess.check_output(
            ["git", "config", "user.email"],
            cwd=git_repo,
            text=True,
        ).strip()
        name = subprocess.check_output(
            ["git", "config", "user.name"],
            cwd=git_repo,
            text=True,
        ).strip()
        assert email == "test@example.com"
        assert name == "Test User"

    def test_git_repo_working_tree_is_clean(self, git_repo):
        """Git repo should start with a clean working tree."""
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=git_repo,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_git_repo_changes_current_directory(self, git_repo):
        """Git repo fixture should change to the temporary directory."""
        current_dir = Path.cwd()
        assert current_dir == git_repo

    def test_git_repo_modifies_path_environment(self, git_repo):
        """Git repo fixture should prepend bin directory to PATH."""
        path_env = os.environ.get("PATH", "")
        bin_dir = str(git_repo / "bin")
        assert bin_dir in path_env
        assert path_env.startswith(bin_dir)
