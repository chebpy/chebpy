"""Tests for the bump.sh script using a sandboxed git environment.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

Provides test fixtures for testing git-based workflows and version management.
"""

import subprocess

import pytest


@pytest.mark.parametrize(
    "choice, expected_version",
    [
        ("1", "0.1.1"),  # patch
        ("2", "0.2.0"),  # minor
        ("3", "1.0.0"),  # major
    ],
)
def test_bump_updates_version_no_commit(git_repo, choice, expected_version):
    """Running `bump` interactively updates pyproject.toml correctly."""
    script = git_repo / ".rhiza" / "scripts" / "bump.sh"

    # Input: choice -> n (no commit)
    input_str = f"{choice}\nn\n"

    result = subprocess.run([str(script)], cwd=git_repo, input=input_str, capture_output=True, text=True)

    assert result.returncode == 0
    assert f"-> {expected_version} in pyproject.toml" in result.stdout

    # Verify pyproject.toml updated
    with open(git_repo / "pyproject.toml") as f:
        content = f.read()
        assert f'version = "{expected_version}"' in content

    # Verify no tag created yet
    tags = subprocess.check_output(["git", "tag"], cwd=git_repo, text=True)
    assert f"v{expected_version}" not in tags


def test_bump_commit_push(git_repo):
    """Bump with commit and push."""
    script = git_repo / ".rhiza" / "scripts" / "bump.sh"

    # Input: 1 (patch) -> y (commit) -> y (push)
    input_str = "1\ny\ny\n"

    result = subprocess.run([str(script)], cwd=git_repo, input=input_str, capture_output=True, text=True)

    assert result.returncode == 0
    assert "Version committed" in result.stdout
    assert "Pushed to origin/master" in result.stdout

    # Verify commit on remote
    remote_log = subprocess.check_output(["git", "log", "origin/master", "-1", "--pretty=%B"], cwd=git_repo, text=True)
    assert "chore: bump version to 0.1.1" in remote_log


def test_uncommitted_changes_failure(git_repo):
    """Script fails if there are uncommitted changes."""
    script = git_repo / ".rhiza" / "scripts" / "bump.sh"

    # Create a tracked file and commit it
    tracked_file = git_repo / "tracked_file.txt"
    tracked_file.touch()
    subprocess.run(["git", "add", "tracked_file.txt"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-m", "Add tracked file"], cwd=git_repo, check=True)

    # Modify tracked file to create uncommitted change
    with open(tracked_file, "a") as f:
        f.write("\n# change")

    # Input: 1 (patch)
    result = subprocess.run([str(script)], cwd=git_repo, input="1\n", capture_output=True, text=True)

    assert result.returncode == 1
    assert "You have uncommitted changes" in result.stdout


@pytest.mark.parametrize(
    "input_version, expected_version",
    [
        ("1.2.3", "1.2.3"),
        ("v1.2.4", "1.2.4"),
        ("2.0.0rc1", "2.0.0rc1"),
        ("2.0.0a1", "2.0.0a1"),
        ("2.0.0.post1", "2.0.0.post1"),
    ],
)
def test_bump_explicit_version(git_repo, input_version, expected_version):
    """Bump with explicit version."""
    script = git_repo / ".rhiza" / "scripts" / "bump.sh"

    # Input: 4 (explicit) -> input_version -> n (no commit)
    input_str = f"4\n{input_version}\nn\n"

    result = subprocess.run([str(script)], cwd=git_repo, input=input_str, capture_output=True, text=True)

    assert result.returncode == 0
    assert f"-> {expected_version} in pyproject.toml" in result.stdout
    with open(git_repo / "pyproject.toml") as f:
        content = f.read()
        assert f'version = "{expected_version}"' in content


def test_bump_explicit_version_invalid(git_repo):
    """Bump fails with invalid explicit version."""
    script = git_repo / ".rhiza" / "scripts" / "bump.sh"
    version = "not-a-version"

    # Input: 4 (explicit) -> not-a-version
    input_str = f"4\n{version}\n"

    result = subprocess.run([str(script)], cwd=git_repo, input=input_str, capture_output=True, text=True)

    assert result.returncode == 1
    assert f"Invalid version format: {version}" in result.stdout


def test_bump_fails_existing_tag(git_repo):
    """Bump fails if tag already exists."""
    script = git_repo / ".rhiza" / "scripts" / "bump.sh"

    # Create tag v0.1.1
    subprocess.run(["git", "tag", "v0.1.1"], cwd=git_repo, check=True)

    # Try to bump to 0.1.1 (patch bump from 0.1.0)
    # Input: 1 (patch)
    result = subprocess.run([str(script)], cwd=git_repo, input="1\n", capture_output=True, text=True)

    assert result.returncode == 1
    assert "Tag 'v0.1.1' already exists locally" in result.stdout


def test_warn_on_non_default_branch(git_repo):
    """Script warns if not on default branch."""
    script = git_repo / ".rhiza" / "scripts" / "bump.sh"

    # Create and switch to new branch
    subprocess.run(["git", "checkout", "-b", "feature"], cwd=git_repo, check=True)

    # Run bump (input 1 (patch), then 'y' to proceed with non-default branch, then n (no commit))
    input_str = "1\ny\nn\n"

    result = subprocess.run([str(script)], cwd=git_repo, input=input_str, capture_output=True, text=True)
    assert result.returncode == 0
    assert "You are on branch 'feature' but the default branch is 'master'" in result.stdout


def test_bump_fails_if_pyproject_toml_dirty(git_repo):
    """Bump fails if pyproject.toml has uncommitted changes."""
    script = git_repo / ".rhiza" / "scripts" / "bump.sh"

    # Modify pyproject.toml
    with open(git_repo / "pyproject.toml", "a") as f:
        f.write("\n# dirty")

    # Input: 1 (patch)
    result = subprocess.run([str(script)], cwd=git_repo, input="1\n", capture_output=True, text=True)

    assert result.returncode == 1
    assert "You have uncommitted changes" in result.stdout
