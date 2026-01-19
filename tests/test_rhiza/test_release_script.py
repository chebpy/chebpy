"""Tests for the release.sh script using a sandboxed git environment.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

The script exposes the `release` command (creates and pushes tags).
Tests call the script from a temporary clone and use a small mock `uv`
to avoid external dependencies.
"""

import shutil
import subprocess

# Get absolute paths for executables to avoid S607 warnings from CodeFactor/Bandit
SHELL = shutil.which("sh") or "/bin/sh"
GIT = shutil.which("git") or "/usr/bin/git"


def test_release_creates_tag(git_repo):
    """Release creates a tag."""
    script = git_repo / ".rhiza" / "scripts" / "release.sh"

    # Run release
    # 1. Prompts to create tag -> y
    # 2. Prompts to push tag -> y
    result = subprocess.run([SHELL, str(script)], cwd=git_repo, input="y\ny\n", capture_output=True, text=True)
    assert result.returncode == 0
    assert "Tag 'v0.1.0' created locally" in result.stdout

    # Verify the tag exists
    verify_result = subprocess.run(
        [GIT, "tag", "-l", "v0.1.0"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )
    assert "v0.1.0" in verify_result.stdout


def test_release_fails_if_local_tag_exists(git_repo):
    """If the target tag already exists locally, release should warn and abort if user says no."""
    script = git_repo / ".rhiza" / "scripts" / "release.sh"

    # Create a local tag that matches current version
    subprocess.run([GIT, "tag", "v0.1.0"], cwd=git_repo, check=True)

    # Input 'n' to abort
    result = subprocess.run([SHELL, str(script)], cwd=git_repo, input="n\n", capture_output=True, text=True)

    assert result.returncode == 0
    assert "Tag 'v0.1.0' already exists locally" in result.stdout
    assert "Aborted by user" in result.stdout


def test_release_fails_if_remote_tag_exists(git_repo):
    """Release fails if tag exists on remote."""
    script = git_repo / ".rhiza" / "scripts" / "release.sh"

    # Create tag locally and push to remote
    subprocess.run([GIT, "tag", "v0.1.0"], cwd=git_repo, check=True)
    subprocess.run([GIT, "push", "origin", "v0.1.0"], cwd=git_repo, check=True)

    result = subprocess.run([SHELL, str(script)], cwd=git_repo, input="y\n", capture_output=True, text=True)

    assert result.returncode == 1
    assert "already exists on remote" in result.stdout


def test_release_uncommitted_changes_failure(git_repo):
    """Release fails if there are uncommitted changes (even pyproject.toml)."""
    script = git_repo / ".rhiza" / "scripts" / "release.sh"

    # Modify pyproject.toml (which is allowed in bump but NOT in release)
    with open(git_repo / "pyproject.toml", "a") as f:
        f.write("\n# comment")

    result = subprocess.run([SHELL, str(script)], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 1
    assert "You have uncommitted changes" in result.stdout


def test_release_pushes_if_ahead_of_remote(git_repo):
    """Release prompts to push if local branch is ahead of remote."""
    script = git_repo / ".rhiza" / "scripts" / "release.sh"

    # Create a commit locally that isn't on remote
    tracked_file = git_repo / "file.txt"
    tracked_file.touch()
    subprocess.run([GIT, "add", "file.txt"], cwd=git_repo, check=True)
    subprocess.run([GIT, "commit", "-m", "Local commit"], cwd=git_repo, check=True)

    # Run release
    # 1. Prompts to push -> y
    # 2. Prompts to create tag -> y
    # 3. Prompts to push tag -> y
    result = subprocess.run([SHELL, str(script)], cwd=git_repo, input="y\ny\ny\n", capture_output=True, text=True)

    assert result.returncode == 0
    assert "Your branch is ahead" in result.stdout
    assert "Unpushed commits:" in result.stdout
    assert "Local commit" in result.stdout
    assert "Push changes to remote before releasing?" in result.stdout


def test_release_fails_if_behind_remote(git_repo):
    """Release fails if local branch is behind remote."""
    script = git_repo / ".rhiza" / "scripts" / "release.sh"

    # Create a commit on remote that isn't local
    # We need to clone another repo to push to remote
    other_clone = git_repo.parent / "other_clone"
    subprocess.run([GIT, "clone", str(git_repo.parent / "remote.git"), str(other_clone)], check=True)

    # Configure git user for other_clone (needed in CI)
    subprocess.run([GIT, "config", "user.email", "test@example.com"], cwd=other_clone, check=True)
    subprocess.run([GIT, "config", "user.name", "Test User"], cwd=other_clone, check=True)

    # Commit and push from other clone
    with open(other_clone / "other.txt", "w") as f:
        f.write("content")
    subprocess.run([GIT, "add", "other.txt"], cwd=other_clone, check=True)
    subprocess.run([GIT, "commit", "-m", "Remote commit"], cwd=other_clone, check=True)
    subprocess.run([GIT, "push"], cwd=other_clone, check=True)

    # Run release (it will fetch and see it's behind)
    result = subprocess.run([SHELL, str(script)], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 1
    assert "Your branch is behind" in result.stdout


def test_dry_run_flag_recognized(git_repo):
    """Test that --dry-run flag is recognized and script executes."""
    script = git_repo / ".rhiza" / "scripts" / "release.sh"

    # Run with --dry-run flag
    result = subprocess.run([SHELL, str(script), "--dry-run"], cwd=git_repo, capture_output=True, text=True)

    # Should exit successfully
    assert result.returncode == 0
    # Should show dry-run messages
    assert "[DRY-RUN]" in result.stdout


def test_dry_run_no_git_operations(git_repo):
    """Test that no actual git operations are performed in dry-run mode."""
    script = git_repo / ".rhiza" / "scripts" / "release.sh"

    # Get initial git state
    tags_before = subprocess.run(
        [GIT, "tag", "-l"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    ).stdout

    # Run with --dry-run
    result = subprocess.run([SHELL, str(script), "--dry-run"], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0

    # Verify no tags were created
    tags_after = subprocess.run(
        [GIT, "tag", "-l"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    ).stdout
    assert tags_before == tags_after

    # Verify tag doesn't exist using consistent pattern with other tests
    tag_check = subprocess.run(
        [GIT, "tag", "-l", "v0.1.0"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )
    assert "v0.1.0" not in tag_check.stdout


def test_dry_run_shows_appropriate_messages(git_repo):
    """Test that appropriate DRY-RUN messages are displayed."""
    script = git_repo / ".rhiza" / "scripts" / "release.sh"

    result = subprocess.run([SHELL, str(script), "--dry-run"], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0

    # Check for key dry-run messages indicating simulation mode
    assert "[DRY-RUN]" in result.stdout
    assert "Would prompt" in result.stdout
    assert "Would run: git tag" in result.stdout
    assert "would be created locally" in result.stdout
    assert "Would run: git push origin refs/tags/v0.1.0" in result.stdout
    assert "would be pushed to remote" in result.stdout
    assert "would trigger the release workflow" in result.stdout


def test_dry_run_exits_successfully_without_creating_tags(git_repo):
    """Test that script exits successfully without creating or pushing tags in dry-run mode."""
    script = git_repo / ".rhiza" / "scripts" / "release.sh"

    # Run with --dry-run
    result = subprocess.run([SHELL, str(script), "--dry-run"], cwd=git_repo, capture_output=True, text=True)

    # Should exit successfully
    assert result.returncode == 0

    # Verify no local tag was created
    local_tag_check = subprocess.run(
        [GIT, "tag", "-l", "v0.1.0"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )
    assert "v0.1.0" not in local_tag_check.stdout

    # Verify no remote tag was pushed
    remote_tag_check = subprocess.run(
        [GIT, "ls-remote", "--tags", "origin", "v0.1.0"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )
    assert "v0.1.0" not in remote_tag_check.stdout

    # Verify output indicates dry-run mode with specific indicators
    assert "[DRY-RUN]" in result.stdout
    assert "Would run:" in result.stdout
