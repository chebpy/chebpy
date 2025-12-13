"""Pytest configuration and fixtures for setting up a mock git repository with versioning."""

import logging
import os
import pathlib
import shutil
import subprocess

import pytest

MOCK_UV_SCRIPT = """#!/usr/bin/env python3
import sys
import re

def get_version():
    with open("pyproject.toml", "r") as f:
        content = f.read()
    match = re.search(r'version = "(.*?)"', content)
    return match.group(1) if match else "0.0.0"

def set_version(new_version):
    with open("pyproject.toml", "r") as f:
        content = f.read()
    new_content = re.sub(r'version = ".*?"', f'version = "{new_version}"', content)
    with open("pyproject.toml", "w") as f:
        f.write(new_content)

def bump_version(current, bump_type):
    major, minor, patch = map(int, current.split('.'))
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    return current

def main():
    args = sys.argv[1:]
    # Expected invocations from release.sh start with 'version'
    if not args or args[0] != "version":
        sys.exit(1)

    # uv version --short
    if "--short" in args and "--bump" not in args:
        print(get_version())
        return

    # uv version --bump <type> --dry-run --short
    if "--bump" in args and "--dry-run" in args and "--short" in args:
        bump_idx = args.index("--bump") + 1
        bump_type = args[bump_idx]
        current = get_version()
        print(bump_version(current, bump_type))
        return

    # uv version --bump <type> (actual update)
    if "--bump" in args and "--dry-run" not in args:
        bump_idx = args.index("--bump") + 1
        bump_type = args[bump_idx]
        current = get_version()
        new_ver = bump_version(current, bump_type)
        set_version(new_ver)
        return

    # uv version <version> --dry-run
    if len(args) >= 2 and not args[1].startswith("-") and "--dry-run" in args:
        # Just exit 0 if valid
        return

    # uv version <version> (actual update)
    if len(args) == 2 and not args[1].startswith("-"):
        set_version(args[1])
        return

if __name__ == "__main__":
    main()
"""


@pytest.fixture(scope="session")
def root():
    """Return the repository root directory as a pathlib.Path.

    Used by tests to locate files and scripts relative to the project root.
    """
    return pathlib.Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def logger():
    """Provide a session-scoped logger for tests.

    Returns:
        logging.Logger: Logger configured for the test session.
    """
    return logging.getLogger(__name__)


@pytest.fixture
def git_repo(root, tmp_path, monkeypatch):
    """Sets up a remote bare repo and a local clone with necessary files."""
    remote_dir = tmp_path / "remote.git"
    local_dir = tmp_path / "local"

    # 1. Create bare remote
    remote_dir.mkdir()
    subprocess.run(["git", "init", "--bare", str(remote_dir)], check=True)
    # Ensure the remote's default HEAD points to master for predictable behavior
    subprocess.run(["git", "symbolic-ref", "HEAD", "refs/heads/master"], cwd=remote_dir, check=True)

    # 2. Clone to local
    subprocess.run(["git", "clone", str(remote_dir), str(local_dir)], check=True)

    # Use monkeypatch to safely change cwd for the duration of the test
    monkeypatch.chdir(local_dir)

    # Ensure local default branch is 'master' to match test expectations
    subprocess.run(["git", "checkout", "-b", "master"], check=True)

    # Create pyproject.toml
    with open("pyproject.toml", "w") as f:
        f.write('[project]\nname = "test-project"\nversion = "0.1.0"\n')

    # Create dummy uv.lock
    with open("uv.lock", "w") as f:
        f.write("")

    # Create bin/uv mock
    bin_dir = local_dir / "bin"
    bin_dir.mkdir()

    uv_path = bin_dir / "uv"
    with open(uv_path, "w") as f:
        f.write(MOCK_UV_SCRIPT)
    uv_path.chmod(0o755)

    # Ensure our bin comes first on PATH so 'uv' resolves to mock
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")

    # Copy scripts
    script_dir = local_dir / ".github" / "scripts"
    script_dir.mkdir(parents=True)

    shutil.copy(root / ".github" / "scripts" / "release.sh", script_dir / "release.sh")
    shutil.copy(root / ".github" / "scripts" / "bump.sh", script_dir / "bump.sh")

    (script_dir / "release.sh").chmod(0o755)
    (script_dir / "bump.sh").chmod(0o755)

    # Commit and push initial state
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)
    subprocess.run(["git", "push", "origin", "master"], check=True)

    yield local_dir
