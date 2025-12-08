"""Pytest configuration and fixtures for setting up a mock git repository with versioning and GPG signing."""

import os
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


@pytest.fixture
def git_repo(root, tmp_path, monkeypatch):
    """Sets up a remote bare repo and a local clone with necessary files."""
    remote_dir = tmp_path / "remote.git"
    local_dir = tmp_path / "local"
    gnupg_home = tmp_path / "gnupg"

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

    # Create bin/gpg mock and add bin to PATH
    gpg_path = bin_dir / "gpg"
    with open(gpg_path, "w") as f:
        f.write("""#!/bin/sh
# Mock gpg that produces a dummy signature for git tag -s
echo "ARGS: $@" >> /tmp/gpg_args.log

# Check if we are signing (look for -s or -b or -bsau)
SIGN=0
for arg in "$@"; do
    case "$arg" in
        *-bsau*) SIGN=1 ;;
        *-s*)    SIGN=1 ;;
        *-b*)    SIGN=1 ;;
    esac
done

if [ "$SIGN" = "1" ]; then
    # Output status to stderr (fd 2) as requested by --status-fd=2
    echo "[GNUPG:] SIG_CREATED D 1 2 00 1234567890 1" >&2

    # Output signature to stdout
    echo "-----BEGIN PGP SIGNATURE-----"
    echo ""
    echo "Dummy Signature"
    echo "-----END PGP SIGNATURE-----"
fi
exit 0
""")
    gpg_path.chmod(0o755)

    # Also provide a gpgv shim that delegates to our gpg mock (git may use gpgv for verification)
    gpgv_path = bin_dir / "gpgv"
    with open(gpgv_path, "w") as f:
        f.write("#!/bin/sh\nexit 0\n")
    gpgv_path.chmod(0o755)

    # Ensure our bin comes first on PATH so 'gpg' and 'uv' resolve to mocks
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")

    # Provide a git shim to make `git tag -v` succeed in this mocked environment
    git_wrapper = bin_dir / "git"
    with open(git_wrapper, "w") as f:
        f.write('#!/bin/sh\nif [ "$1" = "tag" ] && [ "$2" = "-v" ]; then exit 0; fi\nexec /usr/bin/git "$@"\n')
    git_wrapper.chmod(0o755)

    # Copy scripts
    script_dir = local_dir / ".github" / "scripts"
    script_dir.mkdir(parents=True)

    shutil.copy(root / ".github" / "scripts" / "release.sh", script_dir / "release.sh")
    shutil.copy(root / ".github" / "scripts" / "bump.sh", script_dir / "bump.sh")

    (script_dir / "release.sh").chmod(0o755)
    (script_dir / "bump.sh").chmod(0o755)

    # Set up a test GPG key for tag signing
    gnupg_home.mkdir(mode=0o700)
    monkeypatch.setenv("GNUPGHOME", str(gnupg_home))

    # Commit and push initial state
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)
    subprocess.run(["git", "push", "origin", "master"], check=True)

    yield local_dir
