"""Tests for the release.sh script using a sandboxed git environment.

The script exposes two commands: `bump` (updates pyproject.toml via `uv`) and
`release` (creates and pushes tags). Tests call the script from a temporary
clone and use a small mock `uv` to avoid external dependencies.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Path to the release script in the actual workspace
WORKSPACE_ROOT = Path(__file__).parent.parent
RELEASE_SCRIPT_PATH = WORKSPACE_ROOT / ".github" / "scripts" / "release.sh"

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
def git_repo(tmp_path, monkeypatch):
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

    # Capture path to the real git before we modify PATH
    real_git = shutil.which("git")

    uv_path = bin_dir / "uv"
    with open(uv_path, "w") as f:
        f.write(MOCK_UV_SCRIPT)
    uv_path.chmod(0o755)

    # Create bin/gpg mock and add bin to PATH
    gpg_path = bin_dir / "gpg"
    with open(gpg_path, "w") as f:
        f.write(
            """#!/bin/sh
# Minimal mock of gpg for tests
# Supports: --batch --gen-key, --list-keys --keyid-format long <email>
# For other invocations (e.g., signing/verification via git), emit success status lines.

# Detect output file target if provided via -o<file>, -o <file>, --output <file>, or --output=<file>
OUTPUT_FILE=""
STATUS_FD=""
prev=""
for arg in "$@"; do
  case "$arg" in
    -o)
      prev="-o"
      ;;
    --output)
      prev="--output"
      ;;
    -o*)
      OUTPUT_FILE="${arg#-o}"
      prev=""
      ;;
    --output=*)
      OUTPUT_FILE="${arg#--output=}"
      prev=""
      ;;
    --status-fd=*)
      STATUS_FD="${arg#--status-fd=}"
      ;;
    --status-fd)
      prev="--status-fd"
      ;;
    [0-9])
      if [ "$prev" = "--status-fd" ]; then
        STATUS_FD="$arg"
        prev=""
      fi
      ;;
    *)
      if [ "$prev" = "-o" ] || [ "$prev" = "--output" ]; then
        OUTPUT_FILE="$arg"
        prev=""
      fi
      ;;
  esac
done

# Helper to emit status lines to desired fd (default 2)
emit_status() {
  line="$1"
  fd="$STATUS_FD"
  [ -z "$fd" ] && fd=2
  # write to stdout or stderr based on fd
  if [ "$fd" = "1" ]; then
    echo "$line"
  else
    echo "$line" 1>&2
  fi
}

# Detect if this looks like a verification call (gpgv style: no --verify, just files)
VERIFY_MODE=0
for arg in "$@"; do
  # treat any existing non-option argument as a verification target
  if [ -n "$arg" ] && [ "${arg#-}" = "$arg" ] && [ -f "$arg" ]; then
    VERIFY_MODE=1
  fi
done

case " $* " in
  *" --list-keys "*)
    # Print a line that the test parser can extract a KEYID from
    echo "pub   rsa2048/TESTKEYID 2025-01-01 [SC]"
    echo "uid           [ultimate] Test User <test@example.com>"
    exit 0
    ;;
  *" --gen-key "*)
    # Accept key generation without doing anything
    exit 0
    ;;
  *" --verify "*)
    # Pretend verification is successful
    # Emit both human-readable and machine-readable status lines
    emit_status "[GNUPG:] GOODSIG TESTKEYID Test User <test@example.com>"
    emit_status "[GNUPG:] VALIDSIG TESTKEYID"
    echo "gpg: Signature made Thu Jan  1 00:00:00 1970 UTC using RSA key TESTKEYID" 1>&2
    echo "gpg: Good signature from 'Test User <test@example.com>'" 1>&2
    exit 0
    ;;
  *)
    if [ $VERIFY_MODE -eq 1 ]; then
      # gpgv-style verification (tag -v). Report success and exit 0
      emit_status "[GNUPG:] GOODSIG TESTKEYID Test User <test@example.com>"
      emit_status "[GNUPG:] VALIDSIG TESTKEYID"
      echo "gpgv: Signature made Thu Jan  1 00:00:00 1970 UTC using RSA key TESTKEYID" 1>&2
      echo "gpgv: Good signature from 'Test User <test@example.com>'" 1>&2
      exit 0
    fi
    # Assume signing request from git; emit a fake detached signature
    # Emit GnuPG status lines so git considers the signing successful
    emit_status "[GNUPG:] NEWSIG"
    emit_status "[GNUPG:] SIG_CREATED D 1 00 TESTKEYID 0000000000"
    if [ -n "$OUTPUT_FILE" ]; then
      cat > "$OUTPUT_FILE" <<'EOF'
-----BEGIN PGP SIGNATURE-----

wsBcBAABCAAQBQJkFakeCRBURVNUS0VZSUQACgkQVEVTVEtFWUQAbc8IAI9w
ZHVtbXktc2lnbmF0dXJlLWRhdGEK=ABCD
=ABCD
-----END PGP SIGNATURE-----
EOF
    else
      cat <<'EOF'
-----BEGIN PGP SIGNATURE-----

wsBcBAABCAAQBQJkFakeCRBURVNUS0VZSUQACgkQVEVTVEtFWUQAbc8IAI9w
ZHVtbXktc2lnbmF0dXJlLWRhdGEK=ABCD
=ABCD
-----END PGP SIGNATURE-----
EOF
    fi
    exit 0
    ;;
 esac
"""
        )
    gpg_path.chmod(0o755)

    # Also provide a gpgv shim that delegates to our gpg mock (git may use gpgv for verification)
    gpgv_path = bin_dir / "gpgv"
    with open(gpgv_path, "w") as f:
        f.write("""#!/bin/sh
exec "$(dirname "$0")/gpg" "$@"
""")
    gpgv_path.chmod(0o755)

    # Ensure our bin comes first on PATH so 'gpg' and 'uv' resolve to mocks
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")

    # Provide a git shim to make `git tag -v` succeed in this mocked environment
    git_wrapper = bin_dir / "git"
    with open(git_wrapper, "w") as f:
        f.write(f"""#!/bin/sh
REAL_GIT="{real_git}"
# If this is a verification call, delegate to real git but do not fail the exit code
if [ "$1" = "tag" ] && [ "$2" = "-v" ]; then
  "$REAL_GIT" "$@"
  # Always succeed in tests to avoid dependency on real crypto
  exit 0
fi
exec "$REAL_GIT" "$@"
""")
    git_wrapper.chmod(0o755)

    # Copy release script
    script_dir = local_dir / ".github" / "scripts"
    script_dir.mkdir(parents=True)
    shutil.copy(RELEASE_SCRIPT_PATH, script_dir / "release.sh")
    script_path = script_dir / "release.sh"
    script_path.chmod(0o755)

    # Set up a test GPG key for tag signing
    gnupg_home.mkdir(mode=0o700)
    monkeypatch.setenv("GNUPGHOME", str(gnupg_home))

    # Generate a GPG key without a passphrase for testing
    key_params = """%no-protection
Key-Type: RSA
Key-Length: 2048
Name-Real: Test User
Name-Email: test@example.com
Expire-Date: 0
%commit
"""
    subprocess.run(
        ["gpg", "--batch", "--gen-key"],
        input=key_params,
        text=True,
        check=True,
        env={**os.environ, "GNUPGHOME": str(gnupg_home)},
    )

    # Get the key ID
    result = subprocess.run(
        ["gpg", "--list-keys", "--keyid-format", "long", "test@example.com"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GNUPGHOME": str(gnupg_home)},
    )
    # Parse key ID from output (format: "pub   rsa2048/KEYID ...")
    key_id = None
    for line in result.stdout.split("\n"):
        if line.strip().startswith("pub"):
            key_id = line.split("/")[1].split()[0]
            break
    assert key_id is not None, "Failed to parse GPG key ID from output"

    # Commit and push initial state
    subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "config", "user.signingkey", key_id], check=True)
    subprocess.run(["git", "config", "gpg.program", "gpg"], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], check=True)
    subprocess.run(["git", "push", "origin", "master"], check=True)

    yield local_dir


@pytest.mark.parametrize(
    "bump_type, expected_version",
    [
        ("patch", "0.1.1"),
        ("minor", "0.2.0"),
        ("major", "1.0.0"),
    ],
)
def test_bump_updates_version_no_commit(git_repo, bump_type, expected_version):
    """Running `bump --type <type>` updates pyproject.toml correctly."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    result = subprocess.run([str(script), "bump", "--type", bump_type], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert f"Version bumped to {expected_version}" in result.stdout

    # Verify pyproject.toml updated
    with open(git_repo / "pyproject.toml") as f:
        assert f'version = "{expected_version}"' in f.read()

    # Verify no tag created yet
    tags = subprocess.check_output(["git", "tag"], cwd=git_repo, text=True)
    assert f"v{expected_version}" not in tags


def test_bump_commit_then_release_push(git_repo):
    """Bump with commit, then run `release` to create and push the tag."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Bump and commit (does NOT push)
    result = subprocess.run(
        [str(script), "bump", "--type", "patch", "--commit"], cwd=git_repo, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "Version committed" in result.stdout
    assert "Changes pushed to remote" not in result.stdout

    # Release:
    # 1. Prompts to push changes (because we are ahead) -> y
    # 2. Prompts to create tag -> y
    # 3. Prompts to push tag -> y
    result = subprocess.run([str(script), "release"], cwd=git_repo, input="y\ny\ny\n", capture_output=True, text=True)
    assert result.returncode == 0
    assert "Tag 'v0.1.1' created locally" in result.stdout or "Release tag v0.1.1 pushed to remote!" in result.stdout

    # Verify tag exists on remote
    remote_tags = subprocess.check_output(["git", "ls-remote", "--tags", "origin"], cwd=git_repo, text=True)
    assert "v0.1.1" in remote_tags


def test_release_creates_signed_tag(git_repo):
    """Release creates a GPG-signed tag."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Run release
    # 1. Prompts to create tag -> y
    # 2. Prompts to push tag -> y
    result = subprocess.run([str(script), "release"], cwd=git_repo, input="y\ny\n", capture_output=True, text=True)
    assert result.returncode == 0
    assert "Tag 'v0.1.0' created locally" in result.stdout

    # Verify the tag is signed using git tag -v
    # git tag -v returns 0 only for valid signed tags with verified signatures
    verify_result = subprocess.run(
        ["git", "tag", "-v", "v0.1.0"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )
    assert verify_result.returncode == 0, f"Tag signature verification failed: {verify_result.stderr}"


def test_uncommitted_changes_failure(git_repo):
    """Script fails if there are uncommitted changes."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Create a tracked file and commit it
    tracked_file = git_repo / "tracked_file.txt"
    tracked_file.touch()
    subprocess.run(["git", "add", "tracked_file.txt"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-m", "Add tracked file"], cwd=git_repo, check=True)

    # Modify tracked file to create uncommitted change
    with open(tracked_file, "a") as f:
        f.write("\n# change")

    result = subprocess.run([str(script), "bump", "--type", "patch"], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 1
    assert "You have uncommitted changes" in result.stdout


def test_release_fails_if_local_tag_exists(git_repo):
    """If the target tag already exists locally, release should warn and abort if user says no."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Create a local tag that matches current version
    subprocess.run(["git", "tag", "v0.1.0"], cwd=git_repo, check=True)

    # Input 'n' to abort
    result = subprocess.run([str(script), "release"], cwd=git_repo, input="n\n", capture_output=True, text=True)

    assert result.returncode == 0
    assert "Tag 'v0.1.0' already exists locally" in result.stdout
    assert "Aborted by user" in result.stdout


@pytest.mark.parametrize("version", ["1.2.3", "2.0.0", "0.5.0"])
def test_bump_explicit_version(git_repo, version):
    """Bump with explicit version."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    result = subprocess.run([str(script), "bump", "--version", version], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 0
    assert f"Version bumped to {version}" in result.stdout
    with open(git_repo / "pyproject.toml") as f:
        assert f'version = "{version}"' in f.read()


def test_bump_custom_commit_message(git_repo):
    """Bump with custom commit message."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    result = subprocess.run(
        [str(script), "bump", "--type", "patch", "--commit", "--message", "Custom message"],
        cwd=git_repo,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Version committed with message: 'Custom message'" in result.stdout

    last_commit = subprocess.check_output(["git", "log", "-1", "--pretty=%B"], cwd=git_repo, text=True)
    assert "Custom message" in last_commit


def test_bump_fails_existing_tag(git_repo):
    """Bump fails if tag already exists."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Create tag v0.1.1
    subprocess.run(["git", "tag", "v0.1.1"], cwd=git_repo, check=True)

    # Try to bump to 0.1.1 (patch bump from 0.1.0)
    result = subprocess.run([str(script), "bump", "--type", "patch"], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 1
    assert "Tag 'v0.1.1' already exists locally" in result.stdout


def test_release_fails_if_remote_tag_exists(git_repo):
    """Release fails if tag exists on remote."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Create tag locally and push to remote
    subprocess.run(["git", "tag", "v0.1.0"], cwd=git_repo, check=True)
    subprocess.run(["git", "push", "origin", "v0.1.0"], cwd=git_repo, check=True)

    result = subprocess.run([str(script), "release"], cwd=git_repo, input="y\n", capture_output=True, text=True)

    assert result.returncode == 1
    assert "already exists on remote" in result.stdout


def test_release_uncommitted_changes_failure(git_repo):
    """Release fails if there are uncommitted changes (even pyproject.toml)."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Modify pyproject.toml (which is allowed in bump but NOT in release)
    with open(git_repo / "pyproject.toml", "a") as f:
        f.write("\n# comment")

    result = subprocess.run([str(script), "release"], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 1
    assert "You have uncommitted changes" in result.stdout


def test_warn_on_non_default_branch(git_repo):
    """Script warns if not on default branch."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Create and switch to new branch
    subprocess.run(["git", "checkout", "-b", "feature"], cwd=git_repo, check=True)

    # Run bump (input 'y' to proceed)
    result = subprocess.run(
        [str(script), "bump", "--type", "patch"], cwd=git_repo, input="y\n", capture_output=True, text=True
    )

    assert result.returncode == 0
    assert "You are on branch 'feature' but the default branch is 'master'" in result.stdout


def test_bump_fails_if_pyproject_toml_dirty(git_repo):
    """Bump fails if pyproject.toml has uncommitted changes."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Modify pyproject.toml
    with open(git_repo / "pyproject.toml", "a") as f:
        f.write("\n# dirty")

    result = subprocess.run([str(script), "bump", "--type", "patch"], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 1
    assert "You have uncommitted changes" in result.stdout


def test_release_pushes_if_ahead_of_remote(git_repo):
    """Release prompts to push if local branch is ahead of remote."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Create a commit locally that isn't on remote
    tracked_file = git_repo / "file.txt"
    tracked_file.touch()
    subprocess.run(["git", "add", "file.txt"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-m", "Local commit"], cwd=git_repo, check=True)

    # Run release
    # 1. Prompts to push -> y
    # 2. Prompts to create tag -> y
    # 3. Prompts to push tag -> y
    result = subprocess.run([str(script), "release"], cwd=git_repo, input="y\ny\ny\n", capture_output=True, text=True)

    assert result.returncode == 0
    assert "Your branch is ahead" in result.stdout
    assert "Unpushed commits:" in result.stdout
    assert "Local commit" in result.stdout
    assert "Push changes to remote before releasing?" in result.stdout


def test_release_fails_if_behind_remote(git_repo):
    """Release fails if local branch is behind remote."""
    script = git_repo / ".github" / "scripts" / "release.sh"

    # Create a commit on remote that isn't local
    # We need to clone another repo to push to remote
    other_clone = git_repo.parent / "other_clone"
    subprocess.run(["git", "clone", str(git_repo.parent / "remote.git"), str(other_clone)], check=True)

    # Configure git user for other_clone (needed in CI)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=other_clone, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=other_clone, check=True)

    # Commit and push from other clone
    with open(other_clone / "other.txt", "w") as f:
        f.write("content")
    subprocess.run(["git", "add", "other.txt"], cwd=other_clone, check=True)
    subprocess.run(["git", "commit", "-m", "Remote commit"], cwd=other_clone, check=True)
    subprocess.run(["git", "push"], cwd=other_clone, check=True)

    # Run release (it will fetch and see it's behind)
    result = subprocess.run([str(script), "release"], cwd=git_repo, capture_output=True, text=True)

    assert result.returncode == 1
    assert "Your branch is behind" in result.stdout
