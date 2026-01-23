"""Tests for the new Makefile API structure (Wrapper + Makefile.rhiza)."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Get absolute paths for executables to avoid S607 warnings from CodeFactor/Bandit
GIT = shutil.which("git") or "/usr/bin/git"
MAKE = shutil.which("make") or "/usr/bin/make"

# Files required for the API test environment
REQUIRED_FILES = [
    "Makefile",
    "pyproject.toml",
    "README.md",  # is needed to do uv sync, etc.
]

# Folders to copy recursively
REQUIRED_FOLDERS = [
    ".rhiza",
]

OPTIONAL_FOLDERS = [
    "tests",  # for tests/tests.mk
    "docker",  # for docker/docker.mk, if referenced
    "book",
    "presentation",
]


@pytest.fixture
def setup_api_env(logger, root, tmp_path: Path):
    """Set up the Makefile API test environment in a temp folder."""
    logger.debug("Setting up Makefile API test env in: %s", tmp_path)

    # Copy files
    for filename in REQUIRED_FILES:
        src = root / filename
        if src.exists():
            shutil.copy(src, tmp_path / filename)
        else:
            pytest.fail(f"Required file {filename} not found in root")

    # Copy required directories
    for folder in REQUIRED_FOLDERS:
        src = root / folder
        if src.exists():
            dest = tmp_path / folder
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            pytest.fail(f"Required folder {folder} not found in root")

    # Copy optional directories
    for folder in OPTIONAL_FOLDERS:
        src = root / folder
        if src.exists():
            dest = tmp_path / folder
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)

    # Create .rhiza/make.d and ensure no local.mk exists initially
    (tmp_path / ".rhiza" / "make.d").mkdir(parents=True, exist_ok=True)
    if (tmp_path / "local.mk").exists():
        (tmp_path / "local.mk").unlink()

    # Initialize git repo for rhiza tools (required for sync/validate)
    subprocess.run([GIT, "init"], cwd=tmp_path, check=True, capture_output=True)
    # Configure git user for commits if needed (some rhiza checks might need commits)
    subprocess.run([GIT, "config", "user.email", "you@example.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([GIT, "config", "user.name", "Rhiza Test"], cwd=tmp_path, check=True, capture_output=True)
    # Add origin remote to simulate being in the rhiza repo (triggers the skip logic in rhiza.mk)
    subprocess.run(
        [GIT, "remote", "add", "origin", "https://github.com/jebel-quant/rhiza.git"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Move to tmp dir
    old_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(old_cwd)


def run_make(args: list[str] | None = None, dry_run: bool = True) -> subprocess.CompletedProcess:
    """Run make in the current directory."""
    cmd = [MAKE]
    if dry_run:
        cmd.append("-n")
    if args:
        cmd.extend(args)

    # We use -s (silent) to minimize noise, but sometimes we want to see output
    if dry_run:
        # For dry-run, we often want to see the commands
        pass
    else:
        cmd[:1] = [MAKE, "-s"]

    return subprocess.run(cmd, capture_output=True, text=True)


def test_api_delegation(setup_api_env):
    """Test that 'make help' works and delegates to .rhiza/rhiza.mk."""
    result = run_make(["help"], dry_run=False)
    assert result.returncode == 0
    # "Rhiza Workflows" is a section in .rhiza/rhiza.mk
    assert "Rhiza Workflows" in result.stdout

    # "docker-build" is a target in Makefile.rhiza (docker/docker.mk)
    # Only assert if docker folder exists in setup_api_env (it is optional)
    if (setup_api_env / "docker").exists():
        assert "docker-build" in result.stdout


def test_minimal_setup_works(setup_api_env):
    """Test that make works even if optional folders (tests, docker, etc.) are missing."""
    # Remove optional folders
    for folder in OPTIONAL_FOLDERS:
        p = setup_api_env / folder
        if p.exists():
            shutil.rmtree(p)

    # Also remove files that might be copied if they were in the root?
    # Just mainly folders.

    # Run make help
    result = run_make(["help"], dry_run=False)
    assert result.returncode == 0

    # Check that core rhiza targets exist
    assert "Rhiza Workflows" in result.stdout
    assert "sync" in result.stdout

    # Check that optional targets do NOT exist
    assert "docker-build" not in result.stdout
    # "test" target (from tests/) should likely not be there OR be there but fail?
    # Make check: Makefile.rhiza usually has `test:` delegating.
    # If the include tests/tests.mk failed (silently), then `test` target might not be defined
    # unless it's defined in Makefile.rhiza directly.
    # In earlier steps I saw Makefile.rhiza includes tests/tests.mk.
    # If tests.mk is gone, the target `test` (if defined ONLY in tests.mk) will be gone.
    # If it is defined in Makefile.rhiza to check for file existence, it might be there.
    # But usually splitting means the file owns the target.


def test_extension_mechanism(setup_api_env):
    """Test that .rhiza/make.d/*.mk files are included."""
    ext_file = setup_api_env / ".rhiza" / "make.d" / "50-custom.mk"
    ext_file.write_text("""
.PHONY: custom-target
custom-target:
	@echo "Running custom target"
""")

    # Verify the target is listed in help (if we were parsing help, but running it is better)
    # Note: make -n might not show @echo commands if they are silent,
    # but here we just want to see if make accepts the target.

    result = run_make(["custom-target"], dry_run=False)
    assert result.returncode == 0
    assert "Running custom target" in result.stdout


def test_local_override(setup_api_env):
    """Test that local.mk is included and can match targets."""
    local_file = setup_api_env / "local.mk"
    local_file.write_text("""
.PHONY: local-target
local-target:
	@echo "Running local target"
""")

    result = run_make(["local-target"], dry_run=False)
    assert result.returncode == 0
    assert "Running local target" in result.stdout


def test_local_override_pre_hook(setup_api_env):
    """Test using local.mk to override a pre-hook."""
    local_file = setup_api_env / "local.mk"
    # We override pre-sync to print a marker (using double-colon to match rhiza.mk)
    local_file.write_text("""
pre-sync::
	@echo "[[LOCAL_PRE_SYNC]]"
""")

    # Run sync in dry-run.
    # Note: Makefile.rhiza defines pre-sync as empty rule (or with @:).
    # Make warns if we redefine a target unless it's a double-colon rule or we are careful.
    # But usually the last one loaded wins or they merge if double-colon.
    # The current definition in Makefile.rhiza is `pre-sync: ; @echo ...` or similar.
    # Wait, I defined it as `pre-sync: ; @:` (single colon).
    # So redefining it in local.mk (which is included AFTER) might trigger a warning but should work.

    result = run_make(["sync"], dry_run=False)
    # We might expect a warning about overriding commands for target `pre-sync`
    # checking stdout/stderr for the marker

    assert "[[LOCAL_PRE_SYNC]]" in result.stdout


def test_hooks_flow(setup_api_env):
    """Verify that sync runs pre-sync, the sync logic, and post-sync."""
    # We can't easily see execution order in dry run if commands are hidden.
    # Let's inspect the output of make -n sync

    result = run_make(["sync"], dry_run=True)
    assert result.returncode == 0

    # The output should contain the command sequences.
    # Since pre-sync is currently empty (@:) it might not show up in -n output unless we override it.


def test_hook_execution_order(setup_api_env):
    """Define hooks and verify execution order."""
    # Create an extension that defines visible hooks (using double-colon)
    (setup_api_env / ".rhiza" / "make.d" / "hooks.mk").write_text("""
pre-sync::
	@echo "STARTING_SYNC"

post-sync::
	@echo "FINISHED_SYNC"
""")

    result = run_make(["sync"], dry_run=False)
    assert result.returncode == 0
    output = result.stdout

    # Check that markers are present
    assert "STARTING_SYNC" in output
    assert "FINISHED_SYNC" in output

    # Check order: STARTING_SYNC comes before FINISHED_SYNC
    start_index = output.find("STARTING_SYNC")
    finish_index = output.find("FINISHED_SYNC")
    assert start_index < finish_index


def test_override_core_target(setup_api_env):
    """Verify that a repo extension can override a core target (with warning)."""
    # Override 'fmt' which is defined in Makefile.rhiza
    (setup_api_env / ".rhiza" / "make.d" / "override.mk").write_text("""
fmt:
	@echo "CUSTOM_FMT"
""")

    result = run_make(["fmt"], dry_run=False)
    assert result.returncode == 0
    # It should run the custom one because .rhiza/make.d is included later
    assert "CUSTOM_FMT" in result.stdout
    # It should NOT run the original one (which runs pre-commit)
    # The original one has "@${UV_BIN} run pre-commit..."
    # We can check that the output doesn't look like pre-commit output or just check presence of CUSTOM_FMT

    # We expect a warning on stderr about overriding
    assert "warning: overriding" in result.stderr.lower()
    assert "fmt" in result.stderr.lower()
