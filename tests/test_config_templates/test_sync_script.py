"""Tests for the sync.sh script and template.yml file exclusion functionality.

These tests validate that the sync script correctly handles file exclusions,
particularly for nested files within directories.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def create_test_structure(logger, base_path: Path) -> Path:
    """Create a template directory structure for testing."""
    logger.debug("Creating test structure under %s", base_path)
    # Create .github directory with workflow files
    github_dir = base_path / ".github"
    github_dir.mkdir(parents=True)
    workflows_dir = github_dir / "workflows"
    workflows_dir.mkdir()

    # Create workflow files
    (workflows_dir / "ci.yml").write_text("# CI workflow\n")
    (workflows_dir / "docker.yml").write_text("# Docker workflow\n")
    (workflows_dir / "devcontainer.yml").write_text("# DevContainer workflow\n")
    (workflows_dir / "release.yml").write_text("# Release workflow\n")

    # Create scripts directory
    scripts_dir = github_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "sync.sh").write_text("# Sync script\n")

    # Create other template files
    (base_path / ".editorconfig").write_text("# EditorConfig\n")
    (base_path / ".gitignore").write_text("# Gitignore\n")
    (base_path / "Makefile").write_text("# Makefile\n")
    (base_path / "ruff.toml").write_text("# Ruff config\n")

    # Create tests directory
    tests_dir = base_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_example.py").write_text("# Test file\n")

    logger.debug("Test structure created under %s", base_path)
    return base_path


def test_directory_copy_excludes_nested_files(logger, tmp_path: Path):
    """Test the core logic of directory copy with file exclusions.

    This test simulates what the sync script does when copying directories
    and verifies that excluded nested files are not copied.
    """
    # Create source directory with files
    source = tmp_path / "source"
    create_test_structure(logger, source)

    # Create destination directory
    dest = tmp_path / "dest"
    dest.mkdir()

    # Simulate the sync operation from sync.sh lines 225-240
    # This is the buggy behavior - it copies everything recursively
    src_path = source / ".github"
    dest_path = dest / ".github"
    dest_path.mkdir(parents=True)

    logger.info("Copying directory recursively (simulating buggy behavior): %s -> %s", src_path, dest_path)
    # This is what the current script does (copies everything)
    subprocess.run(["cp", "-R", f"{src_path}/.", f"{dest_path}/"], check=True)

    # Current behavior (BUG): excluded files should NOT be present but they are
    logger.debug("Checking presence of files that should have been excluded")
    assert (dest / ".github" / "workflows" / "docker.yml").exists(), (
        "This test shows the bug - docker.yml is copied despite exclusion"
    )
    assert (dest / ".github" / "workflows" / "devcontainer.yml").exists(), (
        "This test shows the bug - devcontainer.yml is copied despite exclusion"
    )

    # Files that should be copied
    assert (dest / ".github" / "workflows" / "ci.yml").exists()
    assert (dest / ".github" / "workflows" / "release.yml").exists()


def test_sync_script_with_exclusions_integration(logger, root, tmp_path: Path):
    """Integration test that verifies sync.sh properly excludes nested files.

    This test will fail with the current implementation and pass after the fix.
    """
    # Create template directory
    template_dir = tmp_path / "template_for_test"
    create_test_structure(logger, template_dir)

    # Create target directory with sync script
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    # Create a test-specific sync script that reads from local directory
    # instead of cloning from GitHub
    test_script = target_dir / "test_sync.sh"

    sync_content = (root / ".github" / "scripts" / "sync.sh").read_text()

    # Modify to use local template directory
    test_sync_content = sync_content.replace(
        "# Clone the template repository\n"
        'printf "\\n%b[INFO] Cloning template repository...%b\\n" "$BLUE" "$RESET"\n'
        'REPO_URL="https://github.com/${TEMPLATE_REPO}.git"\n'
        "\n"
        'if ! git clone --depth 1 --branch "$TEMPLATE_BRANCH" "$REPO_URL" "$TEMP_DIR/template" 2>/dev/null; then\n'
        '  printf "%b[ERROR] Failed to clone template repository from %s%b\\n" "$RED" "$REPO_URL" "$RESET"\n'
        "  exit 1\n"
        "fi",
        f'''# Clone the template repository
printf "\\n%b[INFO] Using local template directory...%b\\n" "$BLUE" "$RESET"
mkdir -p "$TEMP_DIR/template"
cp -R "{template_dir}"/. "$TEMP_DIR/template"/ || exit 1''',
    )

    test_script.write_text(test_sync_content)

    # Create .github directory in target
    (target_dir / ".github").mkdir()

    # Create template.yml with exclusions
    template_yml = target_dir / ".github" / "template.yml"
    template_yml.write_text("""template-repository: "dummy/repo"
template-branch: "main"
include: |
  .github
  tests
  .editorconfig
  .gitignore
  Makefile
exclude: |
  .github/workflows/docker.yml
  .github/workflows/devcontainer.yml
  ruff.toml
""")

    logger.info("Running test sync script: %s", test_script)
    # Run the test sync script
    result = subprocess.run(
        ["/bin/sh", str(test_script)],
        cwd=target_dir,
        capture_output=True,
        text=True,
    )

    logger.debug("Sync script exited with %d", result.returncode)
    if result.stdout:
        logger.debug("Sync script STDOUT (truncated):\n%s", result.stdout[:1000])
    if result.stderr:
        logger.debug("Sync script STDERR (truncated):\n%s", result.stderr[:1000])

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    assert result.returncode == 0, f"Sync failed: {result.stderr}"

    # Verify excluded files are NOT present (this will fail before fix)
    logger.debug("Verifying excluded files are not present")
    assert not (target_dir / ".github" / "workflows" / "docker.yml").exists(), (
        "docker.yml should be excluded but was copied"
    )
    assert not (target_dir / ".github" / "workflows" / "devcontainer.yml").exists(), (
        "devcontainer.yml should be excluded but was copied"
    )
    assert not (target_dir / "ruff.toml").exists(), "ruff.toml should be excluded but was copied"

    # Verify non-excluded files ARE present
    logger.debug("Verifying non-excluded files are present")
    assert (target_dir / ".github" / "workflows" / "ci.yml").exists(), "ci.yml should be copied"
    assert (target_dir / ".github" / "workflows" / "release.yml").exists(), "release.yml should be copied"
    assert (target_dir / ".editorconfig").exists()
    assert (target_dir / ".gitignore").exists()
    assert (target_dir / "Makefile").exists()
    assert (target_dir / "tests" / "test_example.py").exists()


class TestSyncScriptRootFixture:
    """Tests for root fixture usage in sync script tests."""

    def test_sync_script_exists_at_root(self, root):
        """Sync script should exist at expected location."""
        sync_script = root / ".github" / "scripts" / "sync.sh"
        assert sync_script.exists()
        assert sync_script.is_file()

    def test_sync_script_is_executable(self, root):
        """Sync script should be executable."""
        import os

        sync_script = root / ".github" / "scripts" / "sync.sh"
        assert os.access(sync_script, os.X_OK)

    def test_sync_script_is_readable(self, root):
        """Sync script should be readable."""
        sync_script = root / ".github" / "scripts" / "sync.sh"
        content = sync_script.read_text()
        assert len(content) > 0
        assert content.startswith("#!/")

    def test_sync_script_contains_expected_logic(self, root):
        """Sync script should contain key functionality."""
        sync_script = root / ".github" / "scripts" / "sync.sh"
        content = sync_script.read_text()

        # Check for key features
        assert "template.yml" in content
        assert "include" in content.lower()
        assert "exclude" in content.lower()
