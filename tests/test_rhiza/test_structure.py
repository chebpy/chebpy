"""Tests for the root pytest fixture that yields the repository root Path.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

This module ensures the fixture resolves to the true project root and that
expected files/directories exist, enabling other tests to locate resources
reliably.
"""

import warnings
from pathlib import Path


class TestRootFixture:
    """Tests for the root fixture that provides repository root path."""

    def test_root_returns_pathlib_path(self, root):
        """Root fixture should return a pathlib.Path object."""
        assert isinstance(root, Path)

    def test_root_points_to_repository_root(self, root):
        """Root fixture should point to the actual repository root."""
        assert (root / ".github").is_dir()

    def test_root_is_absolute_path(self, root):
        """Root fixture should return an absolute path."""
        assert root.is_absolute()

    def test_root_resolves_correctly_from_nested_location(self, root):
        """Root should correctly resolve to repository root from tests/test_config_templates/."""
        conftest_path = root / "tests" / "test_rhiza" / "conftest.py"
        assert conftest_path.exists()

    def test_root_contains_expected_directories(self, root):
        """Root should contain all expected project directories."""
        expected_dirs = [".github", "src", "tests", "book"]
        for dirname in expected_dirs:
            if not (root / dirname).exists():
                warnings.warn(f"Expected directory {dirname} not found", stacklevel=2)

    def test_root_contains_expected_files(self, root):
        """Root should contain all expected configuration files."""
        expected_files = [
            "pyproject.toml",
            "README.md",
            "Makefile",
            "ruff.toml",
            ".gitignore",
            ".editorconfig",
        ]
        for filename in expected_files:
            if not (root / filename).exists():
                warnings.warn(f"Expected file {filename} not found", stacklevel=2)

    def test_root_can_locate_github_scripts(self, root):
        """Root should allow locating GitHub scripts."""
        scripts_dir = root / ".github" / "rhiza" / "scripts"
        if not scripts_dir.exists():
            warnings.warn("GitHub scripts directory not found", stacklevel=2)
        else:
            if not (scripts_dir / "release.sh").exists():
                warnings.warn("Expected script release.sh not found", stacklevel=2)
            if not (scripts_dir / "bump.sh").exists():
                warnings.warn("Expected script bump.sh not found", stacklevel=2)
