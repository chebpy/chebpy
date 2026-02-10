"""Tests for Git LFS template structure and files.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

Verifies that LFS-related files and configurations are present.
"""

import pytest


class TestLFSTemplateStructure:
    """Tests for LFS template file structure."""

    def test_lfs_makefile_exists(self, root):
        """LFS makefile should exist in make.d directory."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        if not lfs_makefile.exists():
            pytest.skip("lfs.mk not found, skipping test")

    def test_lfs_documentation_exists(self, root):
        """LFS documentation should exist."""
        if not (root / ".rhiza" / "make.d" / "lfs.mk").exists():
            pytest.skip("lfs.mk not found, skipping test")

        lfs_doc = root / "docs" / "LFS.md"
        assert lfs_doc.exists(), "LFS.md documentation not found"

    def test_lfs_makefile_has_targets(self, root):
        """LFS makefile should define all expected targets."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        if not lfs_makefile.exists():
            pytest.skip("lfs.mk not found, skipping test")

        content = lfs_makefile.read_text()

        required_targets = [
            "lfs-install:",
            "lfs-pull:",
            "lfs-track:",
            "lfs-status:",
        ]

        for target in required_targets:
            assert target in content, f"Target {target} not found in lfs.mk"

    def test_lfs_makefile_has_phony_declarations(self, root):
        """LFS makefile should declare targets as phony."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        if not lfs_makefile.exists():
            pytest.skip("lfs.mk not found, skipping test")

        content = lfs_makefile.read_text()

        assert ".PHONY:" in content
        assert "lfs-install" in content
        assert "lfs-pull" in content
        assert "lfs-track" in content
        assert "lfs-status" in content

    def test_lfs_makefile_has_help_comments(self, root):
        """LFS makefile should have help comments for targets."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        if not lfs_makefile.exists():
            pytest.skip("lfs.mk not found, skipping test")

        content = lfs_makefile.read_text()

        # Check for ##@ section header
        assert "##@ Git LFS" in content

        # Check for target descriptions
        assert "##" in content

    def test_lfs_documentation_has_sections(self, root):
        """LFS documentation should have all expected sections."""
        if not (root / ".rhiza" / "make.d" / "lfs.mk").exists():
            pytest.skip("lfs.mk not found, skipping test")

        lfs_doc = root / "docs" / "LFS.md"
        content = lfs_doc.read_text()

        expected_sections = [
            "# Git LFS",
            "## Overview",
            "## Available Make Targets",
            "## Typical Workflow",
            "## CI/CD Integration",
            "## Troubleshooting",
        ]

        for section in expected_sections:
            assert section in content, f"Section '{section}' not found in LFS.md"

    def test_lfs_documentation_describes_all_targets(self, root):
        """LFS documentation should describe all make targets."""
        if not (root / ".rhiza" / "make.d" / "lfs.mk").exists():
            pytest.skip("lfs.mk not found, skipping test")

        lfs_doc = root / "docs" / "LFS.md"
        content = lfs_doc.read_text()

        targets = [
            "lfs-install",
            "lfs-pull",
            "lfs-track",
            "lfs-status",
        ]

        for target in targets:
            assert target in content, f"Target {target} not documented in LFS.md"

    def test_lfs_makefile_cross_platform_support(self, root):
        """LFS makefile should support multiple platforms."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        if not lfs_makefile.exists():
            pytest.skip("lfs.mk not found, skipping test")

        content = lfs_makefile.read_text()

        # Check for OS detection
        assert "uname -s" in content
        assert "Darwin" in content
        assert "Linux" in content

        # Check for architecture detection (macOS)
        assert "uname -m" in content
        assert "arm64" in content
        assert "amd64" in content

    def test_lfs_makefile_error_handling(self, root):
        """LFS makefile should include error handling."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        if not lfs_makefile.exists():
            pytest.skip("lfs.mk not found, skipping test")

        content = lfs_makefile.read_text()

        # Check for error messages
        assert "ERROR" in content
        assert "exit 1" in content

    def test_lfs_makefile_uses_color_variables(self, root):
        """LFS makefile should use standard color variables."""
        lfs_makefile = root / ".rhiza" / "make.d" / "lfs.mk"
        if not lfs_makefile.exists():
            pytest.skip("lfs.mk not found, skipping test")

        content = lfs_makefile.read_text()

        # Check for color variable usage
        assert "BLUE" in content
        assert "RED" in content
        assert "RESET" in content


class TestLFSBundleDefinition:
    """Tests for LFS bundle in template-bundles.yml."""

    def test_lfs_bundle_exists(self, root):
        """LFS bundle should be defined in template-bundles.yml."""
        if not (root / ".rhiza" / "make.d" / "lfs.mk").exists():
            pytest.skip("lfs.mk not found, skipping test")

        bundles_file = root / ".rhiza" / "template-bundles.yml"
        assert bundles_file.exists()

        import yaml

        with open(bundles_file) as f:
            bundles = yaml.safe_load(f)

        assert "bundles" in bundles
        assert "lfs" in bundles["bundles"]

    def test_lfs_bundle_has_required_fields(self, root):
        """LFS bundle should have all required fields."""
        if not (root / ".rhiza" / "make.d" / "lfs.mk").exists():
            pytest.skip("lfs.mk not found, skipping test")

        bundles_file = root / ".rhiza" / "template-bundles.yml"

        import yaml

        with open(bundles_file) as f:
            bundles = yaml.safe_load(f)

        lfs_bundle = bundles["bundles"]["lfs"]

        assert "description" in lfs_bundle
        assert "files" in lfs_bundle
        assert isinstance(lfs_bundle["files"], list)

    def test_lfs_bundle_includes_makefile(self, root):
        """LFS bundle should include the makefile."""
        if not (root / ".rhiza" / "make.d" / "lfs.mk").exists():
            pytest.skip("lfs.mk not found, skipping test")

        bundles_file = root / ".rhiza" / "template-bundles.yml"

        import yaml

        with open(bundles_file) as f:
            bundles = yaml.safe_load(f)

        lfs_bundle = bundles["bundles"]["lfs"]
        files = lfs_bundle["files"]

        assert ".rhiza/make.d/lfs.mk" in files

    def test_lfs_bundle_includes_documentation(self, root):
        """LFS bundle should include documentation."""
        if not (root / ".rhiza" / "make.d" / "lfs.mk").exists():
            pytest.skip("lfs.mk not found, skipping test")

        bundles_file = root / ".rhiza" / "template-bundles.yml"

        import yaml

        with open(bundles_file) as f:
            bundles = yaml.safe_load(f)

        lfs_bundle = bundles["bundles"]["lfs"]
        files = lfs_bundle["files"]

        assert "docs/LFS.md" in files

    def test_lfs_bundle_is_standalone(self, root):
        """LFS bundle should be marked as standalone."""
        if not (root / ".rhiza" / "make.d" / "lfs.mk").exists():
            pytest.skip("lfs.mk not found, skipping test")

        bundles_file = root / ".rhiza" / "template-bundles.yml"

        import yaml

        with open(bundles_file) as f:
            bundles = yaml.safe_load(f)

        lfs_bundle = bundles["bundles"]["lfs"]

        assert "standalone" in lfs_bundle
        assert lfs_bundle["standalone"] is True

    def test_lfs_bundle_has_no_requirements(self, root):
        """LFS bundle should have no dependencies."""
        if not (root / ".rhiza" / "make.d" / "lfs.mk").exists():
            pytest.skip("lfs.mk not found, skipping test")

        bundles_file = root / ".rhiza" / "template-bundles.yml"

        import yaml

        with open(bundles_file) as f:
            bundles = yaml.safe_load(f)

        lfs_bundle = bundles["bundles"]["lfs"]

        assert "requires" in lfs_bundle
        assert lfs_bundle["requires"] == []
