"""Tests for the template bundles configuration file.

This file validates the structure and consistency of .rhiza/template-bundles.yml,
ensuring all bundle definitions are properly formatted and reference existing files.
"""

# This test file should not(!) be copied into repositories further downstream
from __future__ import annotations

import tomllib

import pytest
import yaml


@pytest.fixture
def template_bundles_path(root):
    """Return path to template-bundles.yml."""
    return root / ".rhiza" / "template-bundles.yml"


@pytest.fixture
def template_bundles(template_bundles_path):
    """Load and return template bundles configuration."""
    with open(template_bundles_path) as f:
        return yaml.safe_load(f)


class TestTemplateBundlesStructure:
    """Tests for template bundles YAML structure."""

    def test_template_bundles_file_exists(self, template_bundles_path):
        """Template bundles configuration file should exist."""
        assert template_bundles_path.exists()

    def test_template_bundles_is_valid_yaml(self, template_bundles_path):
        """Template bundles file should be valid YAML."""
        with open(template_bundles_path) as f:
            data = yaml.safe_load(f)
            assert data is not None

    def test_has_version_field(self, template_bundles):
        """Template bundles should have a version field."""
        assert "version" in template_bundles
        assert isinstance(template_bundles["version"], str)

    def test_version_matches_pyproject(self, template_bundles, root):
        """Template bundles version should match pyproject.toml version."""
        pyproject_path = root / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)

        pyproject_version = pyproject["project"]["version"]
        bundles_version = template_bundles["version"]

        assert bundles_version == pyproject_version, (
            f"Version mismatch: template-bundles.yml has '{bundles_version}' "
            f"but pyproject.toml has '{pyproject_version}'. "
            "Run 'make bump' to sync versions."
        )

    def test_has_bundles_section(self, template_bundles):
        """Template bundles should have a bundles section."""
        assert "bundles" in template_bundles
        assert isinstance(template_bundles["bundles"], dict)


class TestTemplateBundleDefinitions:
    """Tests for individual bundle definitions."""

    def test_all_bundles_have_required_fields(self, template_bundles):
        """Each bundle should have required fields."""
        bundles = template_bundles.get("bundles", {})
        required_fields = {"description", "files"}

        for bundle_name, bundle_config in bundles.items():
            assert isinstance(bundle_config, dict), f"Bundle {bundle_name} should be a dict"
            for field in required_fields:
                assert field in bundle_config, f"Bundle {bundle_name} missing {field}"

    def test_bundle_descriptions_are_strings(self, template_bundles):
        """Bundle descriptions should be strings."""
        bundles = template_bundles.get("bundles", {})
        for bundle_name, bundle_config in bundles.items():
            assert isinstance(bundle_config["description"], str), f"Bundle {bundle_name} description should be a string"

    def test_bundle_files_are_lists(self, template_bundles):
        """Bundle files should be lists."""
        bundles = template_bundles.get("bundles", {})
        for bundle_name, bundle_config in bundles.items():
            assert isinstance(bundle_config["files"], list), f"Bundle {bundle_name} files should be a list"

    def test_core_bundle_is_marked_required(self, template_bundles):
        """Core bundle should be marked as required."""
        bundles = template_bundles.get("bundles", {})
        assert "core" in bundles
        assert bundles["core"].get("required") is True
