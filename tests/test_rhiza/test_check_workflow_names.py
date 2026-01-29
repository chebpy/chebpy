"""Unit tests for .rhiza/scripts/check_workflow_names.py.

Tests the workflow name prefix checker used in pre-commit hooks.
"""

import sys
from pathlib import Path

# Add .rhiza/scripts to path so we can import check_workflow_names
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".rhiza" / "scripts"))

from check_workflow_names import check_file


class TestCheckFile:
    """Tests for check_file function."""

    def test_correct_prefix_returns_true(self, tmp_path):
        """File with correct (RHIZA) prefix returns True."""
        workflow = tmp_path / "workflow.yml"
        workflow.write_text('name: "(RHIZA) MY WORKFLOW"\non: push\n')

        assert check_file(str(workflow)) is True

    def test_missing_prefix_updates_file(self, tmp_path):
        """File without (RHIZA) prefix is updated and returns False."""
        workflow = tmp_path / "workflow.yml"
        workflow.write_text("name: My Workflow\non: push\n")

        result = check_file(str(workflow))

        assert result is False
        content = workflow.read_text()
        assert "(RHIZA) MY WORKFLOW" in content

    def test_missing_name_field_returns_false(self, tmp_path, capsys):
        """File without name field returns False with error message."""
        workflow = tmp_path / "workflow.yml"
        workflow.write_text("on: push\njobs:\n  test:\n    runs-on: ubuntu-latest\n")

        result = check_file(str(workflow))

        assert result is False
        captured = capsys.readouterr()
        assert "missing 'name' field" in captured.out

    def test_invalid_yaml_returns_false(self, tmp_path, capsys):
        """Invalid YAML returns False with error message."""
        workflow = tmp_path / "workflow.yml"
        workflow.write_text("name: test\n  invalid: yaml: syntax:\n")

        result = check_file(str(workflow))

        assert result is False
        captured = capsys.readouterr()
        assert "Error parsing YAML" in captured.out

    def test_empty_file_returns_true(self, tmp_path):
        """Empty YAML file returns True (nothing to check)."""
        workflow = tmp_path / "workflow.yml"
        workflow.write_text("")

        assert check_file(str(workflow)) is True

    def test_preserves_other_content(self, tmp_path):
        """Updating name prefix preserves other file content."""
        original = """name: CI Pipeline
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
"""
        workflow = tmp_path / "workflow.yml"
        workflow.write_text(original)

        check_file(str(workflow))

        content = workflow.read_text()
        # Check name was updated
        assert "(RHIZA) CI PIPELINE" in content
        # Check other content preserved
        assert "branches: [main]" in content
        assert "runs-on: ubuntu-latest" in content
        assert "actions/checkout@v4" in content

    def test_quoted_name_with_prefix(self, tmp_path):
        """File with quoted name containing prefix returns True."""
        workflow = tmp_path / "workflow.yml"
        workflow.write_text('name: "(RHIZA) TEST"\non: push\n')

        assert check_file(str(workflow)) is True

    def test_unquoted_name_with_prefix(self, tmp_path):
        """File with unquoted name containing prefix returns True."""
        workflow = tmp_path / "workflow.yml"
        workflow.write_text("name: (RHIZA) TEST\non: push\n")

        assert check_file(str(workflow)) is True

    def test_name_with_special_characters(self, tmp_path):
        """Name with special characters is handled correctly."""
        workflow = tmp_path / "workflow.yml"
        workflow.write_text("name: Build & Deploy\non: push\n")

        check_file(str(workflow))

        content = workflow.read_text()
        assert "(RHIZA) BUILD & DEPLOY" in content
