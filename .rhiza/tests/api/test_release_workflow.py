"""Tests for the rhiza_release.yml workflow configuration.

Validates that the release workflow is correctly defined, including the
update-changelog job that generates and commits CHANGELOG.md on every release.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW_PATH = Path(".github") / "workflows" / "rhiza_release.yml"
EXPECTED_JOBS = {"tag", "build", "draft-release", "update-changelog", "pypi", "devcontainer", "finalise-release"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_workflow(root: Path) -> dict:
    """Load and parse the release workflow YAML file."""
    workflow_file = root / WORKFLOW_PATH
    if not workflow_file.exists():
        pytest.fail(f"Workflow file not found: {workflow_file}")
    with open(workflow_file) as fh:
        return yaml.safe_load(fh)


def _step_commands(job: dict) -> list[str]:
    """Return all ``run`` strings from a job's steps."""
    return [step["run"] for step in job.get("steps", []) if "run" in step]


def _step_uses(job: dict) -> list[str]:
    """Return all ``uses`` strings from a job's steps."""
    return [step["uses"] for step in job.get("steps", []) if "uses" in step]


# ---------------------------------------------------------------------------
# Structure tests — validate the YAML content of rhiza_release.yml
# ---------------------------------------------------------------------------


class TestReleaseWorkflowStructure:
    """Validate the static content of rhiza_release.yml."""

    @pytest.fixture(scope="class")
    def workflow(self, root):
        """Load and return the parsed release workflow YAML."""
        return _load_workflow(root)

    def test_workflow_file_exists(self, root):
        """Workflow file must exist at the expected path."""
        assert (root / WORKFLOW_PATH).exists()

    def test_workflow_triggers_on_version_tags(self, workflow):
        """Workflow must trigger on version tags (v*)."""
        triggers = workflow.get("on") or workflow.get(True) or {}
        push = triggers.get("push", {})
        tags = push.get("tags", [])
        assert any("v*" in tag for tag in tags), "Workflow must trigger on v* tags"

    def test_workflow_has_contents_write_permission(self, workflow):
        """Workflow must have contents: write permission to push CHANGELOG.md."""
        permissions = workflow.get("permissions", {})
        assert permissions.get("contents") == "write", "Workflow must have contents: write permission"
