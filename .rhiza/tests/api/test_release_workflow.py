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

    def test_workflow_defines_all_expected_jobs(self, workflow):
        """Workflow must define all expected jobs including update-changelog."""
        defined_jobs = set(workflow["jobs"].keys())
        for job in EXPECTED_JOBS:
            assert job in defined_jobs, f"Job '{job}' not found in workflow"

    def test_workflow_has_contents_write_permission(self, workflow):
        """Workflow must have contents: write permission to push CHANGELOG.md."""
        permissions = workflow.get("permissions", {})
        assert permissions.get("contents") == "write", "Workflow must have contents: write permission"


class TestUpdateChangelogJob:
    """Validate the update-changelog job in the release workflow."""

    @pytest.fixture(scope="class")
    def workflow(self, root):
        """Load and return the parsed release workflow YAML."""
        return _load_workflow(root)

    @pytest.fixture(scope="class")
    def job(self, workflow):
        """Return the update-changelog job definition."""
        assert "update-changelog" in workflow["jobs"], "update-changelog job must exist"
        return workflow["jobs"]["update-changelog"]

    def test_update_changelog_job_exists(self, workflow):
        """update-changelog job must be present in the workflow."""
        assert "update-changelog" in workflow["jobs"]

    def test_update_changelog_needs_draft_release(self, job):
        """update-changelog must depend on draft-release to ensure notes are ready."""
        needs = job.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "draft-release" in needs, "update-changelog must depend on draft-release"

    def test_update_changelog_needs_tag(self, job):
        """update-changelog must depend on tag to access the tag output."""
        needs = job.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "tag" in needs, "update-changelog must depend on tag"

    def test_update_changelog_runs_on_ubuntu(self, job):
        """update-changelog must run on ubuntu-latest."""
        assert job.get("runs-on") == "ubuntu-latest"

    def test_update_changelog_checks_out_default_branch(self, job):
        """update-changelog must checkout the default branch, not the tag."""
        steps = job.get("steps", [])
        checkout_steps = [s for s in steps if "uses" in s and "actions/checkout" in s["uses"]]
        assert checkout_steps, "update-changelog must have a checkout step"
        checkout = checkout_steps[0]
        ref = checkout.get("with", {}).get("ref", "")
        assert "default_branch" in ref, "update-changelog must checkout the default branch"

    def test_update_changelog_uses_full_git_history(self, job):
        """update-changelog checkout must fetch full history (fetch-depth: 0) for git-cliff."""
        steps = job.get("steps", [])
        checkout_steps = [s for s in steps if "uses" in s and "actions/checkout" in s["uses"]]
        assert checkout_steps, "update-changelog must have a checkout step"
        checkout = checkout_steps[0]
        fetch_depth = checkout.get("with", {}).get("fetch-depth")
        assert fetch_depth == 0, "update-changelog checkout must use fetch-depth: 0"

    def test_update_changelog_generates_changelog_with_git_cliff(self, job):
        """update-changelog must run git-cliff to generate CHANGELOG.md."""
        cmds = _step_commands(job)
        assert any("git-cliff" in cmd and "CHANGELOG.md" in cmd for cmd in cmds), (
            "update-changelog must run git-cliff to generate CHANGELOG.md"
        )

    def test_update_changelog_commits_and_pushes(self, job):
        """update-changelog must commit and push CHANGELOG.md."""
        cmds = _step_commands(job)
        push_cmds = [cmd for cmd in cmds if "git push" in cmd]
        assert push_cmds, "update-changelog must push CHANGELOG.md to the repository"

    def test_update_changelog_configures_git_identity(self, job):
        """update-changelog must configure git user identity for the commit."""
        cmds = _step_commands(job)
        identity_cmds = [cmd for cmd in cmds if "git config user" in cmd]
        assert identity_cmds, "update-changelog must configure git user identity"

    def test_update_changelog_skips_commit_when_no_changes(self, job):
        """update-changelog must skip commit when CHANGELOG.md has not changed."""
        cmds = _step_commands(job)
        idempotent = any("git diff --staged --quiet" in cmd for cmd in cmds)
        assert idempotent, "update-changelog must check for staged changes before committing to be idempotent"
