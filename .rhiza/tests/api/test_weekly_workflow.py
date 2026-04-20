"""Tests for the rhiza_weekly.yml workflow and its referenced Makefile targets.

Covers two layers:
- Structural: parse .github/workflows/rhiza_weekly.yml and assert every job,
  trigger, and key step is correctly defined.
- Behavioural: dry-run (make -n) the Makefile targets that the workflow invokes
  (semgrep, security, test) to confirm they are wired up without actually
  running them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from api.conftest import run_make

WORKFLOW_PATH = Path(".github") / "workflows" / "rhiza_weekly.yml"
EXPECTED_JOBS = {"dep-compat-test", "semgrep", "pip-audit", "link-check"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_workflow(root: Path) -> dict:
    """Load and parse the weekly workflow YAML file."""
    workflow_file = root / WORKFLOW_PATH
    if not workflow_file.exists():
        pytest.fail(f"Workflow file not found: {workflow_file}")
    with open(workflow_file) as fh:
        return yaml.safe_load(fh)


def _get_triggers(workflow: dict) -> dict:
    """Return the 'on' / triggers block.

    PyYAML parses the bare YAML keyword ``on`` as Python ``True``, so we look
    up both the string key and the boolean key to be robust.
    """
    return workflow.get("on") or workflow.get(True) or {}


def _step_commands(job: dict) -> list[str]:
    """Return all ``run`` strings from a job's steps."""
    return [step["run"] for step in job.get("steps", []) if "run" in step]


def _step_uses(job: dict) -> list[str]:
    """Return all ``uses`` strings from a job's steps."""
    return [step["uses"] for step in job.get("steps", []) if "uses" in step]


def _step_with_args(job: dict) -> list[dict]:
    """Return all steps that have a ``with`` block."""
    return [step for step in job.get("steps", []) if "with" in step]


# ---------------------------------------------------------------------------
# Structure tests — validate the YAML content of rhiza_weekly.yml
# ---------------------------------------------------------------------------


class TestWeeklyWorkflowStructure:
    """Validate the static content of rhiza_weekly.yml."""

    @pytest.fixture(scope="class")
    def workflow(self, root):
        """Load and return the parsed weekly workflow YAML."""
        return _load_workflow(root)

    # --- top-level keys ---

    def test_workflow_file_exists(self, root):
        """Workflow file must exist at the expected path."""
        assert (root / WORKFLOW_PATH).exists()

    def test_workflow_name(self, workflow):
        """Workflow name must be '(RHIZA) WEEKLY'."""
        assert workflow.get("name") == "(RHIZA) WEEKLY"

    def test_permissions_contents_read(self, workflow):
        """Workflow must declare contents: read permissions."""
        assert workflow.get("permissions", {}).get("contents") == "read"

    # --- triggers ---

    def test_schedule_trigger_present(self, workflow):
        """Workflow must have a schedule trigger."""
        triggers = _get_triggers(workflow)
        assert "schedule" in triggers, "workflow must have a schedule trigger"

    def test_schedule_cron_is_monday_morning(self, workflow):
        """Schedule cron must fire every Monday at 08:00 UTC."""
        schedules = _get_triggers(workflow)["schedule"]
        crons = [entry["cron"] for entry in schedules]
        assert "0 8 * * 1" in crons, f"Expected Monday 08:00 UTC cron, got: {crons}"

    def test_workflow_dispatch_trigger_present(self, workflow):
        """Workflow must support manual dispatch via workflow_dispatch."""
        assert "workflow_dispatch" in _get_triggers(workflow), (
            "workflow must support manual dispatch via workflow_dispatch"
        )

    # --- jobs present ---

    def test_all_expected_jobs_defined(self, workflow):
        """All four expected jobs must be defined in the workflow."""
        defined = set(workflow.get("jobs", {}).keys())
        missing = EXPECTED_JOBS - defined
        assert not missing, f"Missing jobs in rhiza_weekly.yml: {missing}"

    def test_no_unexpected_jobs(self, workflow):
        """No jobs beyond the four expected ones should exist."""
        defined = set(workflow.get("jobs", {}).keys())
        extra = defined - EXPECTED_JOBS
        assert not extra, f"Unexpected jobs found in rhiza_weekly.yml: {extra}"

    # --- dep-compat-test job ---

    def test_dep_compat_test_checks_out_with_lfs(self, workflow):
        """dep-compat-test must checkout with LFS enabled."""
        job = workflow["jobs"]["dep-compat-test"]
        checkout_steps = [s for s in job.get("steps", []) if "actions/checkout" in s.get("uses", "")]
        assert checkout_steps, "dep-compat-test must have a checkout step"
        assert checkout_steps[0].get("with", {}).get("lfs") is True, "dep-compat-test checkout must enable LFS"

    def test_dep_compat_test_runs_uv_sync_upgrade(self, workflow):
        """dep-compat-test must run 'uv sync --upgrade' to resolve latest deps."""
        job = workflow["jobs"]["dep-compat-test"]
        cmds = _step_commands(job)
        assert any("uv sync" in cmd and "--upgrade" in cmd for cmd in cmds), (
            "dep-compat-test must run 'uv sync --upgrade'"
        )

    def test_dep_compat_test_runs_make_test(self, workflow):
        """dep-compat-test must invoke 'make test' after resolving dependencies."""
        job = workflow["jobs"]["dep-compat-test"]
        cmds = _step_commands(job)
        assert any("make test" in cmd for cmd in cmds), "dep-compat-test must invoke 'make test'"

    def test_dep_compat_test_runs_on_ubuntu(self, workflow):
        """dep-compat-test must run on ubuntu-latest."""
        job = workflow["jobs"]["dep-compat-test"]
        assert job.get("runs-on") == "ubuntu-latest"

    # --- semgrep job ---

    def test_semgrep_runs_make_semgrep(self, workflow):
        """Semgrep job must invoke 'make semgrep'."""
        job = workflow["jobs"]["semgrep"]
        cmds = _step_commands(job)
        assert any("make semgrep" in cmd for cmd in cmds), "semgrep job must invoke 'make semgrep'"

    def test_semgrep_runs_on_ubuntu(self, workflow):
        """Semgrep job must run on ubuntu-latest."""
        job = workflow["jobs"]["semgrep"]
        assert job.get("runs-on") == "ubuntu-latest"

    # --- pip-audit job ---

    def test_pip_audit_runs_uvx_pip_audit(self, workflow):
        """pip-audit job must invoke pip-audit."""
        job = workflow["jobs"]["pip-audit"]
        cmds = _step_commands(job)
        assert any("pip-audit" in cmd for cmd in cmds), "pip-audit job must invoke pip-audit"

    def test_pip_audit_runs_on_ubuntu(self, workflow):
        """pip-audit job must run on ubuntu-latest."""
        job = workflow["jobs"]["pip-audit"]
        assert job.get("runs-on") == "ubuntu-latest"

    # --- link-check job ---

    def test_link_check_uses_lychee_action(self, workflow):
        """link-check job must use the lycheeverse/lychee-action action."""
        job = workflow["jobs"]["link-check"]
        uses = _step_uses(job)
        assert any("lycheeverse/lychee-action" in u for u in uses), "link-check job must use lycheeverse/lychee-action"

    def test_link_check_targets_readme(self, workflow):
        """link-check job must target README.md."""
        job = workflow["jobs"]["link-check"]
        with_steps = _step_with_args(job)
        readme_found = any("README.md" in str(s.get("with", {}).get("args", "")) for s in with_steps)
        assert readme_found, "link-check job must target README.md"

    def test_link_check_fails_on_broken_links(self, workflow):
        """link-check job must set fail: true to break CI on broken links."""
        job = workflow["jobs"]["link-check"]
        with_steps = _step_with_args(job)
        fail_set = any(s.get("with", {}).get("fail") is True for s in with_steps)
        assert fail_set, "link-check job must set fail: true to break CI on broken links"

    def test_link_check_runs_on_ubuntu(self, workflow):
        """link-check job must run on ubuntu-latest."""
        job = workflow["jobs"]["link-check"]
        assert job.get("runs-on") == "ubuntu-latest"


# ---------------------------------------------------------------------------
# Makefile dry-run tests — verify the targets invoked by the workflow compile
# ---------------------------------------------------------------------------


class TestWeeklyWorkflowMakeTargets:
    """Dry-run the Makefile targets that rhiza_weekly.yml invokes."""

    def test_semgrep_target_dry_run(self, logger):
        """Make semgrep must parse and plan without error."""
        result = run_make(logger, ["semgrep"])
        assert result.returncode == 0

    def test_test_target_dry_run(self, logger):
        """Make test must parse and plan without error."""
        result = run_make(logger, ["test"])
        assert result.returncode == 0

    def test_security_target_invokes_pip_audit(self, logger):
        """Make security dry-run must include a pip-audit invocation."""
        result = run_make(logger, ["security"])
        assert result.returncode == 0
        assert "pip-audit" in result.stdout

    def test_pip_audit_args_forwarded(self, logger):
        """PIP_AUDIT_ARGS variable must be forwarded to the pip-audit call."""
        result = run_make(logger, ["security", "PIP_AUDIT_ARGS=--ignore-vuln TEST-0001"])
        assert result.returncode == 0
        assert "--ignore-vuln TEST-0001" in result.stdout

    def test_semgrep_target_in_help(self, logger):
        """Semgrep target must appear in make help output."""
        result = run_make(logger, ["help"], dry_run=False)
        assert "semgrep" in result.stdout

    def test_security_target_in_help(self, logger):
        """Security target must appear in make help output."""
        result = run_make(logger, ["help"], dry_run=False)
        assert "security" in result.stdout
