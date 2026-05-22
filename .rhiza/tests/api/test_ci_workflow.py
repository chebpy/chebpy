"""Tests for the rhiza_ci.yml workflow configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from api.conftest import run_make

WORKFLOW_PATH = Path(".github") / "workflows" / "rhiza_ci.yml"
MULTI_OS_MATRIX = 'RHIZA_CI_OS_MATRIX=["ubuntu-latest","windows-latest"]'


def _load_workflow(root: Path) -> dict:
    """Load and parse the CI workflow YAML file."""
    workflow_file = root / WORKFLOW_PATH
    if not workflow_file.exists():
        pytest.fail(f"Workflow file not found: {workflow_file}")
    with open(workflow_file) as fh:
        return yaml.safe_load(fh)


def test_ci_workflow_uses_generated_os_matrix(root):
    """CI test job must read its OS matrix from generate-matrix job output."""
    workflow = _load_workflow(root)
    test_job = workflow["jobs"]["test"]
    matrix = test_job["strategy"]["matrix"]
    assert matrix["os"] == "${{ fromJson(needs.generate-matrix.outputs.os_matrix) }}"


def test_ci_workflow_defines_os_matrix_output(root):
    """generate-matrix job must expose an os_matrix output from the os step."""
    workflow = _load_workflow(root)
    outputs = workflow["jobs"]["generate-matrix"]["outputs"]
    assert outputs["os_matrix"] == "${{ steps.os.outputs.list }}"


def test_ci_workflow_generates_os_matrix_via_make_target(root):
    """OS matrix generation must delegate to the dedicated Make target."""
    workflow = _load_workflow(root)
    steps = workflow["jobs"]["generate-matrix"]["steps"]
    os_step = next((step for step in steps if step.get("id") == "os"), None)
    assert os_step is not None, "Expected a step with id='os' in generate-matrix job"
    run = os_step["run"]
    assert "ci-os-matrix" in run


def test_ci_os_matrix_make_target_defaults_to_ubuntu_when_env_missing(logger):
    """ci-os-matrix target must default to ubuntu-latest when env value is absent."""
    result = run_make(logger, ["-f", ".rhiza/rhiza.mk", "RHIZA_CI_OS_MATRIX=", "ci-os-matrix"], dry_run=False)
    assert result.returncode == 0
    assert json.loads(result.stdout.strip()) == ["ubuntu-latest"]


def test_ci_os_matrix_make_target_can_be_configured(logger):
    """ci-os-matrix target must use the configured RHIZA_CI_OS_MATRIX value."""
    result = run_make(
        logger,
        ["-f", ".rhiza/rhiza.mk", MULTI_OS_MATRIX, "ci-os-matrix"],
        dry_run=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout.strip()) == ["ubuntu-latest", "windows-latest"]


def test_ci_test_job_retries_uv_install_on_failure(root):
    """Test job must retry uv setup when the first attempt fails."""
    workflow = _load_workflow(root)
    steps = workflow["jobs"]["test"]["steps"]

    install_step = next((step for step in steps if step.get("id") == "install-uv"), None)
    assert install_step is not None, "Expected an install-uv step in test job"
    assert install_step.get("continue-on-error") is True

    retry_step = next((step for step in steps if step.get("name") == "Retry uv installation"), None)
    assert retry_step is not None, "Expected a retry step for uv setup in test job"
    assert retry_step.get("if") == "steps.install-uv.outcome == 'failure'"
