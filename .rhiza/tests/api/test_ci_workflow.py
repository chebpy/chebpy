"""Tests for the rhiza_ci.yml workflow configuration."""

from __future__ import annotations

import json

from api.conftest import run_make

MULTI_OS_MATRIX = 'RHIZA_CI_OS_MATRIX=["ubuntu-latest","windows-latest"]'


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
