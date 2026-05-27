"""Tests for the RHIZA_SYNC_SCHEDULE override mechanism.

These tests validate that users can override the default sync schedule
(cron expression) used in the GitHub Actions sync workflow, and that
the override is applied correctly during `make sync`.

Security Notes:
- S101 (assert usage): Asserts are used in pytest tests to validate conditions
- S603/S607 (subprocess usage): Any subprocess calls are for testing sync targets
  in isolated environments with controlled inputs
- Test code operates in a controlled environment with trusted inputs
"""

from __future__ import annotations

from pathlib import Path

from sync.conftest import run_make, strip_ansi


class TestSyncScheduleVariable:
    """Tests for the RHIZA_SYNC_SCHEDULE Makefile variable."""

    def test_default_sync_schedule_value(self, logger):
        """RHIZA_SYNC_SCHEDULE should default to '0 0 * * 1' (weekly Monday)."""
        proc = run_make(logger, ["print-RHIZA_SYNC_SCHEDULE"], dry_run=False)
        out = strip_ansi(proc.stdout)
        assert "Value of RHIZA_SYNC_SCHEDULE:" in out
        assert "0 0 * * 1" in out

    def test_sync_schedule_overridable_via_env(self, logger, tmp_path: Path):
        """RHIZA_SYNC_SCHEDULE should be overridable via environment variable."""
        import os

        env = os.environ.copy()
        env["RHIZA_SYNC_SCHEDULE"] = "0 9 * * 1-5"

        proc = run_make(logger, ["print-RHIZA_SYNC_SCHEDULE"], dry_run=False, env=env)
        out = strip_ansi(proc.stdout)
        assert "0 9 * * 1-5" in out

    def test_sync_schedule_overridable_via_makefile(self, logger, tmp_path: Path):
        """RHIZA_SYNC_SCHEDULE should be overridable in root Makefile."""
        makefile = tmp_path / "Makefile"
        original = makefile.read_text()
        new_content = "RHIZA_SYNC_SCHEDULE = 0 6 * * *\n\n" + original
        makefile.write_text(new_content)

        proc = run_make(logger, ["print-RHIZA_SYNC_SCHEDULE"], dry_run=False)
        out = strip_ansi(proc.stdout)
        assert "0 6 * * *" in out


class TestApplySyncSchedule:
    """Tests for the _apply-sync-schedule target."""

    def test_apply_sync_schedule_skips_when_default(self, logger, tmp_path: Path):
        """_apply-sync-schedule should not modify files when using default schedule."""
        # Create a mock workflow file matching the actual rhiza_sync.yml format
        workflow_dir = tmp_path / ".github" / "workflows"
        workflow_dir.mkdir(parents=True)
        workflow_file = workflow_dir / "rhiza_sync.yml"
        original_content = "on:\n  schedule:\n    - cron: '0 0 * * 1'  # Weekly on Monday\n"
        workflow_file.write_text(original_content)

        proc = run_make(logger, ["_apply-sync-schedule"], dry_run=False)
        assert proc.returncode == 0

        # File should remain unchanged
        assert workflow_file.read_text() == original_content

    def test_apply_sync_schedule_patches_workflow(self, logger, tmp_path: Path):
        """_apply-sync-schedule should patch workflow when schedule is overridden."""
        # Create a mock workflow file matching the actual rhiza_sync.yml format
        workflow_dir = tmp_path / ".github" / "workflows"
        workflow_dir.mkdir(parents=True)
        workflow_file = workflow_dir / "rhiza_sync.yml"
        workflow_file.write_text("on:\n  schedule:\n    - cron: '0 0 * * 1'  # Weekly on Monday\n")

        # Override the schedule via Makefile
        makefile = tmp_path / "Makefile"
        original = makefile.read_text()
        new_content = "RHIZA_SYNC_SCHEDULE = 0 9 * * 1-5\n\n" + original
        makefile.write_text(new_content)

        proc = run_make(logger, ["_apply-sync-schedule"], dry_run=False)
        assert proc.returncode == 0

        # File should be patched
        patched = workflow_file.read_text()
        assert "0 9 * * 1-5" in patched
        assert "0 0 * * 1" not in patched

    def test_apply_sync_schedule_handles_missing_workflow(self, logger, tmp_path: Path):
        """_apply-sync-schedule should succeed even if workflow file is missing."""
        # Override the schedule but don't create workflow file
        makefile = tmp_path / "Makefile"
        original = makefile.read_text()
        new_content = "RHIZA_SYNC_SCHEDULE = 0 6 * * *\n\n" + original
        makefile.write_text(new_content)

        proc = run_make(logger, ["_apply-sync-schedule"], dry_run=False)
        assert proc.returncode == 0

    def test_apply_sync_schedule_prints_info(self, logger, tmp_path: Path):
        """_apply-sync-schedule should print info message when patching."""
        # Create a mock workflow file matching the actual rhiza_sync.yml format
        workflow_dir = tmp_path / ".github" / "workflows"
        workflow_dir.mkdir(parents=True)
        workflow_file = workflow_dir / "rhiza_sync.yml"
        workflow_file.write_text("on:\n  schedule:\n    - cron: '0 0 * * 1'\n")

        # Override the schedule
        makefile = tmp_path / "Makefile"
        original = makefile.read_text()
        new_content = "RHIZA_SYNC_SCHEDULE = 0 12 * * 0\n\n" + original
        makefile.write_text(new_content)

        proc = run_make(logger, ["_apply-sync-schedule"], dry_run=False)
        out = strip_ansi(proc.stdout)
        assert "Applied custom sync schedule" in out
        assert "0 12 * * 0" in out

    def test_sync_target_calls_apply_schedule(self, logger):
        """The sync target should include _apply-sync-schedule in dry-run output."""
        proc = run_make(logger, ["sync"])
        out = proc.stdout
        assert "_apply-sync-schedule" in out
