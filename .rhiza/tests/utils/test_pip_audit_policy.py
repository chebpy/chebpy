"""Unit tests for pip_audit_policy.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from test_utils import strip_ansi


def _load_module(root: Path):
    """Import the repo's pip_audit_policy.py utility as a standalone module."""
    module_path = root / ".rhiza" / "utils" / "pip_audit_policy.py"
    spec = importlib.util.spec_from_file_location("pip_audit_policy", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pip_audit_policy"] = module
    spec.loader.exec_module(module)
    return module


def test_vuln_ids_includes_primary_id_and_distinct_aliases(root):
    """Vulnerability identifiers should include the primary ID and unique aliases."""
    module = _load_module(root)

    vuln = {"id": "PYSEC-2024-1", "aliases": ["CVE-2024-1234", "PYSEC-2024-1", "GHSA-xxxx-yyyy"]}

    assert module._vuln_ids(vuln) == "PYSEC-2024-1, CVE-2024-1234, GHSA-xxxx-yyyy"


def test_main_returns_zero_and_forwards_args_when_audit_passes(root, monkeypatch, capsys):
    """Successful pip-audit runs should print OK and return zero."""
    module = _load_module(root)
    seen: dict[str, list[str]] = {}

    def _fake_run(cmd: list[str], *, capture_output: bool, text: bool):
        """Record the invoked command and return a passing (returncode 0) pip-audit result."""
        seen["cmd"] = cmd
        assert capture_output is True
        assert text is True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.shutil, "which", lambda name: "/custom/uvx")
    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(module.sys, "argv", ["pip_audit_policy.py", "--ignore-vuln", "CVE-2024-1234"])

    assert module.main() == 0
    assert seen["cmd"] == ["/custom/uvx", "pip-audit", "--format", "json", "--ignore-vuln", "CVE-2024-1234"]
    assert "[OK] pip-audit: no vulnerabilities found" in strip_ansi(capsys.readouterr().out)


def test_main_echoes_raw_output_when_json_parsing_fails(root, monkeypatch, capsys):
    """Non-JSON output should be passed through unchanged and preserve the exit code."""
    module = _load_module(root)

    def _fake_run(*args, **kwargs):
        """Return a failing pip-audit result with non-JSON stdout to exercise the passthrough path."""
        return SimpleNamespace(returncode=2, stdout="oops\n", stderr="bad\n")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(module.sys, "argv", ["pip_audit_policy.py"])

    assert module.main() == 2
    captured = capsys.readouterr()
    assert captured.out == "oops\n"
    assert captured.err == "bad\n"


def test_main_warns_for_tooling_vulnerabilities_without_failing(root, monkeypatch, capsys):
    """Tooling package vulnerabilities should warn but still return success."""
    module = _load_module(root)
    payload = {
        "dependencies": [
            {
                "name": "pip",
                "version": "24.0",
                "vulns": [{"id": "PYSEC-2024-1", "aliases": ["CVE-2024-1234"]}],
            }
        ]
    }

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=json.dumps(payload), stderr=""),
    )
    monkeypatch.setattr(module.sys, "argv", ["pip_audit_policy.py"])

    assert module.main() == 0
    output = strip_ansi(capsys.readouterr().out)
    assert "[WARN] pip==24.0: PYSEC-2024-1, CVE-2024-1234 (tooling — not failing build)" in output
    assert "[FAIL]" not in output


def test_main_fails_for_runtime_vulnerabilities_and_warns_for_tooling(root, monkeypatch, capsys):
    """Runtime package vulnerabilities should fail even when tooling warnings are present."""
    module = _load_module(root)
    payload = {
        "dependencies": [
            {
                "name": "setuptools",
                "version": "70.0",
                "vulns": [{"id": "PYSEC-2024-2", "aliases": []}],
            },
            {
                "name": "requests",
                "version": "2.0.0",
                "vulns": [{"id": "GHSA-abcd", "aliases": ["CVE-2024-5678"]}],
            },
        ]
    }

    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=json.dumps(payload), stderr=""),
    )
    monkeypatch.setattr(module.sys, "argv", ["pip_audit_policy.py"])

    assert module.main() == 1
    output = strip_ansi(capsys.readouterr().out)
    assert "[WARN] setuptools==70.0: PYSEC-2024-2 (tooling — not failing build)" in output
    assert "[FAIL] requests==2.0.0: GHSA-abcd, CVE-2024-5678" in output
