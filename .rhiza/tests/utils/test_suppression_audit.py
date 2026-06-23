"""Unit tests for suppression_audit.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module(root: Path):
    """Import the repo's suppression_audit.py utility as a standalone module."""
    module_path = root / ".rhiza" / "utils" / "suppression_audit.py"
    spec = importlib.util.spec_from_file_location("suppression_audit", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["suppression_audit"] = module
    spec.loader.exec_module(module)
    return module


def test_nosec_cves_extracts_only_nosec_entries(root):
    """Only # nosec suppressions with CVE tags should be captured."""
    module = _load_module(root)

    suppressions = [
        module.Suppression(file="a.py", line_no=1, kind="nosec", raw="# nosec B101 CVE-2024-1234"),
        module.Suppression(file="b.py", line_no=2, kind="noqa", raw="# noqa: E501 CVE-2024-0001"),
        module.Suppression(file="c.py", line_no=3, kind="nosec", raw="# nosec B602"),
    ]

    assert module._nosec_cves(suppressions) == {"CVE-2024-1234"}


def test_active_pip_audit_ids_collects_ids_and_aliases(root, monkeypatch):
    """pip-audit JSON IDs and aliases should be normalized and returned."""
    module = _load_module(root)

    payload = """
    {
      "dependencies": [
        {"name": "pkg", "vulns": [{"id": "PYSEC-1", "aliases": ["CVE-2024-1111"]}]}
      ]
    }
    """

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=payload, stderr=""),
    )

    assert module._active_pip_audit_ids([]) == {"PYSEC-1", "CVE-2024-1111"}
