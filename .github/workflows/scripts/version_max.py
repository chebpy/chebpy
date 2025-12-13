#!/usr/bin/env python3
"""Emit the maximum supported Python version from pyproject.toml.

This helper is used in GitHub Actions to pick a default interpreter.
"""

import json
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"
CANDIDATES = ["3.11", "3.12", "3.13", "3.14"]  # extend as needed


def max_supported_version() -> str:
    """Return the highest supported Python version from pyproject.toml.

    Reads project.requires-python, evaluates candidate versions against the
    specifier, and returns the maximum version that satisfies the constraint.

    Returns:
        str: The maximum Python version (e.g., "3.13") satisfying the spec.
    """
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)

    spec_str = data.get("project", {}).get("requires-python")
    if not spec_str:
        msg = "pyproject.toml: missing 'project.requires-python'"
        raise KeyError(msg)

    spec = SpecifierSet(spec_str)
    max_version = None

    for v in CANDIDATES:
        if Version(v) in spec:
            max_version = v

    if max_version is None:
        msg = "pyproject.toml: no supported Python versions match 'project.requires-python'"
        raise ValueError(msg)

    return max_version


if __name__ == "__main__":
    if PYPROJECT.exists():
        print(json.dumps(max_supported_version()))
    else:
        print(json.dumps("3.13"))
