#!/usr/bin/env python3
"""Emit the list of supported Python versions from pyproject.toml.

This helper is used in GitHub Actions to compute the test matrix.
"""

import json
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
CANDIDATES = ["3.11", "3.12", "3.13", "3.14"]  # extend as needed


def supported_versions() -> list[str]:
    """Return all supported Python versions declared in pyproject.toml.

    Reads project.requires-python, evaluates candidate versions against the
    specifier, and returns the subset that satisfy the constraint, in ascending order.

    Returns:
        list[str]: The supported versions (e.g., ["3.11", "3.12"]).
    """
    # Load pyproject.toml using the tomllib standard library (Python 3.11+)
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)

    # Extract the requires-python field from project metadata
    # This specifies the Python version constraint (e.g., ">=3.11")
    spec_str = data.get("project", {}).get("requires-python")
    if not spec_str:
        msg = "pyproject.toml: missing 'project.requires-python'"
        raise KeyError(msg)

    # Parse the version specifier (e.g., ">=3.11,<3.14")
    spec = SpecifierSet(spec_str)

    # Filter candidate versions to find which ones satisfy the constraint
    versions: list[str] = []
    for v in CANDIDATES:
        # packaging.version.Version parses the version string
        # The 'in' operator checks if the version satisfies the specifier
        if Version(v) in spec:
            versions.append(v)

    if not versions:
        msg = "pyproject.toml: no supported Python versions match 'project.requires-python'"
        raise ValueError(msg)

    return versions


if __name__ == "__main__":
    # Check if pyproject.toml exists in the expected location
    # If it exists, use it to determine supported versions
    # Otherwise, fall back to returning all candidates (for edge cases)
    if PYPROJECT.exists():
        print(json.dumps(supported_versions()))
    else:
        print(json.dumps(CANDIDATES))
