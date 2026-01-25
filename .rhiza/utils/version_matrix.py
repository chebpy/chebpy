#!/usr/bin/env python3
"""Emit the list of supported Python versions from pyproject.toml.

This helper is used in GitHub Actions to compute the test matrix.
"""

import json
import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
CANDIDATES = ["3.11", "3.12", "3.13", "3.14"]  # extend as needed


class RhizaError(Exception):
    """Base exception for Rhiza-related errors."""


class VersionSpecifierError(RhizaError):
    """Raised when a version string or specifier is invalid."""


class PyProjectError(RhizaError):
    """Raised when there are issues with pyproject.toml configuration."""


def parse_version(v: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers.

    This is intentionally simple and only supports numeric components.
    If a component contains non-numeric suffixes (e.g. '3.11.0rc1'),
    the leading numeric portion will be used (e.g. '0rc1' -> 0). If a
    component has no leading digits at all, a VersionSpecifierError is raised.

    Args:
        v: Version string to parse (e.g., "3.11", "3.11.0rc1").

    Returns:
        Tuple of integers representing the version.

    Raises:
        VersionSpecifierError: If a version component has no numeric prefix.
    """
    parts: list[int] = []
    for part in v.split("."):
        match = re.match(r"\d+", part)
        if not match:
            msg = f"Invalid version component {part!r} in version {v!r}; expected a numeric prefix."
            raise VersionSpecifierError(msg)
        parts.append(int(match.group(0)))
    return tuple(parts)


def _check_operator(version_tuple: tuple[int, ...], op: str, spec_v_tuple: tuple[int, ...]) -> bool:
    """Check if a version tuple satisfies an operator constraint."""
    operators = {
        ">=": lambda v, s: v >= s,
        "<=": lambda v, s: v <= s,
        ">": lambda v, s: v > s,
        "<": lambda v, s: v < s,
        "==": lambda v, s: v == s,
        "!=": lambda v, s: v != s,
    }
    return operators[op](version_tuple, spec_v_tuple)


def satisfies(version: str, specifier: str) -> bool:
    """Check if a version satisfies a comma-separated list of specifiers.

    This is a simplified version of packaging.specifiers.SpecifierSet.
    Supported operators: >=, <=, >, <, ==, !=

    Args:
        version: Version string to check (e.g., "3.11").
        specifier: Comma-separated specifier string (e.g., ">=3.11,<3.14").

    Returns:
        True if the version satisfies all specifiers, False otherwise.

    Raises:
        VersionSpecifierError: If the specifier format is invalid.
    """
    version_tuple = parse_version(version)

    # Split by comma for multiple constraints
    for spec in specifier.split(","):
        spec = spec.strip()
        # Match operator and version part
        match = re.match(r"(>=|<=|>|<|==|!=)\s*([\d.]+)", spec)
        if not match:
            # If no operator, assume ==
            if re.match(r"[\d.]+", spec):
                if version_tuple != parse_version(spec):
                    return False
                continue
            msg = f"Invalid specifier {spec!r}; expected format like '>=3.11' or '3.11'"
            raise VersionSpecifierError(msg)

        op, spec_v = match.groups()
        spec_v_tuple = parse_version(spec_v)

        if not _check_operator(version_tuple, op, spec_v_tuple):
            return False

    return True


def supported_versions() -> list[str]:
    """Return all supported Python versions declared in pyproject.toml.

    Reads project.requires-python, evaluates candidate versions against the
    specifier, and returns the subset that satisfy the constraint, in ascending order.

    Returns:
        list[str]: The supported versions (e.g., ["3.11", "3.12"]).

    Raises:
        PyProjectError: If requires-python is missing or no candidates match.
    """
    # Load pyproject.toml using the tomllib standard library (Python 3.11+)
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)

    # Extract the requires-python field from project metadata
    # This specifies the Python version constraint (e.g., ">=3.11")
    spec_str = data.get("project", {}).get("requires-python")
    if not spec_str:
        msg = "pyproject.toml: missing 'project.requires-python'"
        raise PyProjectError(msg)

    # Filter candidate versions to find which ones satisfy the constraint
    versions: list[str] = []
    for v in CANDIDATES:
        if satisfies(v, spec_str):
            versions.append(v)

    if not versions:
        msg = f"pyproject.toml: no supported Python versions match '{spec_str}'. Evaluated candidates: {CANDIDATES}"
        raise PyProjectError(msg)

    return versions


if __name__ == "__main__":
    # Check if pyproject.toml exists in the expected location
    # If it exists, use it to determine supported versions
    # Otherwise, fall back to returning all candidates (for edge cases)
    if PYPROJECT.exists():
        print(json.dumps(supported_versions()))
    else:
        print(json.dumps(CANDIDATES))
