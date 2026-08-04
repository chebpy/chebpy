"""Pytest configuration and fixtures for the rhiza test suite.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

Provides shared session-scoped fixtures (``root``, ``logger`` and ``latest_tag``) used
across the test modules.

Owned by ``core`` rather than by a language layer: the fixtures resolve paths and read
git, neither of which depends on what the project is written in. That is what lets the
Rust and Go layers ship their own ``.rhiza/tests`` modules without shipping a conftest
each — every profile pairs ``core`` with exactly one language layer, so this file is
always present alongside them.

Security Notes:
- S101 (assert usage): Asserts are appropriate in test code for validating conditions
"""

import logging
import pathlib
import shutil
import subprocess  # nosec B404

import pytest

_GIT = shutil.which("git") or "/usr/bin/git"


@pytest.fixture(scope="session")
def root():
    """Return the repository root directory as a pathlib.Path.

    Used by tests to locate files and scripts relative to the project root.
    """
    return pathlib.Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def logger():
    """Provide a session-scoped logger for tests.

    Returns:
        logging.Logger: Logger configured for the test session.
    """
    return logging.getLogger(__name__)


@pytest.fixture(scope="session")
def latest_tag(root):
    """Return the newest ``vX.Y.Z`` git tag, skipping when the repo has none.

    Shared rather than per-module because each language layer asserts the same thing
    against a different file — ``[project].version``, ``[package].version``, or Go's
    ``Version`` constant — and because every layer's release config derives its current
    version from this tag.

    Args:
        root: Repository root, from the ``root`` fixture.

    Returns:
        str: The highest version tag, e.g. ``v1.3.1``.
    """
    result = subprocess.run(  # nosec B603
        [_GIT, "tag", "--list", "v*", "--sort=-version:refname"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    tags = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not tags:
        pytest.skip("No version tags found in repository")
    return tags[0]
