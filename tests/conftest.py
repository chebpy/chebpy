"""Shared pytest fixtures for the test suite.

Provides the 'root' fixture that returns the repository root as a pathlib.Path,
enabling tests to locate files and scripts relative to the project root.
"""

import logging
import pathlib

import pytest


@pytest.fixture(scope="session")
def root():
    """Return the repository root directory as a pathlib.Path.

    Used by tests to locate files and scripts relative to the project root.
    """
    return pathlib.Path(__file__).parent.parent


@pytest.fixture(scope="session")
def logger():
    """Provide a session-scoped logger for tests.

    Returns:
        logging.Logger: Logger configured for the test session.
    """
    return logging.getLogger(__name__)
