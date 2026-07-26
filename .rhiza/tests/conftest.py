"""Pytest configuration and fixtures for the rhiza test suite.

This file and its associated tests flow down via a SYNC action from the jebel-quant/rhiza repository
(https://github.com/jebel-quant/rhiza).

Provides shared session-scoped fixtures (``root`` and ``logger``) used across the test modules.

Security Notes:
- S101 (assert usage): Asserts are appropriate in test code for validating conditions
"""

import logging
import pathlib

import pytest


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
