"""Configuration for pytest."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def root():
    """Fixture for the root directory of the project."""
    return Path(__file__).parent.parent
