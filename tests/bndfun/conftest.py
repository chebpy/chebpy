"""Configuration and fixtures for pytest in the bndfun tests.

This module contains fixtures and helper functions specific to testing
the bndfun class. It provides fixtures for common test objects and
helper functions for testing algebraic operations.
"""

import pytest

from chebpy.core.bndfun import Bndfun


@pytest.fixture
def emptyfun() -> Bndfun:
    """Create an empty Bndfun function for testing.

    This fixture creates an empty Bndfun object that can be used
    to test the behavior of algebraic operations on empty functions.

    Returns:
        Bndfun: An empty Bndfun object
    """
    return Bndfun.initempty()
