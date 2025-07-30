"""Configuration and fixtures for pytest in the chebtech tests.

This module contains fixtures and helper functions specific to testing
the Chebtech2 class. It provides fixtures for common test objects and
helper functions for testing algebraic operations.
"""
import pytest

from chebpy.core.chebtech import Chebtech2


@pytest.fixture
def emptyfun() -> Chebtech2:
    """Create an empty Chebtech2 function for testing.

    This fixture creates an empty Chebtech2 object that can be used
    to test the behavior of algebraic operations on empty functions.

    Returns:
        Chebtech2: An empty Chebtech2 object
    """
    return Chebtech2.initempty()
