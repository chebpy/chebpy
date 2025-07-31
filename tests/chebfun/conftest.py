"""Configuration and fixtures for pytest in the chebfun tests.

This module contains fixtures and helper functions specific to testing
the Chebfun class. It provides fixtures for common test objects and
helper functions for testing various operations.

Note:
    The emptyfun fixture has been moved to the main conftest.py file
    to provide a generic implementation that works across all test modules.
"""

import pytest

from ..utilities import eps


# domain, test_tolerance
@pytest.fixture()
def testdomains() -> list:
    """List of domains and test tolerances for testing algebraic operations."""
    return [
        ([-1, 1], 5 * eps),
        ([-2, 1], 5 * eps),
        ([-1, 2], 5 * eps),
        ([-5, 9], 35 * eps),
    ]
