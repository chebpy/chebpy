"""Configuration and fixtures for pytest in the chebfun tests.

This module contains fixtures and helper functions specific to testing
the Chebfun class. It provides fixtures for common test objects and
helper functions for testing various operations.
"""

import pytest

from chebpy.core.chebfun import Chebfun

from ..utilities import eps, sin


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


@pytest.fixture
def emptyfun() -> Chebfun:
    """Create an empty Chebfun function for testing.

    This fixture creates an empty Chebfun object that can be used
    to test the behavior of operations on empty functions.

    Returns:
        Chebfun: An empty Chebfun object
    """
    return Chebfun.initempty()


@pytest.fixture()
def uf1():
    """Identity function."""
    def f(x: float) -> float:
        """Identity function.

        Args:
            x: Input value or array

        Returns:
            x: The input value or array
        """
        return x

    return f

@pytest.fixture()
def uf2():
    """Sine function with offset."""
    def f(x: float) -> float:
        """Sine function with offset.

        Args:
            x: Input value or array

        Returns:
            float or array: sin(x - 0.5)
        """
        return sin(x - 0.5)

    return f

@pytest.fixture()
def uf3():
    """Sine function with scaling and offset."""
    def f(x: float) -> float:
        """Sine function with scaling and offset.

        Args:
            x: Input value or array

        Returns:
            float or array: sin(25 * x - 1)
        """
        return sin(25 * x - 1)

    return f

