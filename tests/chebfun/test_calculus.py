"""Unit-tests for Chebfun calculus operations.

This module contains tests for the calculus operations of Chebfun,
including sum, diff, cumsum, and dot product.
"""

import numpy as np
import pytest

from chebpy.chebfun import Chebfun

from ..utilities import exp, sin


@pytest.fixture
def calculus_fixtures():
    """Create Chebfun objects for testing calculus operations.

    This fixture creates several Chebfun objects with different characteristics
    for testing various calculus operations.

    Returns:
        dict: Dictionary containing:
            f1: Chebfun representing sin(4x - 1.4)
            f2: Chebfun representing exp(x)
            f3: Chebfun representing x^2
            f4: Chebfun representing x^3
    """

    def f(x):
        return sin(4 * x - 1.4)

    def g(x):
        return exp(x)

    f1 = Chebfun.initfun_adaptive(f, [-1, 1])
    f2 = Chebfun.initfun_adaptive(g, [-1, 1])
    f3 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 1])
    f4 = Chebfun.initfun_adaptive(lambda x: x**3, [-1, 1])

    return {"f1": f1, "f2": f2, "f3": f3, "f4": f4}


def test_sum(calculus_fixtures):
    """Test the sum method of Chebfun objects.

    This test verifies that the sum method correctly computes the definite
    integral of a function over its domain.

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = calculus_fixtures["f1"]
    f2 = calculus_fixtures["f2"]

    # For sin(4x - 1.4), the integral over [-1, 1] is approximately 0.3729
    assert abs(f1.sum() - 0.372895407327895) < 1e-4

    # For exp(x), the integral over [-1, 1] is approximately e - 1/e
    assert abs(f2.sum() - (np.exp(1) - np.exp(-1))) < 1e-10


def test_diff(calculus_fixtures):
    """Test the diff method of Chebfun objects.

    This test verifies that the diff method correctly computes the derivative
    of a function.

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f3 = calculus_fixtures["f3"]
    f4 = calculus_fixtures["f4"]

    # Derivative of x^2 is 2x
    df3 = f3.diff()
    xx = np.linspace(-1, 1, 100)
    assert np.max(np.abs(df3(xx) - 2 * xx)) < 1e-10

    # Derivative of x^3 is 3x^2
    df4 = f4.diff()
    assert np.max(np.abs(df4(xx) - 3 * xx**2)) < 1e-10


def test_cumsum(calculus_fixtures):
    """Test the cumsum method of Chebfun objects.

    This test verifies that the cumsum method correctly computes the indefinite
    integral of a function.

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f3 = calculus_fixtures["f3"]
    f4 = calculus_fixtures["f4"]

    # Indefinite integral of x^2 is x^3/3 + C
    # We need to check that the derivative of the indefinite integral is the original function
    assert (f3.cumsum().diff() - f3).isconst

    # Indefinite integral of x^3 is x^4/4 + C
    assert (f4.cumsum().diff() - f4).isconst


def test_dot(calculus_fixtures):
    """Test the dot method of Chebfun objects.

    This test verifies that the dot method correctly computes the inner product
    of two functions.

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f3 = calculus_fixtures["f3"]
    f4 = calculus_fixtures["f4"]

    # Inner product of x^2 and x^3 over [-1, 1] is 0 (odd function)
    assert abs(f3.dot(f4)) < 1e-10

    # Inner product of x^2 and x^2 over [-1, 1] is 2/5
    assert abs(f3.dot(f3) - 2 / 5) < 1e-10


def test_dot_commute(calculus_fixtures):
    """Test that the dot method is commutative.

    This test verifies that the dot method is commutative,
    i.e., f.dot(g) = g.dot(f).

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = calculus_fixtures["f1"]
    f2 = calculus_fixtures["f2"]

    assert abs(f1.dot(f2) - f2.dot(f1)) < 1e-10


def test_dot_empty(emptyfun, calculus_fixtures):
    """Test the dot method with an empty Chebfun.

    This test verifies that the dot method with an empty Chebfun
    returns 0.

    Args:
        emptyfun: Fixture providing an empty Chebfun object.
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = calculus_fixtures["f1"]

    assert emptyfun.dot(f1) == 0
    assert f1.dot(emptyfun) == 0
