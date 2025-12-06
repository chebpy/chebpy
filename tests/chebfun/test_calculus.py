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


def test_diff_order(calculus_fixtures):
    """Test the diff method with integer order argument.

    This test verifies that diff(n) correctly computes the n-th derivative
    of a function.

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f3 = calculus_fixtures["f3"]  # x^2
    f4 = calculus_fixtures["f4"]  # x^3

    # Second derivative of x^2 is 2 (constant)
    d2f3 = f3.diff(2)
    xx = np.linspace(-1, 1, 100)
    assert np.max(np.abs(d2f3(xx) - 2)) < 1e-10

    # Second derivative of x^3 is 6x
    d2f4 = f4.diff(2)
    assert np.max(np.abs(d2f4(xx) - 6 * xx)) < 1e-10

    # Third derivative of x^3 is 6 (constant)
    d3f4 = f4.diff(3)
    assert np.max(np.abs(d3f4(xx) - 6)) < 1e-10

    # Fourth derivative of x^3 should be zero
    d4f4 = f4.diff(4)
    assert np.max(np.abs(d4f4(xx))) < 1e-10


def test_diff_successive(calculus_fixtures):
    """Test that successive diff() calls are equivalent to diff(n).

    This test verifies that f.diff().diff() is equivalent to f.diff(2).

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f4 = calculus_fixtures["f4"]  # x^3

    # Two successive diff() calls
    d2f4_successive = f4.diff().diff()

    # Single diff(2) call
    d2f4_direct = f4.diff(2)

    # Compare results
    xx = np.linspace(-1, 1, 100)
    assert np.max(np.abs(d2f4_successive(xx) - d2f4_direct(xx))) < 1e-10


def test_norm(calculus_fixtures):
    """Test the norm method of Chebfun objects.

    This test verifies that the norm method correctly computes the L2 norm
    of a function, which is defined as sqrt(integral(f^2)).

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = calculus_fixtures["f1"]
    f2 = calculus_fixtures["f2"]
    f3 = calculus_fixtures["f3"]  # x^2

    # Norm of exp(x) over [-1, 1]
    # integral(exp(2x)) = [exp(2x)/2] from -1 to 1 = (e^2 - e^{-2})/2
    expected_norm_f2 = np.sqrt((np.exp(2) - np.exp(-2)) / 2)
    assert abs(f2.norm() - expected_norm_f2) < 1e-10

    # Norm of x^2 over [-1, 1]
    # integral(x^4) = 2/5 for [-1, 1]
    expected_norm_f3 = np.sqrt(2 / 5)
    assert abs(f3.norm() - expected_norm_f3) < 1e-10

    # Norm should always be non-negative
    assert f1.norm() >= 0
    assert f2.norm() >= 0


def test_norm_empty(emptyfun):
    """Test the norm method with an empty Chebfun.

    This test verifies that the norm of an empty Chebfun is 0.

    Args:
        emptyfun: Fixture providing an empty Chebfun object.
    """
    assert emptyfun.norm() == 0


def test_norm_relation_to_dot(calculus_fixtures):
    """Test the relationship between norm and dot product.

    This test verifies that ||f||^2 = f.dot(f).

    Args:
        calculus_fixtures: Fixture providing test Chebfun objects.
    """
    f2 = calculus_fixtures["f2"]
    f3 = calculus_fixtures["f3"]

    # ||f||^2 should equal f.dot(f)
    assert abs(f2.norm() ** 2 - f2.dot(f2)) < 1e-10
    assert abs(f3.norm() ** 2 - f3.dot(f3)) < 1e-10
