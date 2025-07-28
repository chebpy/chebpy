"""Unit-tests for Chebtech2 calculus operations.

This module contains tests for the calculus operations of Chebtech2,
including sum, cumsum, and diff methods.
"""

import pytest
import numpy as np

from chebpy.core.chebtech import Chebtech2
from .conftest import pi, sin, cos, exp, eps


# tests for the correct results in the empty cases
def test_sum_empty(emptyfun):
    """Test sum method on an empty Chebtech2 object.

    This test verifies that the sum method of an empty Chebtech2 object
    returns 0.

    Args:
        emptyfun: Fixture providing an empty Chebtech2 object
    """
    assert emptyfun.sum() == 0


def test_cumsum_empty(emptyfun):
    """Test cumsum method on an empty Chebtech2 object.

    This test verifies that the cumsum method of an empty Chebtech2 object
    returns an empty Chebtech2 object.

    Args:
        emptyfun: Fixture providing an empty Chebtech2 object
    """
    assert emptyfun.cumsum().isempty


def test_diff_empty(emptyfun):
    """Test diff method on an empty Chebtech2 object.

    This test verifies that the diff method of an empty Chebtech2 object
    returns an empty Chebtech2 object.

    Args:
        emptyfun: Fixture providing an empty Chebtech2 object
    """
    assert emptyfun.diff().isempty


# --------------------------------------
#           definite integrals
# --------------------------------------
def_integrals = [
    # (function, number of points, integral, tolerance)
    (lambda x: sin(x), 14, 0.0, eps),
    (lambda x: sin(4 * pi * x), 40, 0.0, 1e1 * eps),
    (lambda x: cos(x), 15, 1.682941969615793, 2 * eps),
    (lambda x: cos(4 * pi * x), 39, 0.0, 2 * eps),
    (lambda x: exp(cos(4 * pi * x)), 182, 2.532131755504016, 4 * eps),
    (lambda x: cos(3244 * x), 3389, 5.879599674161602e-04, 5e2 * eps),
    (lambda x: exp(x), 15, exp(1) - exp(-1), 2 * eps),
    (lambda x: 1e10 * exp(x), 15, 1e10 * (exp(1) - exp(-1)), 4e10 * eps),
    (lambda x: 0 * x + 1.0, 1, 2, eps),
]


# Ensure reproducibility
np.random.seed(0)


@pytest.mark.parametrize("fun, n, integral, tol", def_integrals)
def test_definite_integral(fun, n, integral, tol):
    """Test definite integration (sum method) of Chebtech2 objects.

    This test verifies that the sum method of a Chebtech2 object correctly
    computes the definite integral of various functions within the specified
    tolerance.

    Args:
        fun: Function to integrate
        n: Number of points to use
        integral: Expected value of the integral
        tol: Tolerance for the comparison
    """
    ff = Chebtech2.initfun_fixedlen(fun, n)
    absdiff = abs(ff.sum() - integral)
    assert absdiff <= tol


# --------------------------------------
#         indefinite integrals
# --------------------------------------
indef_integrals = [
    # (function, derivative, number of points, tolerance)
    (lambda x: sin(x), lambda x: -cos(x), 15, 2 * eps),
    (lambda x: cos(x), lambda x: sin(x), 15, 2 * eps),
    (lambda x: exp(x), lambda x: exp(x), 15, 5 * eps),
    (lambda x: x ** 3, lambda x: 0.25 * x ** 4, 16, 10 * eps),
    (lambda x: 0 * x + 1, lambda x: x, 1, 3.0),
    (lambda x: 0 * x, lambda x: 0 * x, 1, eps),
]


@pytest.mark.parametrize("fun, dfn, n, tol", indef_integrals)
def test_indefinite_integral(fun, dfn, n, tol):
    """Test indefinite integration (cumsum method) of Chebtech2 objects.

    This test verifies that the cumsum method of a Chebtech2 object correctly
    computes the indefinite integral of various functions within the specified
    tolerance. It checks that the indefinite integral of the function matches
    the expected antiderivative (up to a constant).

    Args:
        fun: Function to integrate
        dfn: Expected antiderivative function
        n: Number of points to use
        tol: Tolerance for the comparison
    """
    ff = Chebtech2.initfun_fixedlen(fun, n)
    gg = Chebtech2.initfun_fixedlen(dfn, n)
    xx = np.linspace(-1, 1, 1000)
    absdiff = np.max(ff.cumsum()(xx) - (gg(xx) - gg(-1)))
    assert absdiff <= tol


# --------------------------------------
#            derivatives
# --------------------------------------
derivatives = [
    # (function, derivative, number of points, tolerance)
    (lambda x: sin(x), lambda x: cos(x), 15, 30 * eps),
    (lambda x: cos(x), lambda x: -sin(x), 15, 30 * eps),
    (lambda x: exp(x), lambda x: exp(x), 15, 200 * eps),
    (lambda x: x ** 3, lambda x: 3 * x ** 2, 16, 30 * eps),
    (lambda x: 0 * x + 1, lambda x: 0 * x, 1, eps),
    (lambda x: 0 * x, lambda x: 0 * x, 1, eps),
]


@pytest.mark.parametrize("fun, der, n, tol", derivatives)
def test_derivative(fun, der, n, tol):
    """Test differentiation (diff method) of Chebtech2 objects.

    This test verifies that the diff method of a Chebtech2 object correctly
    computes the derivative of various functions within the specified
    tolerance. It checks that the derivative of the function matches
    the expected derivative function.

    Args:
        fun: Function to differentiate
        der: Expected derivative function
        n: Number of points to use
        tol: Tolerance for the comparison
    """
    ff = Chebtech2.initfun_fixedlen(fun, n)
    gg = Chebtech2.initfun_fixedlen(der, n)
    xx = np.linspace(-1, 1, 1000)
    absdiff = np.max(ff.diff()(xx) - gg(xx))
    assert absdiff <= 10*tol
