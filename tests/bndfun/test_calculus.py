"""Unit-tests for Bndfun calculus operations."""

import numpy as np
import pytest

from chebpy.core.bndfun import Bndfun
from chebpy.core.chebtech import Chebtech2
from chebpy.core.utilities import Interval

from ..utilities import cos, eps, exp, pi, sin


@pytest.fixture
def calculus_fixtures():
    """Create fixtures for testing Bndfun calculus operations."""
    emptyfun = Bndfun(Chebtech2.initempty(), Interval())
    yy = np.linspace(-1, 1, 2000)

    return {"emptyfun": emptyfun, "yy": yy}


def test_sum_empty(calculus_fixtures):
    """Test the sum method on an empty Bndfun."""
    emptyfun = calculus_fixtures["emptyfun"]
    assert emptyfun.sum() == 0


def test_cumsum_empty(calculus_fixtures):
    """Test the cumsum method on an empty Bndfun."""
    emptyfun = calculus_fixtures["emptyfun"]
    assert emptyfun.cumsum().isempty


def test_diff_empty(calculus_fixtures):
    """Test the diff method on an empty Bndfun."""
    emptyfun = calculus_fixtures["emptyfun"]
    assert emptyfun.diff().isempty


# --------------------------------------
#           definite integrals
# --------------------------------------
def_integrals = [
    # Use function, interval, integral, tolerance
    (lambda x: sin(x), [-2, 2], 0.0, 2 * eps),
    (lambda x: sin(4 * pi * x), [-0.1, 0.7], 0.088970317927147, 1e1 * eps),
    (lambda x: cos(x), [-100, 203], 0.426944059057085, 5e2 * eps),
    (lambda x: cos(4 * pi * x), [-1e-1, -1e-3], 0.074682699182803, 2 * eps),
    (lambda x: exp(cos(4 * pi * x)), [-3, 1], 5.064263511008033, 4 * eps),
    (lambda x: cos(3244 * x), [0, 0.4], -3.758628487169980e-05, 5e2 * eps),
    (lambda x: exp(x), [-2, -1], exp(-1) - exp(-2), 2 * eps),
    (lambda x: 1e10 * exp(x), [-1, 2], 1e10 * (exp(2) - exp(-1)), 2e10 * eps),
    (lambda x: 0 * x + 1.0, [-100, 300], 400, eps),
]


@pytest.mark.parametrize("fun, interval, integral, tol", def_integrals)
def test_sum(fun, interval, integral, tol):
    """Test the sum method of Bndfun."""
    subinterval = Interval(*interval)
    ff = Bndfun.initfun_adaptive(fun, subinterval)
    absdiff = abs(ff.sum() - integral)
    assert absdiff <= tol


# --------------------------------------
#          indefinite integrals
# --------------------------------------
indef_integrals = [
    # (function, indefinite integral, interval, tolerance)
    (lambda x: 0 * x + 1.0, lambda x: x, [-2, 3], eps),
    (lambda x: x, lambda x: 1 / 2 * x**2, [-5, 0], 4 * eps),
    (lambda x: x**2, lambda x: 1 / 3 * x**3, [1, 10], 2e2 * eps),
    (lambda x: x**3, lambda x: 1 / 4 * x**4, [-1e-2, 4e-1], 2 * eps),
    (lambda x: x**4, lambda x: 1 / 5 * x**5, [-3, -2], 3e2 * eps),
    (lambda x: x**5, lambda x: 1 / 6 * x**6, [-1e-10, 1], 4 * eps),
    (lambda x: sin(x), lambda x: -cos(x), [-10, 22], 3e1 * eps),
    (lambda x: cos(3 * x), lambda x: 1.0 / 3 * sin(3 * x), [-3, 4], 2 * eps),
    (lambda x: exp(x), lambda x: exp(x), [-60, 1], 1e1 * eps),
    (lambda x: 1e10 * exp(x), lambda x: 1e10 * exp(x), [-1, 1], 1e10 * (3 * eps)),
]


@pytest.mark.parametrize("fun, ifn, interval, tol", indef_integrals)
def test_cumsum(fun, ifn, interval, tol):
    """Test the cumsum method of Bndfun."""
    subinterval = Interval(*interval)
    ff = Bndfun.initfun_adaptive(fun, subinterval)
    gg = Bndfun.initfun_fixedlen(ifn, subinterval, ff.size + 1)
    coeffs = gg.coeffs
    coeffs[0] = coeffs[0] - ifn(np.array([interval[0]]))[0]

    absdiff = np.max(ff.cumsum().coeffs - coeffs)
    assert absdiff <= tol


# --------------------------------------
#            derivatives
# --------------------------------------
derivatives = [
    #     (function, derivative, number of points, tolerance)
    (lambda x: 0 * x + 1.0, lambda x: 0 * x + 0, [-2, 3], eps),
    (lambda x: x, lambda x: 0 * x + 1, [-5, 0], 2e1 * eps),
    (lambda x: x**2, lambda x: 2 * x, [1, 10], 2e2 * eps),
    (lambda x: x**3, lambda x: 3 * x**2, [-1e-2, 4e-1], 3 * eps),
    (lambda x: x**4, lambda x: 4 * x**3, [-3, -2], 1e3 * eps),
    (lambda x: x**5, lambda x: 5 * x**4, [-1e-10, 1], 4e1 * eps),
    (lambda x: sin(x), lambda x: cos(x), [-10, 22], 5e2 * eps),
    (lambda x: cos(3 * x), lambda x: -3 * sin(3 * x), [-3, 4], 5e2 * eps),
    (lambda x: exp(x), lambda x: exp(x), [-60, 1], 2e2 * eps),
    (lambda x: 1e10 * exp(x), lambda x: 1e10 * exp(x), [-1, 1], 1e10 * 2e2 * eps),
]


@pytest.mark.parametrize("fun, der, interval, tol", derivatives)
def test_diff(fun, der, interval, tol):
    """Test the diff method of Bndfun."""
    subinterval = Interval(*interval)
    ff = Bndfun.initfun_adaptive(fun, subinterval)
    gg = Bndfun.initfun_fixedlen(der, subinterval, max(ff.size - 1, 1))

    absdiff = np.max(ff.diff().coeffs - gg.coeffs)
    assert absdiff <= tol
