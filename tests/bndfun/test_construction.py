"""Unit-tests for construction of Bndfun objects."""

import numpy as np
import pytest

from chebpy.core.bndfun import Bndfun
from chebpy.core.chebtech import Chebtech2
from chebpy.core.utilities import Interval

from ..utilities import cos, eps, exp, sin

# Ensure reproducibility
rng = np.random.default_rng(0)


def test_onefun_construction():
    """Test construction of Bndfun from a Chebtech object."""
    coeffs = rng.random(10)
    subinterval = Interval()
    onefun = Chebtech2(coeffs)
    f = Bndfun(onefun, subinterval)
    assert isinstance(f, Bndfun)
    assert np.max(f.coeffs - coeffs) < eps


def test_const_construction():
    """Test construction of a constant Bndfun."""
    subinterval = Interval()
    ff = Bndfun.initconst(1.0, subinterval)
    assert ff.size == 1
    assert ff.isconst
    assert not ff.isempty
    with pytest.raises(ValueError):
        Bndfun.initconst([1.0], subinterval)


def test_empty_construction():
    """Test construction of an empty Bndfun."""
    ff = Bndfun.initempty()
    assert ff.size == 0
    assert not ff.isconst
    assert ff.isempty
    with pytest.raises(TypeError):
        Bndfun.initempty([1.0])


def test_identity_construction():
    """Test construction of an identity Bndfun."""
    for a, b in [(-1, 1), (-10, -2), (-2.3, 1.24), (20, 2000)]:
        itvl = Interval(a, b)
        ff = Bndfun.initidentity(itvl)
        assert ff.size == 2
        xx = np.linspace(a, b, 1001)
        tol = eps * abs(itvl).max()
        assert np.max(ff(xx) - xx) <= tol


# Test functions for adaptive and fixed-length construction
fun_details = [
    # (function, name for the test printouts,
    #  interval, Matlab chebfun adaptive degree on [-2,3])
    (lambda x: x**3 + x**2 + x + 1, "poly3(x)", [-2, 3], 4),
    (lambda x: exp(x), "exp(x)", [-2, 3], 20),
    (lambda x: sin(x), "sin(x)", [-2, 3], 20),
    (lambda x: cos(20 * x), "cos(20x)", [-2, 3], 90),
    (lambda x: 0.0 * x + 1.0, "constfun", [-2, 3], 1),
    (lambda x: 0.0 * x, "zerofun", [-2, 3], 1),
]


@pytest.mark.parametrize("fun, name, interval, funlen", fun_details)
def test_adaptive(fun, name, interval, funlen):
    """Test adaptive construction of Bndfun.

    Args:
        fun: Function to test.
        name: Name of the function for test printouts.
        interval: Domain interval for the function.
        funlen: Expected function length (number of coefficients).
    """
    subinterval = Interval(*interval)
    ff = Bndfun.initfun_adaptive(fun, subinterval)
    assert ff.size in {funlen - 1, funlen}


@pytest.mark.parametrize("fun, name, interval, _", fun_details)
def test_fixedlen(fun, name, interval, _):
    """Test fixed-length construction of Bndfun.

    Args:
        fun: Function to test.
        name: Name of the function for test printouts.
        interval: Domain interval for the function.
        _: Unused parameter (kept for compatibility with fun_details).
    """
    subinterval = Interval(*interval)
    n = 100
    ff = Bndfun.initfun_fixedlen(fun, subinterval, n)
    assert ff.size == n
