"""Unit-tests for Chebfun ufunc operations.

This module contains tests for the ufunc operations of Chebfun,
including absolute, trigonometric, exponential, and logarithmic functions,
as well as piecewise-smooth functions like sign, ceil, and floor.
"""

import numpy as np

from chebpy.chebfun import Chebfun
from chebpy.utilities import Interval

from ..generic.ufuncs import test_emptycase, ufunc_parameter  # noqa: F401
from ..utilities import eps


def test_abs_absolute_alias():
    """Test that abs and absolute are aliases.

    This test verifies that the abs and absolute methods of Chebfun
    are aliases for the same function.
    """
    assert Chebfun.abs == Chebfun.absolute


# Generate test functions for ufuncs
def test_ufuncs():
    """Test ufunc operations on Chebfun objects.

    This test verifies that applying a ufunc to a Chebfun object
    produces the expected result within a specified tolerance.
    """
    yy = np.linspace(-1, 1, 2000)

    for ufunc, f, interval in ufunc_parameter():
        interval = Interval(*interval)
        a, b = interval
        ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))

        def gg(x, ufunc=ufunc, f=f):
            return ufunc(f(x))

        gg_result = getattr(ff, ufunc.__name__)()

        xx = interval(yy)
        vscl = gg_result.vscale
        lscl = sum([fun.size for fun in gg_result])
        assert np.max(np.abs(gg(xx) - gg_result(xx))) <= vscl * lscl * eps


def test_sign_splitting():
    """Test that sign() splits at roots and produces correct piecewise result."""
    f = Chebfun.initfun_adaptive(lambda x: x, [-1, 1])
    g = f.sign()
    xx = np.linspace(-1, 1, 1000)
    assert np.max(np.abs(g(xx) - np.sign(xx))) < eps
    # Verify sign is 0 at the root
    assert g(0.0) == 0.0


def test_sign_no_root():
    """Test sign() on a function with no roots (all positive)."""
    f = Chebfun.initfun_adaptive(lambda x: x + 2, [-1, 1])
    g = f.sign()
    xx = np.linspace(-1, 1, 100)
    assert np.allclose(g(xx), 1.0)


def test_sign_multiple_roots():
    """Test sign() on a function with multiple roots."""
    f = Chebfun.initfun_adaptive(lambda x: np.sin(2 * np.pi * x), [-1, 1])
    g = f.sign()
    # Check at points away from roots
    assert g(0.125) == 1.0
    assert g(-0.125) == -1.0
    assert g(0.625) == -1.0
    assert g(-0.625) == 1.0


def test_ceil_splitting():
    """Test that ceil() splits at integer crossings."""
    f = Chebfun.initfun_adaptive(lambda x: 3 * x, [-1, 1])
    g = f.ceil()
    xx = np.linspace(-1, 1, 1000)
    # Avoid points near breakpoints where floating point rounding disagrees
    bps = g.breakpoints
    mask = np.ones_like(xx, dtype=bool)
    for bp in bps:
        mask &= np.abs(xx - bp) > 1e-10
    xx_safe = xx[mask]
    assert np.max(np.abs(g(xx_safe) - np.ceil(3 * xx_safe))) < eps


def test_ceil_no_crossing():
    """Test ceil() on a function that stays between two integers."""
    f = Chebfun.initfun_adaptive(lambda x: 0.1 * x + 0.5, [-1, 1])
    g = f.ceil()
    xx = np.linspace(-1, 1, 100)
    assert np.allclose(g(xx), 1.0)


def test_floor_splitting():
    """Test that floor() splits at integer crossings."""
    f = Chebfun.initfun_adaptive(lambda x: 3 * x, [-1, 1])
    g = f.floor()
    xx = np.linspace(-1, 1, 1000)
    # Avoid points near breakpoints where floating point rounding disagrees
    bps = g.breakpoints
    mask = np.ones_like(xx, dtype=bool)
    for bp in bps:
        mask &= np.abs(xx - bp) > 1e-10
    xx_safe = xx[mask]
    assert np.max(np.abs(g(xx_safe) - np.floor(3 * xx_safe))) < eps


def test_floor_no_crossing():
    """Test floor() on a function that stays between two integers."""
    f = Chebfun.initfun_adaptive(lambda x: 0.1 * x + 0.5, [-1, 1])
    g = f.floor()
    xx = np.linspace(-1, 1, 100)
    assert np.allclose(g(xx), 0.0)
