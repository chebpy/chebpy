"""Unit-tests for miscellaneous Chebtech class usage.

This module contains tests for various aspects of the Chebtech class,
including emptiness, constantness, size, calling, prolongation, vscale, and copying.
"""

import numpy as np
import pytest

from chebpy.algorithms import standard_chop
from chebpy.chebtech import Chebtech

from ..generic.class_usage import test_constfun_value  # noqa: F401
from ..utilities import cos, eps, exp, pi, sin

# Ensure reproducibility
rng = np.random.default_rng(0)


@pytest.fixture
def chebtech_fixture():
    """Create Chebtech object and test points for testing.

    This fixture creates a Chebtech object representing sin(30*x) with 100 coefficients
    and an array of 100 random test points in the interval [-1, 1].

    Returns:
        dict: Dictionary containing:
            ff: Chebtech object for sin(30*x)
            xx: Array of random test points
    """
    ff = Chebtech.initfun_fixedlen(lambda x: np.sin(30 * x), 100)
    xx = -1 + 2 * rng.random(100)
    return {"ff": ff, "xx": xx}


# check the size() method is working properly
def test_size():
    """Test the size property of Chebtech objects.

    This test verifies that the size property correctly returns the number
    of coefficients in the Chebtech object for various cases:
    1. Empty Chebtech (should have size 0)
    2. Constant Chebtech (should have size 1)
    3. Chebtech with arbitrary coefficients (should match coefficient array size)
    """
    cfs = rng.random(10)
    assert Chebtech(np.array([])).size == 0
    assert Chebtech(np.array([1.0])).size == 1
    assert Chebtech(cfs).size == cfs.size


# test the different permutations of self(xx, ..)
def test_call(chebtech_fixture):
    """Test basic function evaluation of Chebtech objects.

    This test verifies that a Chebtech object can be called with an array of points
    to evaluate the function at those points.

    Args:
        chebtech_fixture: Fixture providing test Chebtech object and points.
    """
    chebtech_fixture["ff"](chebtech_fixture["xx"])


def test_call_bary(chebtech_fixture):
    """Test function evaluation using barycentric interpolation.

    This test verifies that a Chebtech object can be called with the 'bary' method
    to evaluate the function using barycentric interpolation, both with positional
    and keyword arguments.

    Args:
        chebtech_fixture: Fixture providing test Chebtech object and points.
    """
    chebtech_fixture["ff"](chebtech_fixture["xx"], "bary")
    chebtech_fixture["ff"](chebtech_fixture["xx"], how="bary")


def test_call_clenshaw(chebtech_fixture):
    """Test function evaluation using Clenshaw's algorithm.

    This test verifies that a Chebtech object can be called with the 'clenshaw' method
    to evaluate the function using Clenshaw's algorithm, both with positional
    and keyword arguments.

    Args:
        chebtech_fixture: Fixture providing test Chebtech object and points.
    """
    chebtech_fixture["ff"](chebtech_fixture["xx"], "clenshaw")
    chebtech_fixture["ff"](chebtech_fixture["xx"], how="clenshaw")


def test_call_bary_vs_clenshaw(chebtech_fixture):
    """Test that barycentric and Clenshaw evaluation methods give similar results.

    This test verifies that evaluating a Chebtech object using both the barycentric
    and Clenshaw methods gives results that are close to each other within a specified
    tolerance.

    Args:
        chebtech_fixture: Fixture providing test Chebtech object and points.
    """
    b = chebtech_fixture["ff"](chebtech_fixture["xx"], "clenshaw")
    c = chebtech_fixture["ff"](chebtech_fixture["xx"], "bary")
    assert np.max(np.abs(b - c)) <= 5e1 * eps


def test_call_raises(chebtech_fixture):
    """Test that invalid evaluation methods raise appropriate exceptions.

    This test verifies that attempting to evaluate a Chebtech object with an
    invalid method name raises a ValueError, both with positional and keyword arguments.

    Args:
        chebtech_fixture: Fixture providing test Chebtech object and points.
    """
    with pytest.raises(ValueError, match="notamethod"):
        chebtech_fixture["ff"](chebtech_fixture["xx"], "notamethod")
    with pytest.raises(ValueError, match="notamethod"):
        chebtech_fixture["ff"](chebtech_fixture["xx"], how="notamethod")


def test_prolong(chebtech_fixture):
    """Test the prolong method of Chebtech objects.

    This test verifies that the prolong method correctly changes the size of
    a Chebtech object to the specified size for various target sizes.

    Args:
        chebtech_fixture: Fixture providing test Chebtech object.
    """
    ff = chebtech_fixture["ff"]
    for k in [0, 1, 20, ff.size, 200]:
        assert ff.prolong(k).size == k


def test_copy(chebtech_fixture):
    """Test the copy method of Chebtech objects.

    This test verifies that the copy method creates a new Chebtech object
    that is equal to itself but not equal to the original, and that has
    the same coefficients as the original.

    Args:
        chebtech_fixture: Fixture providing test Chebtech object.
    """
    ff = chebtech_fixture["ff"]
    gg = ff.copy()
    assert ff == ff
    assert gg == gg
    assert ff != gg
    assert np.max(np.abs(ff.coeffs - gg.coeffs)) == 0


def test_simplify(chebtech_fixture):
    """Test the simplify method of Chebtech objects.

    This test verifies that the simplify method:
    1. Uses standard_chop to determine the new size
    2. Creates a new Chebtech object with the chopped coefficients
    3. Returns a copy of the coefficients, not a view

    Args:
        chebtech_fixture: Fixture providing test Chebtech object.
    """
    ff = chebtech_fixture["ff"]
    gg = ff.simplify()
    # check that simplify is calling standard_chop underneath
    assert gg.size == standard_chop(ff.coeffs)
    assert np.max(np.abs(ff.coeffs[: gg.size] - gg.coeffs)) == 0
    # check we are returned a copy of self's coeffcients by changing
    # one entry of gg
    fcfs = ff.coeffs
    gcfs = gg.coeffs
    assert (fcfs[: gg.size] - gcfs).sum() == 0
    gg.coeffs[0] = 1
    assert (fcfs[: gg.size] - gcfs).sum() != 0


# --------------------------------------
#          vscale estimates
# --------------------------------------
vscales = [
    # (function, number of points, vscale)
    (lambda x: sin(4 * pi * x), 40, 1),
    (lambda x: cos(x), 15, 1),
    (lambda x: cos(4 * pi * x), 39, 1),
    (lambda x: exp(cos(4 * pi * x)), 181, exp(1)),
    (lambda x: cos(3244 * x), 3389, 1),
    (lambda x: exp(x), 15, exp(1)),
    (lambda x: 1e10 * exp(x), 15, 1e10 * exp(1)),
    (lambda x: 0 * x + 1.0, 1, 1),
]


# Use pytest parametrization for vscale tests
@pytest.mark.parametrize(("fun", "n", "vscale"), vscales)
def test_vscale(fun, n, vscale):
    """Test vscale estimates for various functions.

    This test verifies that the vscale property of Chebtech objects
    correctly estimates the vertical scale of the function within
    a specified tolerance.

    Args:
        fun: Function to test
        n: Number of points to use
        vscale: Expected vscale value
    """
    ff = Chebtech.initfun_fixedlen(fun, n)
    absdiff = abs(ff.vscale - vscale)
    assert absdiff <= vscale
