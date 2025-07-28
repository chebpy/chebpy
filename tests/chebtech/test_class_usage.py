"""Unit-tests for miscellaneous Chebtech2 class usage.

This module contains tests for various aspects of the Chebtech2 class,
including emptiness, constantness, size, calling, prolongation, vscale, and copying.
"""

import pytest
import numpy as np

from chebpy.core.chebtech import Chebtech2
from chebpy.core.algorithms import standard_chop
from .conftest import pi, sin, cos, exp, eps



@pytest.fixture
def chebtech_fixture():
    """Create Chebtech2 object and test points for testing.

    This fixture creates a Chebtech2 object representing sin(30*x) with 100 coefficients
    and an array of 100 random test points in the interval [-1, 1].

    Returns:
        dict: Dictionary containing:
            ff: Chebtech2 object for sin(30*x)
            xx: Array of random test points
    """
    np.random.seed(0)  # Ensure reproducibility
    ff = Chebtech2.initfun_fixedlen(lambda x: np.sin(30 * x), 100)
    xx = -1 + 2 * np.random.rand(100)
    return {"ff": ff, "xx": xx}


# tests for emptiness of Chebtech2 objects
def test_isempty_True():
    """Test that empty Chebtech2 objects report isempty=True.

    This test verifies that a Chebtech2 object created with an empty array
    correctly reports that it is empty via the isempty property.
    """
    f = Chebtech2(np.array([]))
    assert f.isempty
    assert not (not f.isempty)


def test_isempty_False():
    """Test that non-empty Chebtech2 objects report isempty=False.

    This test verifies that a Chebtech2 object created with a non-empty array
    correctly reports that it is not empty via the isempty property.
    """
    f = Chebtech2(np.array([1.0]))
    assert not f.isempty
    assert not f.isempty is True


# tests for constantness of Chebtech2 objects
def test_isconst_True():
    """Test that constant Chebtech2 objects report isconst=True.

    This test verifies that a Chebtech2 object representing a constant function
    correctly reports that it is constant via the isconst property.
    """
    f = Chebtech2(np.array([1.0]))
    assert f.isconst
    assert not (not f.isconst)


def test_isconst_False():
    """Test that non-constant Chebtech2 objects report isconst=False.

    This test verifies that a Chebtech2 object that doesn't represent a constant function
    (in this case, an empty object) correctly reports that it is not constant.
    """
    f = Chebtech2(np.array([]))
    assert not f.isconst
    assert not f.isconst is True


# check the size() method is working properly
def test_size():
    """Test the size property of Chebtech2 objects.

    This test verifies that the size property correctly returns the number
    of coefficients in the Chebtech2 object for various cases:
    1. Empty Chebtech2 (should have size 0)
    2. Constant Chebtech2 (should have size 1)
    3. Chebtech2 with arbitrary coefficients (should match coefficient array size)
    """
    cfs = np.random.rand(10)
    assert Chebtech2(np.array([])).size == 0
    assert Chebtech2(np.array([1.0])).size == 1
    assert Chebtech2(cfs).size == cfs.size


# test the different permutations of self(xx, ..)
def test_call(chebtech_fixture):
    """Test basic function evaluation of Chebtech2 objects.

    This test verifies that a Chebtech2 object can be called with an array of points
    to evaluate the function at those points.

    Args:
        chebtech_fixture: Fixture providing test Chebtech2 object and points.
    """
    chebtech_fixture["ff"](chebtech_fixture["xx"])


def test_call_bary(chebtech_fixture):
    """Test function evaluation using barycentric interpolation.

    This test verifies that a Chebtech2 object can be called with the 'bary' method
    to evaluate the function using barycentric interpolation, both with positional
    and keyword arguments.

    Args:
        chebtech_fixture: Fixture providing test Chebtech2 object and points.
    """
    chebtech_fixture["ff"](chebtech_fixture["xx"], "bary")
    chebtech_fixture["ff"](chebtech_fixture["xx"], how="bary")


def test_call_clenshaw(chebtech_fixture):
    """Test function evaluation using Clenshaw's algorithm.

    This test verifies that a Chebtech2 object can be called with the 'clenshaw' method
    to evaluate the function using Clenshaw's algorithm, both with positional
    and keyword arguments.

    Args:
        chebtech_fixture: Fixture providing test Chebtech2 object and points.
    """
    chebtech_fixture["ff"](chebtech_fixture["xx"], "clenshaw")
    chebtech_fixture["ff"](chebtech_fixture["xx"], how="clenshaw")


def test_call_bary_vs_clenshaw(chebtech_fixture):
    """Test that barycentric and Clenshaw evaluation methods give similar results.

    This test verifies that evaluating a Chebtech2 object using both the barycentric
    and Clenshaw methods gives results that are close to each other within a specified
    tolerance.

    Args:
        chebtech_fixture: Fixture providing test Chebtech2 object and points.
    """
    b = chebtech_fixture["ff"](chebtech_fixture["xx"], "clenshaw")
    c = chebtech_fixture["ff"](chebtech_fixture["xx"], "bary")
    assert np.max(b - c) <= 5e1 * eps


def test_call_raises(chebtech_fixture):
    """Test that invalid evaluation methods raise appropriate exceptions.

    This test verifies that attempting to evaluate a Chebtech2 object with an
    invalid method name raises a ValueError, both with positional and keyword arguments.

    Args:
        chebtech_fixture: Fixture providing test Chebtech2 object and points.
    """
    with pytest.raises(ValueError):
        chebtech_fixture["ff"](chebtech_fixture["xx"], "notamethod")
    with pytest.raises(ValueError):
        chebtech_fixture["ff"](chebtech_fixture["xx"], how="notamethod")


def test_prolong(chebtech_fixture):
    """Test the prolong method of Chebtech2 objects.

    This test verifies that the prolong method correctly changes the size of
    a Chebtech2 object to the specified size for various target sizes.

    Args:
        chebtech_fixture: Fixture providing test Chebtech2 object.
    """
    ff = chebtech_fixture["ff"]
    for k in [0, 1, 20, ff.size, 200]:
        assert ff.prolong(k).size == k


def test_vscale_empty():
    """Test the vscale property of empty Chebtech2 objects.

    This test verifies that an empty Chebtech2 object has a vscale of 0.0.
    """
    gg = Chebtech2(np.array([]))
    assert gg.vscale == 0.0


def test_copy(chebtech_fixture):
    """Test the copy method of Chebtech2 objects.

    This test verifies that the copy method creates a new Chebtech2 object
    that is equal to itself but not equal to the original, and that has
    the same coefficients as the original.

    Args:
        chebtech_fixture: Fixture providing test Chebtech2 object.
    """
    ff = chebtech_fixture["ff"]
    gg = ff.copy()
    assert ff == ff
    assert gg == gg
    assert ff != gg
    assert np.max(ff.coeffs - gg.coeffs) == 0


def test_simplify(chebtech_fixture):
    """Test the simplify method of Chebtech2 objects.

    This test verifies that the simplify method:
    1. Uses standard_chop to determine the new size
    2. Creates a new Chebtech2 object with the chopped coefficients
    3. Returns a copy of the coefficients, not a view

    Args:
        chebtech_fixture: Fixture providing test Chebtech2 object.
    """
    ff = chebtech_fixture["ff"]
    gg = ff.simplify()
    # check that simplify is calling standard_chop underneath
    assert gg.size == standard_chop(ff.coeffs)
    assert np.max(ff.coeffs[: gg.size] - gg.coeffs) == 0
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
@pytest.mark.parametrize("fun, n, vscale", vscales)
def test_vscale(fun, n, vscale):
    """Test vscale estimates for various functions.

    This test verifies that the vscale property of Chebtech2 objects
    correctly estimates the vertical scale of the function within
    a specified tolerance.

    Args:
        fun: Function to test
        n: Number of points to use
        vscale: Expected vscale value
    """
    np.random.seed(0)  # Ensure reproducibility
    ff = Chebtech2.initfun_fixedlen(fun, n)
    absdiff = abs(ff.vscale - vscale)
    assert absdiff <= 0.1 * vscale
