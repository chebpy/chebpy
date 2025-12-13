"""Unit-tests for chebpy/core/algorithms.py.

This module contains tests for the algorithm functions in the chebpy library,
including barycentric interpolation (bary), Clenshaw evaluation (clenshaw),
and coefficient multiplication (coeffmult).

The tests verify that these algorithms handle various input types correctly,
including empty arrays, scalar inputs, and arrays of different sizes. They also
check that the algorithms produce results within expected tolerance of the
true values.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from chebpy.algorithms import bary, clenshaw, coeffmult
from chebpy.chebtech import Chebtech

from .utilities import cos, eps, exp, scaled_tol

rng = np.random.default_rng(0)  # Use a fixed seed for reproducibility

# Turn off 'divide' and 'invalid' RuntimeWarnings
# These warnings are expected and required for the barycentric formula,
# which involves division operations that can produce inf/NaN values
# that are later handled correctly by the algorithm
np.seterr(divide="ignore", invalid="ignore")


@pytest.fixture
def evaluation_fixtures() -> dict[str, Any]:
    """Create fixtures for testing evaluation algorithms.

    Returns:
    -------
    dict[str, Any]
        Dictionary containing test fixtures for evaluation
    """
    npts = 15
    xk = Chebtech._chebpts(npts)
    vk = Chebtech._barywts(npts)
    fk = rng.random(npts)
    ak = rng.random(11)
    xx = -1 + 2 * rng.random(9)
    pts = -1 + 2 * rng.random(1001)
    return {"xk": xk, "vk": vk, "fk": fk, "ak": ak, "xx": xx, "pts": pts}


# check an empty array is returned whenever either or both of the first
# two arguments are themselves empty arrays
def test_bary_empty(evaluation_fixtures: dict[str, Any]) -> None:
    """Test bary function with empty arrays.

    Parameters
    ----------
    evaluation_fixtures : dict[str, Any]
        Dictionary containing test fixtures
    """
    pts = evaluation_fixtures["pts"]
    null = (None, None)
    assert bary(np.array([]), np.array([]), *null).size == 0
    assert bary(np.array([0.1]), np.array([]), *null).size == 0
    assert bary(np.array([]), np.array([0.1]), *null).size == 0
    assert bary(pts, np.array([]), *null).size == 0
    assert bary(np.array([]), pts, *null).size == 0
    assert bary(np.array([0.1]), np.array([0.1]), *null).size != 0


def test_clenshaw_empty(evaluation_fixtures: dict[str, Any]) -> None:
    """Test clenshaw function with empty arrays.

    Parameters
    ----------
    evaluation_fixtures : dict[str, Any]
        Dictionary containing test fixtures
    """
    pts = evaluation_fixtures["pts"]
    assert clenshaw(np.array([]), np.array([])).size == 0
    assert clenshaw(np.array([]), np.array([1.0])).size == 0
    assert clenshaw(np.array([1.0]), np.array([])).size == 0
    assert clenshaw(pts, np.array([])).size == 0
    assert clenshaw(np.array([]), pts).size == 0
    assert clenshaw(np.array([0.1]), np.array([0.1])).size != 0


# check that scalars get evaluated to scalars (not arrays)
def test_clenshaw_scalar_input(evaluation_fixtures: dict[str, Any]) -> None:
    """Test clenshaw function with scalar inputs.

    Parameters
    ----------
    evaluation_fixtures : dict[str, Any]
        Dictionary containing test fixtures
    """
    xx = evaluation_fixtures["xx"]
    ak = evaluation_fixtures["ak"]
    for x in xx:
        assert np.isscalar(clenshaw(x, ak))
    assert not np.isscalar(clenshaw(xx, ak))


def test_bary_scalar_input(evaluation_fixtures: dict[str, Any]) -> None:
    """Test bary function with scalar inputs.

    Parameters
    ----------
    evaluation_fixtures : dict[str, Any]
        Dictionary containing test fixtures
    """
    xx = evaluation_fixtures["xx"]
    fk = evaluation_fixtures["fk"]
    xk = evaluation_fixtures["xk"]
    vk = evaluation_fixtures["vk"]
    for x in xx:
        assert np.isscalar(bary(x, fk, xk, vk))
    assert not np.isscalar(bary(xx, fk, xk, vk))


# Check that we always get float output for constant Chebtechs, even
# when passing in an integer input.
# TODO: Move these tests elsewhere?
def test_bary_float_output() -> None:
    """Test that bary evaluation returns float output for constant Chebtechs."""
    ff = Chebtech.initconst(1)
    gg = Chebtech.initconst(1.0)
    assert isinstance(ff(0, "bary"), float)
    assert isinstance(gg(0, "bary"), float)


def test_clenshaw_float_output() -> None:
    """Test that clenshaw evaluation returns float output for constant Chebtechs."""
    ff = Chebtech.initconst(1)
    gg = Chebtech.initconst(1.0)
    assert isinstance(ff(0, "clenshaw"), float)
    assert isinstance(gg(0, "clenshaw"), float)


# Check that we get consistent output from bary and clenshaw
# TODO: Move these tests elsewhere?
def test_bary_clenshaw_consistency() -> None:
    """Test that bary and clenshaw return consistent output types."""
    coeffs = rng.random(3)
    evalpts = (0.5, np.array([]), np.array([0.5]), np.array([0.5, 0.6]))
    for n in range(len(coeffs)):
        ff = Chebtech(coeffs[:n])
        for xx in evalpts:
            fb = ff(xx, "bary")
            fc = ff(xx, "clenshaw")
            assert isinstance(fb, type(fc))


# Define evaluation points of increasing density for testing algorithm accuracy
evalpts = [np.linspace(-1, 1, int(n)) for n in np.array([1e2, 1e3, 1e4, 1e5])]

# Define arrays of Chebyshev points for interpolation
ptsarry = [Chebtech._chebpts(n) for n in np.array([100, 200])]

# List of evaluation methods to test
methods = [bary, clenshaw]


def _eval_tester(method: Callable, fun: Callable, evalpts: np.ndarray, chebpts: np.ndarray) -> bool:
    """Create a test function for evaluating methods.

    Parameters
    ----------
    method : Callable
        The method to test (bary or clenshaw)
    fun : Callable
        The function to evaluate
    evalpts : np.ndarray
        Points to evaluate the function at
    chebpts : np.ndarray
        Chebyshev points

    Returns:
    -------
    bool
        True if the method's output is within tolerance of the expected result
    """
    x = evalpts
    xk = chebpts
    fvals = fun(xk)

    if method is bary:
        vk = Chebtech._barywts(fvals.size)
        a = bary(x, fvals, xk, vk)
        tol_multiplier = 1e0

    elif method is clenshaw:
        ak = Chebtech._vals2coeffs(fvals)
        a = clenshaw(x, ak)
        tol_multiplier = 2e1

    b = fun(evalpts)
    n = evalpts.size
    tol = tol_multiplier * scaled_tol(n)

    return np.max(np.abs(a - b)) < tol  # inf_norm_less_than_tol(a, b, tol)


def test_bary(testfunctions: list) -> None:
    """Test barycentric interpolation algorithm.

    This test verifies that the barycentric interpolation algorithm (bary)
    correctly evaluates various test functions at different sets of points.
    It checks that the interpolated values are within tolerance of the true values.

    Parameters
    ----------
    testfunctions : list
        List of test functions to evaluate
    """
    for fun, _, _ in testfunctions:
        for j, chebpts in enumerate(ptsarry):
            for k, xx in enumerate(evalpts):
                assert _eval_tester(bary, fun, xx, chebpts)


def test_clenshaw(testfunctions: list) -> None:
    """Test Clenshaw evaluation algorithm.

    This test verifies that the Clenshaw evaluation algorithm (clenshaw)
    correctly evaluates various test functions at different sets of points.
    It checks that the evaluated values are within tolerance of the true values.

    Parameters
    ----------
    testfunctions : list
        List of test functions to evaluate
    """
    for fun, _, _ in testfunctions:
        for j, chebpts in enumerate(ptsarry):
            for k, xx in enumerate(evalpts):
                assert _eval_tester(clenshaw, fun, xx, chebpts)


@pytest.fixture
def coeffmult_fixtures() -> dict[str, Any]:
    """Create fixtures for testing coefficient multiplication.

    Returns:
    -------
    dict[str, Any]
        Dictionary containing test fixtures for coefficient multiplication
    """

    def f(x):
        return exp(x)

    def g(x):
        return cos(x)

    fn = 15
    gn = 15
    return {"f": f, "g": g, "fn": fn, "gn": gn}


def test_coeffmult(coeffmult_fixtures: dict[str, Any]) -> None:
    """Test coefficient multiplication.

    Parameters
    ----------
    coeffmult_fixtures : dict[str, Any]
        Dictionary containing test fixtures
    """
    f = coeffmult_fixtures["f"]
    g = coeffmult_fixtures["g"]
    fn = coeffmult_fixtures["fn"]
    gn = coeffmult_fixtures["gn"]

    def h(x):
        return f(x) * g(x)

    hn = fn + gn - 1
    fc = Chebtech.initfun(f, fn).prolong(hn).coeffs
    gc = Chebtech.initfun(g, gn).prolong(hn).coeffs
    hc = coeffmult(fc, gc)
    hx = Chebtech.initfun(h, hn).coeffs
    assert np.max(np.abs(hc - hx)) <= 2e1 * eps
