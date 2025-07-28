"""Unit-tests for pyfun/utilities.py"""

import pytest
from typing import Any, Callable, List, Tuple, Union

import numpy as np

from chebpy.core.settings import DefaultPreferences
from chebpy.core.chebtech import Chebtech2
from chebpy.core.algorithms import bary, clenshaw, coeffmult

from .utilities import testfunctions, scaled_tol, inf_norm_less_than_tol, infnorm

# aliases
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
eps = DefaultPreferences.eps

np.random.seed(0)

# turn off 'divide' and 'invalid' Runtimewarnings: these are invoked in the
# barycentric formula and the warned-of behaviour is actually required
np.seterr(divide="ignore", invalid="ignore")


@pytest.fixture
def evaluation_fixtures() -> dict[str, Any]:
    """Create fixtures for testing evaluation algorithms.

    Returns
    -------
    dict[str, Any]
        Dictionary containing test fixtures for evaluation
    """
    npts = 15
    xk = Chebtech2._chebpts(npts)
    vk = Chebtech2._barywts(npts)
    fk = np.random.rand(npts)
    ak = np.random.rand(11)
    xx = -1 + 2 * np.random.rand(9)
    pts = -1 + 2 * np.random.rand(1001)
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
    ff = Chebtech2.initconst(1)
    gg = Chebtech2.initconst(1.0)
    assert isinstance(ff(0, "bary"), float)
    assert isinstance(gg(0, "bary"), float)


def test_clenshaw_float_output() -> None:
    """Test that clenshaw evaluation returns float output for constant Chebtechs."""
    ff = Chebtech2.initconst(1)
    gg = Chebtech2.initconst(1.0)
    assert isinstance(ff(0, "clenshaw"), float)
    assert isinstance(gg(0, "clenshaw"), float)


# Check that we get consistent output from bary and clenshaw
# TODO: Move these tests elsewhere?
def test_bary_clenshaw_consistency() -> None:
    """Test that bary and clenshaw return consistent output types."""
    coeffs = np.random.rand(3)
    evalpts = (0.5, np.array([]), np.array([0.5]), np.array([0.5, 0.6]))
    for n in range(len(coeffs)):
        ff = Chebtech2(coeffs[:n])
        for xx in evalpts:
            fb = ff(xx, "bary")
            fc = ff(xx, "clenshaw")
            assert type(fb) == type(fc)


evalpts = [np.linspace(-1, 1, int(n)) for n in np.array([1e2, 1e3, 1e4, 1e5])]
ptsarry = [Chebtech2._chebpts(n) for n in np.array([100, 200])]
methods = [bary, clenshaw]


def evalTester(method: Callable, fun: Callable, evalpts: np.ndarray, chebpts: np.ndarray) -> Callable:
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

    Returns
    -------
    Callable
        Test function
    """
    x = evalpts
    xk = chebpts
    fvals = fun(xk)

    if method is bary:
        vk = Chebtech2._barywts(fvals.size)
        a = bary(x, fvals, xk, vk)
        tol_multiplier = 1e0

    elif method is clenshaw:
        ak = Chebtech2._vals2coeffs(fvals)
        a = clenshaw(x, ak)
        tol_multiplier = 2e1

    b = fun(evalpts)
    n = evalpts.size
    tol = tol_multiplier * scaled_tol(n)

    return inf_norm_less_than_tol(a, b, tol)


# Dynamically create test functions for each method, function, and evaluation points
for method in methods:
    for fun, _, _ in testfunctions:
        for j, chebpts in enumerate(ptsarry):
            for k, xx in enumerate(evalpts):
                test_func = evalTester(method, fun, xx, chebpts)
                test_name = f"test_{method.__name__}_{fun.__name__}_{j:02}_{k:02}"
                # Add the test function to the global namespace
                globals()[test_name] = test_func


@pytest.fixture
def coeffmult_fixtures() -> dict[str, Any]:
    """Create fixtures for testing coefficient multiplication.

    Returns
    -------
    dict[str, Any]
        Dictionary containing test fixtures for coefficient multiplication
    """
    f = lambda x: exp(x)
    g = lambda x: cos(x)
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
    fc = Chebtech2.initfun(f, fn).prolong(hn).coeffs
    gc = Chebtech2.initfun(g, gn).prolong(hn).coeffs
    hc = coeffmult(fc, gc)
    HC = Chebtech2.initfun(h, hn).coeffs
    assert infnorm(hc - HC) <= 2e1 * eps
