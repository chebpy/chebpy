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

from chebpy.algorithms import adaptive, bary, clenshaw, coeffmult, standard_chop
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
        for _j, chebpts in enumerate(ptsarry):
            for _k, xx in enumerate(evalpts):
                print(f"Testing bary {fun.__name__}")
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
        for _j, chebpts in enumerate(ptsarry):
            for _k, xx in enumerate(evalpts):
                print(f"Testing clenshaw {fun.__name__}")
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


# ---------------------------------------------------------------------------
# standard_chop: verify against MATLAB reference test vectors
# (from standardChop.m header comments)
# ---------------------------------------------------------------------------


class TestStandardChop:
    """Tests for standard_chop matching the MATLAB reference implementation."""

    @staticmethod
    def _matlab_vectors():
        """Return the test vectors from the MATLAB standardChop.m header."""
        coeffs = 10.0 ** (-(np.arange(1, 51, dtype=float)))
        random = np.cos(np.arange(1, 51, dtype=float) ** 2)
        return coeffs, random

    def test_clean_decay(self):
        """standardChop(coeffs) should return 18."""
        coeffs, _ = self._matlab_vectors()
        assert standard_chop(coeffs) == 18

    def test_noise_at_eps(self):
        """standardChop(coeffs + 1e-16*random) should return 15."""
        coeffs, random = self._matlab_vectors()
        assert standard_chop(coeffs + 1e-16 * random) == 15

    def test_noise_at_1e_13(self):
        """standardChop(coeffs + 1e-13*random) should return 13."""
        coeffs, random = self._matlab_vectors()
        assert standard_chop(coeffs + 1e-13 * random) == 13

    def test_noise_at_1e_10_default_tol(self):
        """standardChop(coeffs + 1e-10*random) should return 50 (not happy)."""
        coeffs, random = self._matlab_vectors()
        assert standard_chop(coeffs + 1e-10 * random) == 50

    def test_noise_at_1e_10_with_tol(self):
        """standardChop(coeffs + 1e-10*random, 1e-10) should return 10."""
        coeffs, random = self._matlab_vectors()
        assert standard_chop(coeffs + 1e-10 * random, tol=1e-10) == 10

    def test_short_input_unchanged(self):
        """Vectors shorter than 17 should be returned with cutoff == n."""
        for length in [1, 5, 10, 16]:
            c = np.ones(length)
            assert standard_chop(c) == length

    def test_all_zeros(self):
        """All-zero coefficients should chop to 1."""
        assert standard_chop(np.zeros(50)) == 1

    def test_cutoff_at_least_one(self):
        """Cutoff should never be less than 1."""
        # Rapidly decaying coeffs with long plateau at machine eps
        coeffs = np.concatenate([np.array([1.0]), np.full(49, 1e-20)])
        cutoff = standard_chop(coeffs)
        assert cutoff >= 1


# ---------------------------------------------------------------------------
# adaptive: verify vscale guard for near-zero functions
# ---------------------------------------------------------------------------


class TestAdaptiveVscale:
    """Tests for the vscale guard in the adaptive constructor."""

    def test_exact_zero_function(self):
        """The zero function should produce a single zero coefficient."""
        coeffs = adaptive(Chebtech, lambda x: 0 * x)
        assert coeffs.size == 1
        assert coeffs[0] == 0.0

    def test_near_zero_function(self):
        """A function at floating-point noise level should be treated as zero."""
        # np.maximum(0, ...) on values well below zero still produces ~eps noise
        coeffs = adaptive(Chebtech, lambda x: np.maximum(0.0, x - 100.0))
        assert coeffs.size == 1
        assert coeffs[0] == 0.0

    def test_nonzero_function_unaffected(self):
        """Normal functions should not be short-circuited by the vscale guard."""
        coeffs = adaptive(Chebtech, np.sin)
        # sin(x) on [-1,1] needs more than 1 coefficient
        assert coeffs.size > 1
        # Evaluate at a test point to check accuracy
        from chebpy.algorithms import clenshaw

        x = np.array([0.5])
        assert abs(clenshaw(x, coeffs) - np.sin(0.5)) < 1e-12

    def test_small_but_nonzero_function(self):
        """A function with tiny but non-negligible values should still resolve."""
        # 1e-8 * sin(x) has vscale ~ 1e-8, well above eps
        coeffs = adaptive(Chebtech, lambda x: 1e-8 * np.sin(x))
        assert coeffs.size > 1

    def test_hat_function_with_breakpoints(self):
        """Hat functions with breakpoints should converge without warnings."""
        import warnings

        from chebpy import chebfun

        bkpts = [round(-1 + k * 0.2, 10) for k in range(11)]
        for j in range(11):
            xj = round(-1 + j * 0.2, 10)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                chebfun(
                    lambda t, _xj=xj: np.maximum(0, 1 - 5 * np.abs(t - _xj)),
                    bkpts,
                )
