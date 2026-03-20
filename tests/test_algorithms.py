"""Unit-tests for chebpy/algorithms.py.

This module contains tests for the algorithm functions in the chebpy library,
including barycentric interpolation (bary), Clenshaw evaluation (clenshaw),
coefficient multiplication (coeffmult), and convolution-related algorithms
(cheb2leg, leg2cheb, _conv_legendre, Chebfun.conv).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from chebpy.algorithms import (
    _conv_legendre,
    adaptive,
    bary,
    cheb2leg,
    clenshaw,
    coeffmult,
    leg2cheb,
    standard_chop,
)
from chebpy.chebfun import Chebfun
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

    def test_zero_plateau(self):
        """Plateau found where envelope is exactly zero."""
        # A few nonzero coeffs followed by exact zeros; long enough for Step 2
        # to detect a zero plateau, entering the branch cutoff = plateau_point.
        coeffs = np.zeros(50)
        coeffs[0] = 1.0
        coeffs[1] = 0.5
        cutoff = standard_chop(coeffs)
        assert cutoff >= 1
        assert cutoff <= 50


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


# ---------------------------------------------------------------------------
# Convolution helpers
# ---------------------------------------------------------------------------


def _numerical_conv(f, g, x: float, n: int = 100_000) -> float:
    """Numerically compute (f ★ g)(x) via the rectangle rule."""
    lo = max(-1.0, x - 1.0)
    hi = min(1.0, x + 1.0)
    if lo >= hi:
        return 0.0
    t = np.linspace(lo, hi, n + 1)
    return float(np.trapezoid(f(t) * g(x - t), t))


def _eval_legendre_series(coeffs: np.ndarray, s: float) -> float:
    """Evaluate a Legendre series at point s."""
    return float(np.polynomial.legendre.legval(s, coeffs))


# ---------------------------------------------------------------------------
# cheb2leg tests
# ---------------------------------------------------------------------------


class TestCheb2Leg:
    """Tests for the cheb2leg conversion."""

    def test_t0_equals_p0(self) -> None:
        """T_0 = P_0, so cheb2leg([1]) = [1]."""
        result = cheb2leg(np.array([1.0]))
        assert np.allclose(result, [1.0])

    def test_t1_equals_p1(self) -> None:
        """T_1 = P_1, so cheb2leg([0, 1]) = [0, 1]."""
        result = cheb2leg(np.array([0.0, 1.0]))
        assert np.allclose(result, [0.0, 1.0])

    def test_t2_in_legendre(self) -> None:
        """T_2 = 2x^2-1 = (4/3)P_2 - (1/3)P_0 in Legendre basis."""
        result = cheb2leg(np.array([0.0, 0.0, 1.0]))
        assert np.allclose(result, [-1.0 / 3.0, 0.0, 4.0 / 3.0])

    def test_single_element(self) -> None:
        """Single-element arrays pass through unchanged."""
        assert np.allclose(cheb2leg(np.array([3.0])), [3.0])

    def test_zero_array(self) -> None:
        """Zero coefficients map to zero."""
        c = np.zeros(5)
        assert np.allclose(cheb2leg(c), np.zeros(5))

    def test_roundtrip(self) -> None:
        """Cheb -> leg -> cheb should recover the original."""
        c = rng.random(10)
        assert np.allclose(leg2cheb(cheb2leg(c)), c, atol=1e-12)

    def test_polynomial_evaluation(self) -> None:
        """Both representations evaluate to the same polynomial values."""
        c = rng.random(6)
        leg_c = cheb2leg(c)
        xx = np.linspace(-1, 1, 50)
        cheb_vals = np.polynomial.chebyshev.chebval(xx, c)
        leg_vals = np.polynomial.legendre.legval(xx, leg_c)
        assert np.allclose(cheb_vals, leg_vals, atol=1e-12)


# ---------------------------------------------------------------------------
# leg2cheb tests
# ---------------------------------------------------------------------------


class TestLeg2Cheb:
    """Tests for the leg2cheb conversion."""

    def test_empty_input(self) -> None:
        """leg2cheb of an empty array returns an empty array."""
        result = leg2cheb(np.array([]))
        assert result.size == 0

    def test_p0_equals_t0(self) -> None:
        """P_0 = T_0, so leg2cheb([1]) = [1]."""
        assert np.allclose(leg2cheb(np.array([1.0])), [1.0])

    def test_p1_equals_t1(self) -> None:
        """P_1 = T_1, so leg2cheb([0,1]) = [0,1]."""
        assert np.allclose(leg2cheb(np.array([0.0, 1.0])), [0.0, 1.0])

    def test_p2_in_chebyshev(self) -> None:
        """P_2 = (3x^2-1)/2 = (1/4)T_0 + (3/4)T_2 in Chebyshev basis."""
        result = leg2cheb(np.array([0.0, 0.0, 1.0]))
        assert np.allclose(result, [0.25, 0.0, 0.75])

    def test_single_element(self) -> None:
        """Single-element arrays pass through unchanged."""
        assert np.allclose(leg2cheb(np.array([5.0])), [5.0])

    def test_roundtrip(self) -> None:
        """Leg -> cheb -> leg should recover the original."""
        c = rng.random(8)
        assert np.allclose(cheb2leg(leg2cheb(c)), c, atol=1e-12)

    def test_polynomial_evaluation(self) -> None:
        """Both representations evaluate to the same polynomial values."""
        c = rng.random(7)
        cheb_c = leg2cheb(c)
        xx = np.linspace(-1, 1, 50)
        leg_vals = np.polynomial.legendre.legval(xx, c)
        cheb_vals = np.polynomial.chebyshev.chebval(xx, cheb_c)
        assert np.allclose(leg_vals, cheb_vals, atol=1e-12)


# ---------------------------------------------------------------------------
# _conv_legendre tests
# ---------------------------------------------------------------------------


class TestConvLegendre:
    """Tests for the _conv_legendre algorithm."""

    def _eval(self, gamma_left, gamma_right, x):
        """Evaluate the piecewise Legendre result at x ∈ [-2, 2]."""
        if x <= 0:
            return _eval_legendre_series(gamma_left, x + 1.0)
        else:
            return _eval_legendre_series(gamma_right, x - 1.0)

    def test_constant_functions(self) -> None:
        """f=1, g=1 → triangle (x+2 on [-2,0]; 2-x on [0,2])."""
        gl, gr = _conv_legendre(np.array([1.0]), np.array([1.0]))
        assert np.allclose(gl, [1.0, 1.0])
        assert np.allclose(gr, [1.0, -1.0])

    def test_constant_evaluation(self) -> None:
        """Verify constant convolution values at several points."""
        gl, gr = _conv_legendre(np.array([1.0]), np.array([1.0]))
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            expected = max(0.0, 2.0 - abs(x))
            assert abs(self._eval(gl, gr, x) - expected) < 1e-13

    def test_linear_times_constant(self) -> None:
        """f=x, g=1: verify against numerical integration."""

        def identity(t):
            return t

        def ones(t):
            return np.ones_like(t)

        leg_f = np.array([0.0, 1.0])
        leg_g = np.array([1.0])
        gl, gr = _conv_legendre(leg_f, leg_g)
        for x in np.linspace(-1.8, 1.8, 10):
            alg = self._eval(gl, gr, x)
            expected = _numerical_conv(identity, ones, x)
            assert abs(alg - expected) < 1e-10

    def test_linear_times_linear(self) -> None:
        """f=x, g=x: verify against numerical integration."""

        def identity(t):
            return t

        leg_f = np.array([0.0, 1.0])
        leg_g = np.array([0.0, 1.0])
        gl, gr = _conv_legendre(leg_f, leg_g)
        for x in np.linspace(-1.8, 1.8, 10):
            alg = self._eval(gl, gr, x)
            expected = _numerical_conv(identity, identity, x)
            assert abs(alg - expected) < 1e-10

    def test_commutativity(self) -> None:
        """Convolution is commutative: f★g = g★f."""
        a = rng.random(4)
        b = rng.random(3)
        gl_ab, gr_ab = _conv_legendre(a, b)
        gl_ba, gr_ba = _conv_legendre(b, a)
        assert np.allclose(gl_ab, gl_ba, atol=1e-12)
        assert np.allclose(gr_ab, gr_ba, atol=1e-12)

    def test_boundary_values_zero(self) -> None:
        """Convolution at ±2 must be 0 (empty overlap)."""
        a = rng.random(4)
        b = rng.random(4)
        gl, gr = _conv_legendre(a, b)
        assert abs(_eval_legendre_series(gl, -1.0)) < 1e-10
        assert abs(_eval_legendre_series(gr, 1.0)) < 1e-10

    def test_smooth_functions(self) -> None:
        """f=cos, g=sin on [-1,1]: verify against numerical integration."""
        f_cheb = Chebfun.initfun_adaptive(np.cos)
        g_cheb = Chebfun.initfun_adaptive(np.sin)
        leg_f = cheb2leg(f_cheb.funs[0].coeffs)
        leg_g = cheb2leg(g_cheb.funs[0].coeffs)
        gl, gr = _conv_legendre(leg_f, leg_g)
        for x in [-1.5, -0.7, 0.0, 0.7, 1.5]:
            alg = self._eval(gl, gr, x)
            expected = _numerical_conv(np.cos, np.sin, x)
            assert abs(alg - expected) < 1e-12


# ---------------------------------------------------------------------------
# Chebfun.conv tests
# ---------------------------------------------------------------------------


class TestChebfunConv:
    """Tests for the Chebfun.conv method."""

    def test_domain_constant_functions(self) -> None:
        """f=g=1 on [-1,1]: result domain is [-2, 2] with mid at 0."""
        f = Chebfun.initconst(1.0, [-1, 1])
        h = f.conv(f)
        assert np.allclose(h.domain, [-2.0, 0.0, 2.0])

    def test_triangle_function(self) -> None:
        """f=g=1 on [-1,1]: result is the triangle function on [-2,2]."""
        f = Chebfun.initconst(1.0, [-1, 1])
        h = f.conv(f)
        xs = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        expected = np.maximum(0.0, 2.0 - np.abs(xs))
        assert np.allclose(h(xs), expected, atol=1e-12)

    def test_general_interval(self) -> None:
        """f=g=1 on [0,1]: result is the triangle on [0,2]."""
        f = Chebfun.initconst(1.0, [0, 1])
        h = f.conv(f)
        assert np.allclose(h.domain, [0.0, 1.0, 2.0])
        xs = np.array([0.25, 0.5, 1.0, 1.5, 1.75])
        expected = np.where(xs <= 1.0, xs, 2.0 - xs)
        assert np.allclose(h(xs), expected, atol=1e-12)

    def test_sin_cos_convolution(self) -> None:
        """f=sin, g=cos on [-1,1]: compare against numerical quadrature."""
        f = Chebfun.initfun_adaptive(np.sin)
        g = Chebfun.initfun_adaptive(np.cos)
        h = f.conv(g)
        for x in np.linspace(-1.8, 1.8, 9):
            alg = float(h(x))
            expected = _numerical_conv(np.sin, np.cos, x)
            assert abs(alg - expected) < 1e-12

    def test_exp_function(self) -> None:
        """f=g=exp on [-1,1]: compare against numerical quadrature."""
        f = Chebfun.initfun_adaptive(np.exp)
        h = f.conv(f)
        for x in np.linspace(-1.8, 1.8, 7):
            alg = float(h(x))
            expected = _numerical_conv(np.exp, np.exp, x)
            assert abs(alg - expected) < 1e-10

    def test_commutativity(self) -> None:
        """(f ★ g)(x) == (g ★ f)(x) at test points."""
        f = Chebfun.initfun_adaptive(np.sin)
        g = Chebfun.initfun_adaptive(np.cos)
        h_fg = f.conv(g)
        h_gf = g.conv(f)
        xs = np.linspace(-1.8, 1.8, 20)
        assert np.allclose(h_fg(xs), h_gf(xs), atol=1e-12)

    def test_linearity_in_f(self) -> None:
        """(a*f + b*g) ★ h == a*(f ★ h) + b*(g ★ h)."""
        a, b = 2.0, -3.0
        f = Chebfun.initfun_adaptive(np.sin)
        g = Chebfun.initfun_adaptive(np.cos)
        h_func = Chebfun.initfun_adaptive(np.exp)
        xs = np.linspace(-1.8, 1.8, 20)
        lhs = (a * f + b * g).conv(h_func)(xs)
        rhs = a * f.conv(h_func)(xs) + b * g.conv(h_func)(xs)
        assert np.allclose(lhs, rhs, atol=1e-10)

    def test_result_is_two_piece(self) -> None:
        """Result always has exactly two pieces."""
        f = Chebfun.initfun_adaptive(np.sin)
        h = f.conv(f)
        assert h.funs.size == 2

    def test_empty_chebfun(self) -> None:
        """Convolution with an empty Chebfun returns an empty Chebfun."""
        f = Chebfun.initfun_adaptive(np.sin)
        empty = Chebfun.initempty()
        assert f.conv(empty).isempty
        assert empty.conv(f).isempty

    def test_multipiece_now_supported(self) -> None:
        """Conv handles multi-piece Chebfuns without raising."""
        f_multi = Chebfun.initfun_adaptive(np.sin, [-1, 0, 1])
        f_single = Chebfun.initfun_adaptive(np.sin)
        # Should not raise — just verify it produces a Chebfun
        h = f_multi.conv(f_single)
        assert not h.isempty

    def test_different_domains(self) -> None:
        """Conv now supports inputs on different domains."""
        f = Chebfun.initfun_adaptive(np.sin, [-1, 1])
        g = Chebfun.initfun_adaptive(np.cos, [-2, 2])
        h = f.conv(g)
        assert not h.isempty
        # Output domain should be [-1+(-2), 1+2] = [-3, 3]
        assert np.isclose(float(h.domain[0]), -3.0)
        assert np.isclose(float(h.domain[-1]), 3.0)

    def test_boundary_is_zero(self) -> None:
        """Convolution vanishes at the ends of the result domain."""
        f = Chebfun.initfun_adaptive(np.sin)
        h = f.conv(f)
        a, b = float(h.domain[0]), float(h.domain[-1])
        assert abs(float(h(a))) < 1e-12
        assert abs(float(h(b))) < 1e-12

    def test_polynomial_exact(self) -> None:
        """For polynomial f and g the result is exact to machine precision."""
        f = Chebfun.initfun_adaptive(lambda x: x)
        g = Chebfun.initconst(1.0)
        h = f.conv(g)
        xs_left = np.linspace(-1.9, 0.0, 10)
        xs_right = np.linspace(0.0, 1.9, 10)
        assert np.allclose(h(xs_left), xs_left * (xs_left + 2) / 2, atol=1e-13)
        assert np.allclose(h(xs_right), xs_right * (2 - xs_right) / 2, atol=1e-13)

    def test_zero_convolution(self) -> None:
        """Convolving a zero function produces all-zero gamma (mg==0 branch)."""
        f = Chebfun.initconst(0.0)
        g = Chebfun.initconst(1.0)
        h = f.conv(g)
        xs = np.linspace(-1.9, 1.9, 10)
        np.testing.assert_allclose(h(xs), 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# General numerical convolution helper for piecewise tests
# ---------------------------------------------------------------------------


def _numerical_conv_general(f: Chebfun, g: Chebfun, x: float, n: int = 200_000) -> float:
    """Numerically compute (f ★ g)(x) via the trapezoidal rule.

    Works for arbitrary domains: domain(f) = [a, b], domain(g) = [c, d].
    """
    a, b = float(f.domain[0]), float(f.domain[-1])
    c, d = float(g.domain[0]), float(g.domain[-1])
    lo = max(a, x - d)
    hi = min(b, x - c)
    if lo >= hi:
        return 0.0
    t = np.linspace(lo, hi, n + 1)
    return float(np.trapezoid(f(t) * g(x - t), t))


# ---------------------------------------------------------------------------
# Piecewise Chebfun.conv tests
# ---------------------------------------------------------------------------


class TestChebfunConvPiecewise:
    """Tests for convolution of piecewise Chebfuns."""

    def test_two_piece_self_conv(self) -> None:
        """Convolve a 2-piece chebfun with itself via quadrature comparison."""
        f = Chebfun.initfun_adaptive(lambda x: np.abs(x), [-1, 0, 1])
        h = f.conv(f)
        for x in np.linspace(-1.8, 1.8, 9):
            expected = _numerical_conv_general(f, f, x)
            assert abs(float(h(x)) - expected) < 1e-6, f"Mismatch at x={x}: chebpy={float(h(x))}, numerical={expected}"

    def test_piecewise_domain(self) -> None:
        """Output breakpoints are the pairwise sums of input breakpoints."""
        f = Chebfun.initfun_adaptive(lambda x: np.ones_like(x), [-1, 0, 1])
        g = Chebfun.initconst(1.0, [-1, 1])
        h = f.conv(g)
        # f breaks: -1, 0, 1; g breaks: -1, 1
        # pairwise sums: -2, 0, -1, 1, 0, 2 → sorted unique: -2, -1, 0, 1, 2
        expected_breaks = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert np.allclose(h.breakpoints, expected_breaks, atol=1e-12)

    def test_piecewise_constant_triangle(self) -> None:
        """Convolving a 2-piece constant with a single constant is a triangle."""
        f = Chebfun.initconst(1.0, [-1, 0, 1])
        g = Chebfun.initconst(1.0, [-1, 1])
        h = f.conv(g)
        xs = np.array([-1.5, -1.0, 0.0, 1.0, 1.5])
        expected = np.maximum(0.0, 2.0 - np.abs(xs))
        assert np.allclose(h(xs), expected, atol=1e-10)

    def test_multipiece_commutativity(self) -> None:
        """(f ★ g)(x) == (g ★ f)(x) for piecewise inputs."""
        f = Chebfun.initfun_adaptive(lambda x: np.abs(x), [-1, 0, 1])
        g = Chebfun.initfun_adaptive(np.cos, [-1, 1])
        h_fg = f.conv(g)
        h_gf = g.conv(f)
        xs = np.linspace(-1.8, 1.8, 15)
        assert np.allclose(h_fg(xs), h_gf(xs), atol=1e-6)

    def test_piecewise_both_multi(self) -> None:
        """Convolve two 2-piece chebfuns and verify against numerical quadrature."""
        f = Chebfun.initfun_adaptive(lambda x: np.abs(x), [-1, 0, 1])
        g = Chebfun.initfun_adaptive(lambda x: np.where(x < 0, 1 + x, 1 - x), [-1, 0, 1])
        h = f.conv(g)
        for x in np.linspace(-1.8, 1.8, 7):
            expected = _numerical_conv_general(f, g, x)
            assert abs(float(h(x)) - expected) < 1e-5

    def test_different_domain_constant(self) -> None:
        """Convolve constants on different domains: f=1 on [0,1], g=1 on [-1,0]."""
        f = Chebfun.initconst(1.0, [0, 1])
        g = Chebfun.initconst(1.0, [-1, 0])
        h = f.conv(g)
        # Output domain: [0+(-1), 1+0] = [-1, 1], breakpoint at 0
        assert np.isclose(float(h.domain[0]), -1.0)
        assert np.isclose(float(h.domain[-1]), 1.0)
        # Triangle on [-1, 1]: peak at 0 with value 1
        assert abs(float(h(0.0)) - 1.0) < 1e-10

    def test_different_width_domains(self) -> None:
        """Convolve funs on domains of different widths."""
        f = Chebfun.initconst(1.0, [0, 1])
        g = Chebfun.initconst(1.0, [0, 2])
        h = f.conv(g)
        # Output on [0, 3]; breakpoints at 0, 1, 2, 3
        assert np.isclose(float(h.domain[0]), 0.0)
        assert np.isclose(float(h.domain[-1]), 3.0)
        # h(x) should be a trapezoid: ramp up on [0,1], flat=1 on [1,2], ramp down on [2,3]
        assert abs(float(h(0.5)) - 0.5) < 1e-10
        assert abs(float(h(1.5)) - 1.0) < 1e-10
        assert abs(float(h(2.5)) - 0.5) < 1e-10

    def test_piecewise_boundary_zero(self) -> None:
        """Convolution of piecewise chebfuns vanishes at the domain ends."""
        f = Chebfun.initfun_adaptive(lambda x: np.abs(x), [-1, 0, 1])
        g = Chebfun.initfun_adaptive(np.sin, [-1, 1])
        h = f.conv(g)
        a, b = float(h.domain[0]), float(h.domain[-1])
        assert abs(float(h(a))) < 1e-8
        assert abs(float(h(b))) < 1e-8

    def test_three_piece_conv(self) -> None:
        """Convolve a 3-piece chebfun with a single piece."""
        f = Chebfun.initfun_adaptive(lambda x: np.ones_like(x), [-1, -0.5, 0.5, 1])
        g = Chebfun.initconst(1.0, [-1, 1])
        h = f.conv(g)
        # Should give the triangle function (same as 1-piece constant case)
        xs = np.array([0.0, 0.5, 1.0, -0.5, -1.0])
        expected = np.maximum(0.0, 2.0 - np.abs(xs))
        assert np.allclose(h(xs), expected, atol=1e-10)
