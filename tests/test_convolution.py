"""Tests for the convolution-related algorithms and the Chebfun.conv method.

Covers:
- cheb2leg / leg2cheb coefficient conversions
- _conv_legendre: Hale-Townsend algorithm for piecewise Legendre convolution
- Chebfun.conv: user-facing convolution method
"""

import numpy as np
import pytest

from chebpy.algorithms import _conv_legendre, cheb2leg, leg2cheb
from chebpy.chebfun import Chebfun
from chebpy.exceptions import SupportMismatch

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Helpers
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
        """f=1, g=1 → triangle (x+2 on [-2,0], 2-x on [0,2])."""
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

        leg_f = np.array([0.0, 1.0])  # P_1 = x
        leg_g = np.array([1.0])  # P_0 = 1
        gl, gr = _conv_legendre(leg_f, leg_g)
        for x in np.linspace(-1.8, 1.8, 10):
            alg = self._eval(gl, gr, x)
            expected = _numerical_conv(identity, ones, x)
            assert abs(alg - expected) < 1e-10, f"Mismatch at x={x}: {alg} vs {expected}"

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
            assert abs(alg - expected) < 1e-10, f"Mismatch at x={x}: {alg} vs {expected}"

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
        # At x=-2: s=-1 for left piece
        assert abs(_eval_legendre_series(gl, -1.0)) < 1e-10
        # At x=2: s=1 for right piece
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
            assert abs(alg - expected) < 1e-12, f"Mismatch at x={x}"


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
            assert abs(alg - expected) < 1e-12, f"Mismatch at x={x}"

    def test_exp_function(self) -> None:
        """f=g=exp on [-1,1]: compare against numerical quadrature."""
        f = Chebfun.initfun_adaptive(np.exp)
        h = f.conv(f)
        for x in np.linspace(-1.8, 1.8, 7):
            alg = float(h(x))
            expected = _numerical_conv(np.exp, np.exp, x)
            assert abs(alg - expected) < 1e-10, f"Mismatch at x={x}"

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

    def test_raises_for_multipiece(self) -> None:
        """Conv raises NotImplementedError for multi-piece Chebfuns."""
        f_multi = Chebfun.initfun_adaptive(np.sin, [-1, 0, 1])
        f_single = Chebfun.initfun_adaptive(np.sin)
        with pytest.raises(NotImplementedError):
            f_multi.conv(f_single)
        with pytest.raises(NotImplementedError):
            f_single.conv(f_multi)

    def test_raises_for_support_mismatch(self) -> None:
        """Conv raises SupportMismatch when intervals differ."""
        f = Chebfun.initfun_adaptive(np.sin)
        g = Chebfun.initfun_adaptive(np.cos, [-2, 2])
        with pytest.raises(SupportMismatch):
            f.conv(g)

    def test_boundary_is_zero(self) -> None:
        """Convolution vanishes at the ends of the result domain."""
        f = Chebfun.initfun_adaptive(np.sin)
        h = f.conv(f)
        a, b = float(h.domain[0]), float(h.domain[-1])
        assert abs(float(h(a))) < 1e-12
        assert abs(float(h(b))) < 1e-12

    def test_polynomial_exact(self) -> None:
        """For polynomial f and g the result is exact to machine precision."""
        # f = x, g = 1 => (f★g)(x) = x(x+2)/2 on [-2,0], x(2-x)/2 on [0,2]
        f = Chebfun.initfun_adaptive(lambda x: x)
        g = Chebfun.initconst(1.0)
        h = f.conv(g)
        xs_left = np.linspace(-1.9, 0.0, 10)
        xs_right = np.linspace(0.0, 1.9, 10)
        assert np.allclose(h(xs_left), xs_left * (xs_left + 2) / 2, atol=1e-13)
        assert np.allclose(h(xs_right), xs_right * (2 - xs_right) / 2, atol=1e-13)
