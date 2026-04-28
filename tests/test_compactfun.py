"""Tests for the :class:`chebpy.compactfun.CompactFun` class."""

from __future__ import annotations

import numpy as np
import pytest

from chebpy import CompactFun, chebfun
from chebpy.bndfun import Bndfun
from chebpy.exceptions import CompactFunConstructionError, DivergentIntegralError
from chebpy.utilities import Interval


# -----------------------------
# direct construction
# -----------------------------
class TestInitFunAdaptive:
    """Tests for :meth:`CompactFun.initfun_adaptive`."""

    def test_gaussian_doubly_infinite(self) -> None:
        f = CompactFun.initfun_adaptive(lambda x: np.exp(-(x**2)), (-np.inf, np.inf))
        a, b = f.numerical_support
        assert np.isfinite(a)
        assert np.isfinite(b)
        # Gaussian decays well within ±20.
        assert -50.0 < a < -3.0
        assert 3.0 < b < 50.0
        # Logical interval is preserved.
        assert not np.isfinite(f.support[0])
        assert not np.isfinite(f.support[1])

    def test_gaussian_evaluation(self) -> None:
        f = CompactFun.initfun_adaptive(lambda x: np.exp(-(x**2)), (-np.inf, np.inf))
        x = np.linspace(-2.0, 2.0, 11)
        np.testing.assert_allclose(f(x), np.exp(-(x**2)), atol=1e-10)
        # Outside numerical support → exactly zero.
        assert f(1000.0) == 0.0
        assert f(-1000.0) == 0.0

    def test_semi_infinite_left(self) -> None:
        f = CompactFun.initfun_adaptive(lambda x: np.exp(x), (-np.inf, 0.0))
        a, b = f.numerical_support
        assert b == 0.0
        assert np.isfinite(a)

    def test_semi_infinite_right(self) -> None:
        f = CompactFun.initfun_adaptive(lambda x: np.exp(-x), (0.0, np.inf))
        a, b = f.numerical_support
        assert a == 0.0
        assert np.isfinite(b)
        np.testing.assert_allclose(f(0.5), np.exp(-0.5), atol=1e-10)

    def test_finite_interval_behaves_like_bndfun(self) -> None:
        f = CompactFun.initfun_adaptive(lambda x: np.sin(x), (-1.0, 2.0))
        assert tuple(f.support) == (-1.0, 2.0)
        assert tuple(f.numerical_support) == (-1.0, 2.0)
        x = np.linspace(-1.0, 2.0, 17)
        np.testing.assert_allclose(f(x), np.sin(x), atol=1e-12)

    def test_heavy_tail_refused(self) -> None:
        # 1/(1+x^2) decays only as 1/x^2 — too slowly for our threshold/budget.
        with pytest.raises(CompactFunConstructionError):
            CompactFun.initfun_adaptive(lambda x: 1.0 / (1.0 + x * x), (-np.inf, np.inf))

    def test_oscillating_tail_refused(self) -> None:
        # sin(x) does not converge to a constant at infinity.
        with pytest.raises(CompactFunConstructionError):
            CompactFun.initfun_adaptive(lambda x: np.sin(x), (-np.inf, np.inf))


class TestInitConst:
    """Tests for :meth:`CompactFun.initconst`."""

    def test_zero_constant_unbounded_ok(self) -> None:
        f = CompactFun.initconst(0.0, (-np.inf, np.inf))
        assert f(1000.0) == 0.0
        assert f(0.0) == 0.0

    def test_nonzero_constant_unbounded_promoted_to_tails(self) -> None:
        # Non-zero constants are now representable on unbounded intervals via
        # tail-constant metadata.
        f = CompactFun.initconst(1.0, (-np.inf, np.inf))
        assert f.tail_left == 1.0
        assert f.tail_right == 1.0
        assert f(1e6) == 1.0
        g = CompactFun.initconst(2.5, (0.0, np.inf))
        assert g.tail_left == 0.0  # finite endpoint
        assert g.tail_right == 2.5
        assert g(1e6) == 2.5

    def test_constant_finite_ok(self) -> None:
        f = CompactFun.initconst(3.0, (-1.0, 2.0))
        assert f(0.5) == pytest.approx(3.0)


class TestInitIdentity:
    """Tests for :meth:`CompactFun.initidentity`."""

    def test_unbounded_refused(self) -> None:
        with pytest.raises(CompactFunConstructionError):
            CompactFun.initidentity((-np.inf, np.inf))
        with pytest.raises(CompactFunConstructionError):
            CompactFun.initidentity((0.0, np.inf))

    def test_finite_ok(self) -> None:
        f = CompactFun.initidentity((-1.0, 2.0))
        x = np.linspace(-1.0, 2.0, 5)
        np.testing.assert_allclose(f(x), x, atol=1e-12)


class TestInitEmpty:
    """Tests for :meth:`CompactFun.initempty`."""

    def test_initempty(self) -> None:
        f = CompactFun.initempty()
        assert f.isempty


# -----------------------------
# evaluation
# -----------------------------
class TestCall:
    """Tests for :meth:`CompactFun.__call__`."""

    def test_scalar_returns_scalar(self) -> None:
        f = CompactFun.initfun_adaptive(lambda x: np.exp(-(x**2)), (-np.inf, np.inf))
        result = f(0.0)
        assert np.isscalar(result) or np.ndim(result) == 0

    def test_array_returns_array(self) -> None:
        f = CompactFun.initfun_adaptive(lambda x: np.exp(-(x**2)), (-np.inf, np.inf))
        x = np.array([-1.0, 0.0, 1.0, 1000.0])
        y = f(x)
        assert y.shape == x.shape
        assert y[-1] == 0.0


# -----------------------------
# properties
# -----------------------------
class TestProperties:
    """Tests for the support / endvalues / repr properties of CompactFun."""

    def test_endvalues_at_inf_are_zero(self) -> None:
        f = CompactFun.initfun_adaptive(lambda x: np.exp(-(x**2)), (-np.inf, np.inf))
        ev = f.endvalues
        assert ev[0] == 0.0
        assert ev[1] == 0.0

    def test_endvalues_semi_infinite(self) -> None:
        f = CompactFun.initfun_adaptive(lambda x: np.exp(-x), (0.0, np.inf))
        ev = f.endvalues
        assert ev[0] == pytest.approx(1.0, abs=1e-10)
        assert ev[1] == 0.0


# -----------------------------
# calculus / utilities
# -----------------------------
class TestCumsumZeroTail:
    """Tests for :meth:`CompactFun.cumsum` on zero-tail inputs."""

    def test_cumsum_gaussian(self) -> None:
        # cumsum of exp(-x^2) anchored at -inf has tail_right = sqrt(pi).
        f = CompactFun.initfun_adaptive(lambda x: np.exp(-(x**2)), (-np.inf, np.inf))
        F = f.cumsum()
        assert isinstance(F, CompactFun)
        assert F.tail_left == 0.0
        assert F.tail_right == pytest.approx(np.sqrt(np.pi), abs=1e-10)
        # Outside numerical support, F evaluates to its tails.
        assert F(-1000.0) == 0.0
        assert F(1000.0) == pytest.approx(np.sqrt(np.pi), abs=1e-10)


class TestRestrictAndTranslate:
    """Tests for :meth:`restrict` and :meth:`translate`."""

    def test_restrict_to_finite_returns_bndfun(self) -> None:
        f = CompactFun.initfun_adaptive(lambda x: np.exp(-(x**2)), (-np.inf, np.inf))
        g = f.restrict(Interval(-1.0, 1.0))
        assert isinstance(g, Bndfun)
        x = np.linspace(-1.0, 1.0, 11)
        np.testing.assert_allclose(g(x), np.exp(-(x**2)), atol=1e-10)

    def test_translate_preserves_logical_interval(self) -> None:
        f = CompactFun.initfun_adaptive(lambda x: np.exp(-(x**2)), (-np.inf, np.inf))
        g = f.translate(2.0)
        assert isinstance(g, CompactFun)
        # logical interval still infinite
        assert not np.isfinite(g.support[0])
        assert not np.isfinite(g.support[1])
        # numerical support shifts by 2
        np.testing.assert_allclose(g.numerical_support, np.asarray(f.numerical_support) + 2.0, atol=1e-12)
        # value relation
        np.testing.assert_allclose(g(2.0), f(0.0), atol=1e-12)


# -----------------------------
# integration with chebfun()
# -----------------------------
class TestChebfunIntegration:
    """Tests that ``chebfun(...)`` constructs CompactFun pieces on infinite intervals."""

    def test_chebfun_doubly_infinite(self) -> None:
        f = chebfun(lambda x: np.exp(-(x**2)), [-np.inf, np.inf])
        assert len(f.funs) == 1
        assert isinstance(f.funs[0], CompactFun)
        assert f.sum() == pytest.approx(np.sqrt(np.pi), abs=1e-10)

    def test_chebfun_semi_infinite(self) -> None:
        f = chebfun(lambda x: np.exp(-x), [0.0, np.inf])
        assert isinstance(f.funs[0], CompactFun)
        assert f.sum() == pytest.approx(1.0, abs=1e-10)

    def test_chebfun_piecewise_mixed(self) -> None:
        f = chebfun(lambda x: np.exp(-np.abs(x)), [-np.inf, 0.0, np.inf])
        assert len(f.funs) == 2
        assert all(isinstance(p, CompactFun) for p in f.funs)
        assert f.sum() == pytest.approx(2.0, abs=1e-10)

    def test_chebfun_piecewise_with_finite_middle(self) -> None:
        # Outer pieces unbounded; middle piece finite — middle should be a Bndfun.
        f = chebfun(lambda x: np.exp(-(x**2)), [-np.inf, -1.0, 1.0, np.inf])
        assert len(f.funs) == 3
        assert isinstance(f.funs[0], CompactFun)
        assert isinstance(f.funs[1], Bndfun)
        assert isinstance(f.funs[2], CompactFun)
        assert f.sum() == pytest.approx(np.sqrt(np.pi), abs=1e-10)

    def test_chebfun_piecewise_multiple_finite_breakpoints(self) -> None:
        # chebfun(f, [-inf, -2, 0, 3, +inf]): two CompactFun ends, two Bndfun middles.
        f = chebfun(lambda x: np.exp(-(x**2)), [-np.inf, -2.0, 0.0, 3.0, np.inf])
        assert len(f.funs) == 4
        assert isinstance(f.funs[0], CompactFun)
        assert isinstance(f.funs[1], Bndfun)
        assert isinstance(f.funs[2], Bndfun)
        assert isinstance(f.funs[3], CompactFun)

        # Logical breakpoints: leftmost = -inf, rightmost = +inf, interior finite.
        bps = f.breakpoints
        assert not np.isfinite(bps[0])
        assert bps[1] == pytest.approx(-2.0)
        assert bps[2] == pytest.approx(0.0)
        assert bps[3] == pytest.approx(3.0)
        assert not np.isfinite(bps[-1])

        # Per-piece supports line up correctly.
        assert not np.isfinite(f.funs[0].support[0])
        assert f.funs[0].support[1] == pytest.approx(-2.0)
        np.testing.assert_allclose(f.funs[1].support, [-2.0, 0.0])
        np.testing.assert_allclose(f.funs[2].support, [0.0, 3.0])
        assert f.funs[3].support[0] == pytest.approx(3.0)
        assert not np.isfinite(f.funs[3].support[1])

        # Total integral is the same Gaussian integral, regardless of breakpoint choice.
        assert f.sum() == pytest.approx(np.sqrt(np.pi), abs=1e-10)

        # Pointwise evaluation matches the underlying function across all four pieces
        # (sample inside each), at interior breakpoints, and far outside numerical support.
        x = np.array([-1000.0, -10.0, -2.0, -1.0, 0.0, 1.5, 3.0, 10.0, 1000.0])
        expected = np.exp(-(x**2))
        # Far outside numerical support → exact zero (function is below tolerance there).
        expected[0] = 0.0
        expected[-1] = 0.0
        np.testing.assert_allclose(f(x), expected, atol=1e-9)

    def test_chebfun_evaluation_outside_storage(self) -> None:
        f = chebfun(lambda x: np.exp(-(x**2)), [-np.inf, np.inf])
        # Far outside numerical support but inside logical domain → 0, not NaN.
        y = f(np.array([-1000.0, 0.0, 1000.0]))
        assert y[0] == 0.0
        assert y[2] == 0.0
        assert y[1] == pytest.approx(1.0, abs=1e-10)


# -----------------------------
# convolution
# -----------------------------
class TestConvolution:
    """Tests for ``Chebfun.conv`` involving :class:`CompactFun` inputs."""

    def test_gaussian_self_convolution(self) -> None:
        # (exp(-x^2)) ★ (exp(-x^2)) = sqrt(pi/2) * exp(-x^2/2); total mass = pi.
        f = chebfun(lambda x: np.exp(-(x**2)), [-np.inf, np.inf])
        h = f.conv(f)
        assert h.sum() == pytest.approx(np.pi, abs=1e-8)
        assert h(0.0) == pytest.approx(np.sqrt(np.pi / 2.0), abs=1e-8)
        assert h(2.0) == pytest.approx(np.sqrt(np.pi / 2.0) * np.exp(-2.0), abs=1e-8)
        # Outermost pieces should be CompactFun (logical support is unbounded).
        assert isinstance(h.funs[0], CompactFun)
        assert isinstance(h.funs[-1], CompactFun)
        assert not np.isfinite(h.breakpoints[0])
        assert not np.isfinite(h.breakpoints[-1])

    def test_exp_self_convolution_semi_infinite(self) -> None:
        # exp(-x) ★ exp(-x) on [0, inf) = x * exp(-x) (Gamma(2,1)); total mass 1.
        f = chebfun(lambda x: np.exp(-x), [0.0, np.inf])
        h = f.conv(f)
        assert h.sum() == pytest.approx(1.0, abs=1e-8)
        for xi in (0.5, 1.0, 3.0):
            assert h(xi) == pytest.approx(xi * np.exp(-xi), abs=1e-8)
        # Right edge piece is CompactFun, left endpoint is finite (= 0.0).
        assert isinstance(h.funs[-1], CompactFun)
        assert h.breakpoints[0] == pytest.approx(0.0)
        assert not np.isfinite(h.breakpoints[-1])

    def test_compact_with_finite_bndfun(self) -> None:
        # exp(-x^2) on R convolved with a bump on [-1, 1].
        f = chebfun(lambda x: np.exp(-(x**2)), [-np.inf, np.inf])
        g = chebfun(lambda x: 1.0 - x * x, [-1.0, 1.0])
        h = f.conv(g)
        # Mass of result equals product of masses.
        assert h.sum() == pytest.approx(np.sqrt(np.pi) * (4.0 / 3.0), abs=1e-8)
        # Result has unbounded logical support.
        assert not np.isfinite(h.breakpoints[0])
        assert not np.isfinite(h.breakpoints[-1])


# -----------------------------
# rootfinding
# -----------------------------
class TestRoots:
    """Tests for the spurious-root filtering on :class:`CompactFun`."""

    def test_no_spurious_roots_for_strictly_positive_decay(self) -> None:
        # exp(-x^2) has no real roots; raw polynomial roots in noise regions
        # (near the storage-interval boundary) must be filtered out.
        f = chebfun(lambda x: np.exp(-(x**2)), [-np.inf, np.inf])
        roots = f.roots()
        assert roots.size == 0

    def test_genuine_root_kept_in_doubly_infinite(self) -> None:
        # (x - 2) * exp(-x^2) has a single sign change at x = 2.
        f = chebfun(lambda x: (x - 2.0) * np.exp(-(x**2)), [-np.inf, np.inf])
        roots = f.roots()
        assert roots.size == 1
        assert roots[0] == pytest.approx(2.0, abs=1e-8)

    def test_multiple_genuine_roots_in_semi_infinite(self) -> None:
        # (x - 1)(x - 3) * exp(-x) has two simple roots; spurious noise roots
        # at large x must be filtered.
        f = chebfun(lambda x: (x - 1.0) * (x - 3.0) * np.exp(-x), [0.0, np.inf])
        roots = f.roots()
        assert roots.size == 2
        np.testing.assert_allclose(roots, [1.0, 3.0], atol=1e-8)

    def test_oscillation_roots_match_expected_grid(self) -> None:
        # sin(10x) * exp(-x) on [0, inf) has roots at k*pi/10 for as many k as
        # exp(-x) stays above noise. All retained roots must lie on this grid.
        # Roots in the well-resolved region (exp(-x) >> noise floor) should
        # match k*pi/10 to high precision.
        f = chebfun(lambda x: np.sin(10.0 * x) * np.exp(-x), [0.0, np.inf])
        roots = f.roots()
        assert roots.size > 50  # should retain dozens of genuine roots
        well_resolved = roots[roots < 25.0]
        assert well_resolved.size > 70
        ks = np.rint(well_resolved * 10.0 / np.pi)
        np.testing.assert_allclose(well_resolved, ks * np.pi / 10.0, atol=1e-8)
        # No retained root should be outside the storage interval.
        a_s, b_s = f.funs[0].numerical_support
        assert roots.min() >= a_s
        assert roots.max() <= b_s

    def test_roots_via_chebfun_filtered(self) -> None:
        # Calling .roots() on the parent Chebfun also benefits from filtering.
        f = chebfun(lambda x: np.exp(-(x**2)), [-np.inf, np.inf])
        assert f.roots().size == 0


# -----------------------------
# tail constants (plan 02b)
# -----------------------------
class TestTailConstants:
    """Tests for non-zero asymptotic limits via tail metadata."""

    def test_tanh_construction(self) -> None:
        f = CompactFun.initfun_adaptive(np.tanh, (-np.inf, np.inf))
        assert f.tail_left == pytest.approx(-1.0, abs=1e-12)
        assert f.tail_right == pytest.approx(1.0, abs=1e-12)
        # Outside numerical support, the tail constants are returned exactly.
        assert f(-1e10) == pytest.approx(-1.0)
        assert f(1e10) == pytest.approx(1.0)
        # Inside the support, evaluation matches np.tanh.
        x = np.linspace(-3.0, 3.0, 7)
        np.testing.assert_allclose(f(x), np.tanh(x), atol=1e-10)

    def test_logistic_one_sided_tail(self) -> None:
        # Logistic 1/(1+exp(-x)) -> 0 at -inf, 1 at +inf.
        f = CompactFun.initfun_adaptive(lambda x: 1.0 / (1.0 + np.exp(-x)), (-np.inf, np.inf))
        assert f.tail_left == pytest.approx(0.0, abs=1e-12)
        assert f.tail_right == pytest.approx(1.0, abs=1e-12)

    def test_repr_shows_tails(self) -> None:
        f = CompactFun.initfun_adaptive(np.tanh, (-np.inf, np.inf))
        assert "tails=" in repr(f)

    def test_endvalues_use_tails(self) -> None:
        f = CompactFun.initfun_adaptive(np.tanh, (-np.inf, np.inf))
        ev = f.endvalues
        assert ev[0] == pytest.approx(-1.0)
        assert ev[1] == pytest.approx(1.0)

    def test_sum_diverges_with_nonzero_tail(self) -> None:
        f = CompactFun.initfun_adaptive(np.tanh, (-np.inf, np.inf))
        with pytest.raises(DivergentIntegralError):
            f.sum()

    def test_cumsum_diverges_with_nonzero_tail(self) -> None:
        f = CompactFun.initfun_adaptive(np.tanh, (-np.inf, np.inf))
        with pytest.raises(DivergentIntegralError):
            f.cumsum()

    def test_diff_zeros_tails(self) -> None:
        f = CompactFun.initfun_adaptive(np.tanh, (-np.inf, np.inf))
        df = f.diff()
        assert df.tail_left == 0.0
        assert df.tail_right == 0.0

    def test_neg_flips_tails(self) -> None:
        f = CompactFun.initfun_adaptive(np.tanh, (-np.inf, np.inf))
        g = -f
        assert g.tail_left == pytest.approx(1.0, abs=1e-12)
        assert g.tail_right == pytest.approx(-1.0, abs=1e-12)

    def test_scalar_add_propagates_tails(self) -> None:
        f = CompactFun.initfun_adaptive(np.tanh, (-np.inf, np.inf))
        g = f + 2.0
        assert g.tail_left == pytest.approx(1.0, abs=1e-12)
        assert g.tail_right == pytest.approx(3.0, abs=1e-12)
        h = 2.0 + f
        assert h.tail_left == pytest.approx(1.0, abs=1e-12)
        assert h.tail_right == pytest.approx(3.0, abs=1e-12)

    def test_scalar_sub_propagates_tails(self) -> None:
        f = CompactFun.initfun_adaptive(np.tanh, (-np.inf, np.inf))
        g = f - 1.0
        assert g.tail_left == pytest.approx(-2.0, abs=1e-12)
        assert g.tail_right == pytest.approx(0.0, abs=1e-12)
        h = 1.0 - f
        assert h.tail_left == pytest.approx(2.0, abs=1e-12)
        assert h.tail_right == pytest.approx(0.0, abs=1e-12)

    def test_scalar_mul_propagates_tails(self) -> None:
        f = CompactFun.initfun_adaptive(np.tanh, (-np.inf, np.inf))
        g = 3.0 * f
        assert g.tail_left == pytest.approx(-3.0, abs=1e-12)
        assert g.tail_right == pytest.approx(3.0, abs=1e-12)

    def test_conv_refuses_nonzero_tails(self) -> None:
        # Convolution against tanh would diverge \u2014 we refuse early.
        f = chebfun(np.tanh, [-np.inf, np.inf])
        g = chebfun(lambda x: np.exp(-(x**2)), [-np.inf, np.inf])
        with pytest.raises(DivergentIntegralError):
            f.conv(g)
        with pytest.raises(DivergentIntegralError):
            g.conv(f)
