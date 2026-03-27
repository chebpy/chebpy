"""Unit-tests for the chebfun module (src/chebpy/chebfun.py).

This single file consolidates every test that exercises the Chebfun class,
including shared (generic) tests imported from ``tests._shared``.
"""

# ---------------------------------------------------------------------------
# Shared / generic tests  (algebra, calculus-empty, class_usage, complex,
# plotting-generic, roots-empty, ufuncs-empty)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Module-specific imports
# ---------------------------------------------------------------------------
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebpy import chebfun, pwc
from chebpy.bndfun import Bndfun
from chebpy.chebfun import Chebfun
from chebpy.exceptions import (
    BadFunLengthArgument,
    IntervalGap,
    IntervalOverlap,
    InvalidDomain,
)
from chebpy.utilities import Domain, Interval
from tests.generic.algebra import *  # noqa: F403
from tests.generic.class_usage import test_support, test_translate_empty  # noqa: F401
from tests.generic.complex import (  # noqa: F401
    test_calculus,
    test_complexfun_properties,
    test_real_imag,
    test_rho_ellipse_construction,
    test_roots,
)
from tests.generic.plotting import test_plot, test_plot_complex  # noqa: F401
from tests.generic.roots import (
    rootstestfuns,
    test_empty,  # noqa: F401
)
from tests.generic.ufuncs import test_emptycase, ufunc_parameter  # noqa: F401
from tests.utilities import cos, eps, exp, sin

pi = np.pi

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def testdomains() -> list:
    """List of domains and test tolerances for testing algebraic operations."""
    return [
        ([-1, 1], 5 * eps),
        ([-2, 1], 5 * eps),
        ([-1, 2], 5 * eps),
        ([-5, 9], 35 * eps),
    ]


# ---------------------------------------------------------------------------
#  Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for Chebfun construction methods."""

    @staticmethod
    def _make_funs():
        f = lambda x: exp(x)  # noqa: E731
        fun0 = Bndfun.initfun_adaptive(f, Interval(-1, 0))
        fun1 = Bndfun.initfun_adaptive(f, Interval(0, 1))
        fun2 = Bndfun.initfun_adaptive(f, Interval(-0.5, 0.5))
        fun3 = Bndfun.initfun_adaptive(f, Interval(2, 2.5))
        fun4 = Bndfun.initfun_adaptive(f, Interval(-3, -2))
        return fun0, fun1, fun2, fun3, fun4

    def test__init__pass(self):
        fun0, fun1, fun2, *_ = self._make_funs()
        Chebfun([fun0])
        Chebfun([fun1])
        Chebfun([fun2])
        Chebfun([fun0, fun1])

    def test__init__fail(self):
        fun0, fun1, fun2, fun3, fun4 = self._make_funs()
        with pytest.raises(IntervalOverlap):
            Chebfun([fun1, fun0, fun2])
        with pytest.raises(IntervalOverlap):
            Chebfun([fun1, fun2])
        with pytest.raises(IntervalGap):
            Chebfun([fun0, fun3])
        with pytest.raises(IntervalGap):
            Chebfun([fun1, fun4])

    def test_initempty(self):
        assert Chebfun.initempty().funs.size == 0

    def test_initconst(self):
        fun0, fun1, fun2, *_ = self._make_funs()
        assert Chebfun.initconst(1, [-1, 1]).isconst
        assert Chebfun.initconst(-10, np.linspace(-1, 1, 11)).isconst
        assert Chebfun.initconst(3, [-2, 0, 1]).isconst
        assert Chebfun.initconst(3.14, np.linspace(-100, -90, 11)).isconst
        assert not Chebfun([fun0]).isconst
        assert not Chebfun([fun1]).isconst
        assert not Chebfun([fun2]).isconst
        assert not Chebfun([fun0, fun1]).isconst

    def test_initidentity(self):
        _doms = (
            np.linspace(-1, 1, 2),
            np.linspace(-1, 1, 11),
            np.linspace(-10, 17, 351),
            np.linspace(-9.3, -3.2, 22),
            np.linspace(2.5, 144.3, 2112),
        )
        for _dom in _doms:
            ff = Chebfun.initidentity(_dom)
            a, b = ff.support
            xx = np.linspace(a, b, 1001)
            tol = eps * ff.hscale
            assert np.max(np.abs(ff(xx) - xx)) <= tol
        ff = Chebfun.initidentity()
        a, b = ff.support
        xx = np.linspace(a, b, 1001)
        tol = eps * ff.hscale
        assert np.max(np.abs(ff(xx) - xx)) <= tol

    def test_initfun_adaptive_continuous_domain(self):
        f = lambda x: exp(x)  # noqa: E731
        ff = Chebfun.initfun_adaptive(f, [-2, -1])
        assert ff.funs.size == 1
        xx = np.linspace(-2, -1, 1001)
        assert np.max(np.abs(f(xx) - ff(xx))) <= 2 * eps

    def test_initfun_adaptive_piecewise_domain(self):
        f = lambda x: exp(x)  # noqa: E731
        ff = Chebfun.initfun_adaptive(f, [-2, -1, 0, 1, 2])
        assert ff.funs.size == 4
        xx = np.linspace(-2, 2, 1001)
        assert np.max(np.abs(f(xx) - ff(xx))) <= 10 * eps

    def test_initfun_adaptive_raises(self):
        f = lambda x: exp(x)  # noqa: E731
        with pytest.raises(InvalidDomain):
            Chebfun.initfun_adaptive(f, [-2])
        with pytest.raises(InvalidDomain):
            Chebfun.initfun_adaptive(f, domain=[-2])
        with pytest.raises(InvalidDomain):
            Chebfun.initfun_adaptive(f, domain=0)

    def test_initfun_adaptive_empty_domain(self):
        f = lambda x: exp(x)  # noqa: E731
        cheb = Chebfun.initfun_adaptive(f, domain=[])
        assert cheb.isempty

    def test_initfun_fixedlen_continuous_domain(self):
        f = lambda x: exp(x)  # noqa: E731
        ff = Chebfun.initfun_fixedlen(f, 20, [-2, -1])
        assert ff.funs.size == 1
        xx = np.linspace(-2, -1, 1001)
        assert np.max(np.abs(f(xx) - ff(xx))) <= 1e1 * eps

    def test_initfun_fixedlen_piecewise_domain_0(self):
        f = lambda x: exp(x)  # noqa: E731
        ff = Chebfun.initfun_fixedlen(f, 30, [-2.0, 0.0, 1.0])
        assert ff.funs.size == 2
        xx = np.linspace(-2, 1, 1001)
        assert np.max(np.abs(f(xx) - ff(xx))) <= 1e1 * eps

    def test_initfun_fixedlen_piecewise_domain_1(self):
        f = lambda x: exp(x)  # noqa: E731
        ff = Chebfun.initfun_fixedlen(f, [30, 20], [-2, 0, 1])
        assert ff.funs.size == 2
        xx = np.linspace(-2, 1, 1001)
        assert np.max(np.abs(f(xx) - ff(xx))) <= 1e1 * eps

    def test_initfun_fixedlen_raises(self):
        f = lambda x: exp(x)  # noqa: E731
        initfun = Chebfun.initfun_fixedlen
        with pytest.raises(InvalidDomain):
            initfun(f, 10, [-2])
        with pytest.raises(InvalidDomain):
            initfun(f, n=10, domain=[-2])
        with pytest.raises(InvalidDomain):
            initfun(f, n=10, domain=0)
        with pytest.raises(BadFunLengthArgument):
            initfun(f, [30, 40], [-1, 1])
        with pytest.raises(TypeError):
            initfun(f, [], [-2, -1, 0])

    def test_initfun_fixedlen_empty_domain(self):
        f = lambda x: exp(x)  # noqa: E731
        cheb = Chebfun.initfun_fixedlen(f, n=10, domain=[])
        assert cheb.isempty

    def test_initfun_fixedlen_succeeds(self):
        f = lambda x: exp(x)  # noqa: E731
        dom = [-2, -1, 0]
        g0 = Chebfun.initfun_adaptive(f, dom)
        g1 = Chebfun.initfun_fixedlen(f, [None, None], dom)
        g2 = Chebfun.initfun_fixedlen(f, [None, 40], dom)
        g3 = Chebfun.initfun_fixedlen(f, None, dom)
        for fun_a, fun_b in zip(g1, g0, strict=False):
            assert np.sum(fun_a.coeffs - fun_b.coeffs) == 0
        for fun_a, fun_b in zip(g3, g0, strict=False):
            assert np.sum(fun_a.coeffs - fun_b.coeffs) == 0
        assert np.sum(g2.funs[0].coeffs - g0.funs[0].coeffs) == 0


# ---------------------------------------------------------------------------
#  Properties
# ---------------------------------------------------------------------------


class TestProperties:
    """Tests for Chebfun properties."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.f0 = Chebfun.initempty()
        self.f1 = Chebfun.initconst(1.0)
        self.f2 = Chebfun.initidentity([-1, 1])
        self.f3 = Chebfun.initfun_adaptive(lambda x: x**2, [-2, 0, 1, 3])

    def test_breakpoints(self):
        assert np.all(self.f3.breakpoints == np.array([-2, 0, 1, 3]))

    def test_domain(self):
        assert self.f0.domain == Domain([])
        assert self.f1.domain == Domain([-1, 1])
        assert self.f2.domain == Domain([-1, 1])
        assert self.f3.domain == Domain([-2, 0, 1, 3])

    def test_hscale(self):
        assert self.f0.hscale == 0.0
        assert self.f1.hscale == 1.0
        assert self.f3.hscale == 3.0

    def test_isempty(self):
        assert self.f0.isempty
        assert not self.f1.isempty

    def test_isconst(self):
        assert not self.f0.isconst
        assert self.f1.isconst
        assert not self.f2.isconst
        assert not self.f3.isconst

    def test_support(self):  # noqa: F811
        assert self.f0.support.size == 0
        assert np.all(self.f1.support == np.array([-1, 1]))
        assert np.all(self.f2.support == np.array([-1, 1]))
        assert np.all(self.f3.support == np.array([-2, 3]))

    def test_vscale(self):
        assert self.f1.vscale == 1.0


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------


class TestEvaluation:
    """Tests for Chebfun evaluation."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.f0 = Chebfun.initempty()
        self.f1 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 1])
        self.f2 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 0, 1, 2])

    def test__call__empty_chebfun(self):
        assert self.f0(np.linspace(-1, 1, 100)).size == 0

    def test__call__empty_array(self):
        assert self.f0(np.array([])).size == 0
        assert self.f1(np.array([])).size == 0
        assert self.f2(np.array([])).size == 0

    def test__call__point_evaluation(self):
        assert np.isscalar(self.f1(0.1))

    def test__call__singleton(self):
        a = self.f1(np.array(0.1))
        b = self.f1(np.array([0.1]))
        c = self.f1([0.1])
        assert a.size == 1
        assert b.size == 1
        assert c.size == 1
        assert np.equal(a, b).all()
        assert np.equal(b, c).all()
        assert np.equal(a, c).all()

    def test__call__breakpoints(self):
        x1 = self.f1.breakpoints
        x2 = self.f2.breakpoints
        assert np.equal(self.f1(x1), [1, 1]).all()
        assert np.equal(self.f2(x2), [1, 0, 1, 4]).all()

    def test__call__outside_interval(self):
        x = np.linspace(-3, 3, 100)
        assert np.isfinite(self.f1(x)).all()
        assert np.isfinite(self.f2(x)).all()

    def test__call__general_evaluation(self):
        def f(x):
            return sin(4 * x) + exp(cos(14 * x)) - 1.4

        npts = 50000
        dom1 = [-1, 1]
        dom2 = [-1, 0, 1]
        dom3 = [-2, -0.3, 1.2]
        ff1 = Chebfun.initfun_adaptive(f, dom1)
        ff2 = Chebfun.initfun_adaptive(f, dom2)
        ff3 = Chebfun.initfun_adaptive(f, dom3)
        x1 = np.linspace(dom1[0], dom1[-1], npts)
        x2 = np.linspace(dom2[0], dom2[-1], npts)
        x3 = np.linspace(dom3[0], dom3[-1], npts)
        assert np.max(np.abs(f(x1) - ff1(x1))) <= 5e1 * eps
        assert np.max(np.abs(f(x2) - ff2(x2))) <= 5e1 * eps
        assert np.max(np.abs(f(x3) - ff3(x3))) <= 5e1 * eps


# ---------------------------------------------------------------------------
#  Calculus
# ---------------------------------------------------------------------------


class TestCalculus:
    """Tests for Chebfun calculus operations."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.f1 = Chebfun.initfun_adaptive(lambda x: sin(4 * x - 1.4), [-1, 1])
        self.f2 = Chebfun.initfun_adaptive(exp, [-1, 1])
        self.f3 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 1])
        self.f4 = Chebfun.initfun_adaptive(lambda x: x**3, [-1, 1])

    def test_sum(self):
        assert abs(self.f1.sum() - 0.372895407327895) < 1e-14
        assert abs(self.f2.sum() - (np.exp(1) - np.exp(-1))) < 1e-14

    def test_diff(self):
        xx = np.linspace(-1, 1, 100)
        assert np.max(np.abs(self.f3.diff()(xx) - 2 * xx)) < 1e-14
        assert np.max(np.abs(self.f4.diff()(xx) - 3 * xx**2)) < 1e-14

    def test_cumsum(self):
        assert (self.f3.cumsum().diff() - self.f3).isconst
        assert (self.f4.cumsum().diff() - self.f4).isconst

    def test_dot(self):
        assert abs(self.f3.dot(self.f4)) < 1e-14
        assert abs(self.f3.dot(self.f3) - 2 / 5) < 1e-14

    def test_dot_commute(self):
        assert abs(self.f1.dot(self.f2) - self.f2.dot(self.f1)) < 1e-14

    def test_dot_empty(self, emptyfun):
        assert emptyfun.dot(self.f1) == 0
        assert self.f1.dot(emptyfun) == 0

    def test_diff_order(self):
        xx = np.linspace(-1, 1, 100)
        assert np.max(np.abs(self.f3.diff(2)(xx) - 2)) < 1e-14
        assert np.max(np.abs(self.f4.diff(2)(xx) - 6 * xx)) < 1e-14
        assert np.max(np.abs(self.f4.diff(3)(xx) - 6)) < 1e-14
        assert np.max(np.abs(self.f4.diff(4)(xx))) < 1e-14

    def test_diff_successive(self):
        xx = np.linspace(-1, 1, 100)
        assert np.max(np.abs(self.f4.diff().diff()(xx) - self.f4.diff(2)(xx))) < 1e-14

    def test_norm(self):
        expected_norm_f2 = np.sqrt((np.exp(2) - np.exp(-2)) / 2)
        assert abs(self.f2.norm() - expected_norm_f2) < 1e-14
        expected_norm_f3 = np.sqrt(2 / 5)
        assert abs(self.f3.norm() - expected_norm_f3) < 1e-14
        assert self.f1.norm() >= 0
        assert self.f2.norm() >= 0

    def test_norm_empty(self, emptyfun):
        assert emptyfun.norm() == 0

    def test_norm_relation_to_dot(self):
        assert abs(self.f2.norm() ** 2 - self.f2.dot(self.f2)) < 1e-14
        assert abs(self.f3.norm() ** 2 - self.f3.dot(self.f3)) < 1e-14


class TestCalculusEdgeCases:
    """Additional edge-case tests for Chebfun calculus operations."""

    def test_diff_edge_cases(self):
        f = chebfun(lambda x: x**3, [-1, 1])
        f0 = f.diff(0)
        assert f0 == f
        with pytest.raises(ValueError, match="-1"):
            f.diff(-1)

    def test_diff_higher_order(self):
        f = chebfun(lambda x: x**4, [-1, 1])
        xx = np.linspace(-1, 1, 20)
        assert np.allclose(f.diff(4)(xx), 24, atol=1e-14)

    def test_multipiece_cumsum_with_many_pieces(self):
        f = chebfun(lambda x: x * 0 + 1, [-1, -0.5, 0, 0.5, 1])
        f_int = f.cumsum()
        xx = np.array([-0.5, 0, 0.5])
        left = xx - 1e-10
        right = xx + 1e-10
        assert np.allclose(f_int(left), f_int(right), atol=1e-5)

    def test_cumsum_multipiece_continuity(self):
        f = pwc(domain=[-1, 0, 1], values=[-1, 1])
        f_cumsum = f.cumsum()
        left_val = f_cumsum(-1e-13)
        right_val = f_cumsum(1e-13)
        assert np.abs(left_val - right_val) < 1e-12

    def test_norm_l1(self):
        f = chebfun(lambda x: x, [-1, 1])
        assert np.isclose(f.norm(p=1), 1.0, atol=1e-14)

    def test_norm_l2(self):
        f = chebfun(lambda x: x, [-1, 1])
        assert np.isclose(f.norm(), np.sqrt(2 / 3), atol=1e-14)

    def test_norm_linf(self):
        f = chebfun(lambda x: x**2, [-1, 1])
        assert np.isclose(f.norm(np.inf), 1.0, atol=1e-14)

    def test_norm_l3(self):
        f = chebfun(lambda x: x * 0 + 1, [-1, 1])
        assert np.isclose(f.norm(p=3), 2 ** (1 / 3), atol=1e-14)

    def test_norm_invalid_p(self):
        f = chebfun(lambda x: x, [-1, 1])
        with pytest.raises(ValueError, match="-1"):
            f.norm(p=-1)
        with pytest.raises(ValueError, match="0"):
            f.norm(p=0)

    def test_norm_l2_multipiece(self):
        f = chebfun(lambda x: x, [-1, 0, 1])
        assert np.isclose(f.norm(), np.sqrt(2 / 3), atol=1e-14)

    def test_norm_linf_with_multiple_extrema(self):
        f = chebfun(lambda x: np.sin(3 * x), [-np.pi, np.pi])
        assert np.isclose(f.norm(np.inf), 1.0, atol=1e-14)


# ---------------------------------------------------------------------------
#  Class usage
# ---------------------------------------------------------------------------


class TestClassUsage:
    """Tests for Chebfun class usage."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.f1 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 1])
        self.f2 = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 0, 1, 2])

    def test__str__(self):
        assert str(self.f1) == repr(self.f1)
        assert "Chebfun" in str(self.f1)
        assert "interval" in str(self.f1)

    def test__repr__(self):
        assert "Chebfun" in repr(self.f1)
        assert "domain" in repr(self.f1)

    def test__iter__(self):
        assert len(list(self.f1)) == 1
        assert len(list(self.f2)) == 3

    def test_x_property(self):
        xx = np.linspace(-1, 1, 100)
        assert np.max(np.abs(self.f1.x(xx) - xx)) <= eps
        xx = np.linspace(-1, 2, 100)
        assert np.max(np.abs(self.f2.x(xx) - xx)) <= eps

    def test_restrict_(self):
        g1 = self.f1.copy()
        g1.restrict_([-0.5, 0.5])
        assert g1.domain == Domain([-0.5, 0.5])
        xx = np.linspace(-0.5, 0.5, 100)
        assert np.max(np.abs(g1(xx) - self.f1(xx))) <= 5 * eps

        g2 = self.f2.copy()
        g2.restrict_([-0.5, 1.5])
        assert g2.domain == Domain([-0.5, 0, 1, 1.5])
        xx = np.linspace(-0.5, 1.5, 100)
        assert np.max(np.abs(g2(xx) - self.f2(xx))) <= 5 * eps

        g2 = self.f2.copy()
        g2.restrict_([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
        assert g2.domain == Domain([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
        xx = np.linspace(-0.8, 1.8, 100)
        assert np.max(np.abs(g2(xx) - self.f2(xx))) <= 5 * eps

    def test_simplify(self):
        assert self.f1.simplify() == self.f1
        assert self.f2.simplify() == self.f2

    def test_simplify_empty(self, emptyfun):
        assert emptyfun.simplify().isempty

    def test_restrict(self):
        g1 = self.f1.restrict([-0.5, 0.5])
        assert g1.domain == Domain([-0.5, 0.5])
        xx = np.linspace(-0.5, 0.5, 100)
        assert np.max(np.abs(g1(xx) - self.f1(xx))) <= 5 * eps

        g2 = self.f2.restrict([-0.5, 1.5])
        assert g2.domain == Domain([-0.5, 0, 1, 1.5])
        xx = np.linspace(-0.5, 1.5, 100)
        assert np.max(np.abs(g2(xx) - self.f2(xx))) <= 5 * eps

        g2 = self.f2.restrict([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
        assert g2.domain == Domain([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8])
        xx = np.linspace(-0.8, 1.8, 100)
        assert np.max(np.abs(g2(xx) - self.f2(xx))) <= 5 * eps

    def test_translate(self):
        g1 = self.f1.translate(1)
        assert g1.domain == Domain([0, 2])
        xx = np.linspace(-1, 1, 100)
        yy = xx + 1
        assert np.max(np.abs(g1(yy) - self.f1(xx))) <= 2 * eps

        g2 = self.f2.translate(1)
        assert g2.domain == Domain([0, 1, 2, 3])
        xx = np.linspace(-1, 2, 100)
        yy = xx + 1
        assert np.max(np.abs(g2(yy) - self.f2(xx))) <= 2 * eps

        g1 = self.f1.translate(-1)
        assert g1.domain == Domain([-2, 0])
        xx = np.linspace(-1, 1, 100)
        yy = xx - 1
        assert np.max(np.abs(g1(yy) - self.f1(xx))) <= 2 * eps

    def test_copy(self, constfun):
        ff = constfun
        gg = ff.copy()
        assert ff == ff
        assert gg == gg
        assert ff == gg
        gg.domain = Domain([-1, 0, 0.5, 1])
        assert gg != ff


# ---------------------------------------------------------------------------
#  Domain-breaking operations
# ---------------------------------------------------------------------------


class TestDomainBreakingOps:
    """Tests for Chebfun domain-breaking operations (max, min)."""

    def test_maximum_multipiece(self):
        x = chebfun("x", np.linspace(-2, 3, 11))
        y = chebfun(2, x.domain)
        g = (x**y).maximum(1.5)
        t = np.linspace(-2, 3, 2001)
        assert np.max(np.abs(np.maximum(t**2, 1.5) - g(t))) <= 1e1 * eps

    def test_minimum_multipiece(self):
        x = chebfun("x", np.linspace(-2, 3, 11))
        y = chebfun(2, x.domain)
        g = (x**y).minimum(1.5)
        t = np.linspace(-2, 3, 2001)
        assert np.max(np.abs(np.minimum(t**2, 1.5) - g(t))) <= 1e1 * eps

    @pytest.mark.parametrize(
        ("domain", "tol"),
        [([-1, 1], eps), ([-1, 0, 1], eps), ([-2, 0, 3], eps)],
    )
    def test_maximum_identity_constant(self, domain, tol):
        x = chebfun("x", domain)
        g = x.maximum(0)
        xx = np.linspace(domain[0], domain[-1], 1001)
        assert np.max(np.abs(np.maximum(xx, 0) - g(xx))) <= tol

    @pytest.mark.parametrize(
        ("domain", "tol"),
        [([-1, 1], eps), ([-1, 0, 1], eps), ([-2, 0, 3], eps)],
    )
    def test_minimum_identity_constant(self, domain, tol):
        x = chebfun("x", domain)
        g = x.minimum(0)
        xx = np.linspace(domain[0], domain[-1], 1001)
        assert np.max(np.abs(np.minimum(xx, 0) - g(xx))) <= tol

    @pytest.mark.parametrize(
        ("domain", "tol"),
        [([-1, 1], eps), ([-1, 0, 1], eps), ([-2, 0, 3], eps)],
    )
    def test_maximum_sin_cos(self, domain, tol):
        f1 = chebfun(np.sin, domain)
        f2 = chebfun(np.cos, domain)
        g = f1.maximum(f2)
        xx = np.linspace(domain[0], domain[-1], 1001)
        vscl = max([f1.vscale, f2.vscale])
        hscl = max([f1.hscale, f2.hscale])
        lscl = max([fun.size for fun in np.append(f1.funs, f2.funs)])
        assert np.max(np.abs(np.maximum(np.sin(xx), np.cos(xx)) - g(xx))) <= vscl * hscl * lscl * tol

    @pytest.mark.parametrize(
        ("domain", "tol"),
        [([-1, 1], eps), ([-1, 0, 1], eps), ([-2, 0, 3], eps)],
    )
    def test_minimum_sin_cos(self, domain, tol):
        f1 = chebfun(np.sin, domain)
        f2 = chebfun(np.cos, domain)
        g = f1.minimum(f2)
        xx = np.linspace(domain[0], domain[-1], 1001)
        vscl = max([f1.vscale, f2.vscale])
        hscl = max([f1.hscale, f2.hscale])
        lscl = max([fun.size for fun in np.append(f1.funs, f2.funs)])
        assert np.max(np.abs(np.minimum(np.sin(xx), np.cos(xx)) - g(xx))) <= vscl * hscl * lscl * tol

    def test_maximum_empty(self):
        f_empty = Chebfun.initempty()
        f = chebfun(np.sin, [-1, 1])
        assert f_empty.maximum(f).isempty
        assert f.maximum(f_empty).isempty
        assert f_empty.maximum(f_empty).isempty

    def test_minimum_empty(self):
        f_empty = Chebfun.initempty()
        f = chebfun(np.sin, [-1, 1])
        assert f_empty.minimum(f).isempty
        assert f.minimum(f_empty).isempty
        assert f_empty.minimum(f_empty).isempty


# ---------------------------------------------------------------------------
#  Roots
# ---------------------------------------------------------------------------


class TestRoots:
    """Tests for Chebfun roots functionality."""

    @pytest.mark.parametrize(("f", "roots"), rootstestfuns)
    def test_roots(self, f, roots):  # noqa: F811
        ff = Chebfun.initfun_adaptive(f)
        rts = ff.roots()
        assert np.max(np.abs(rts - roots)) <= 1e-15

    def test_roots_const(self):
        f_nonzero = Chebfun.initconst(1.0, [-1, 1])
        assert f_nonzero.roots().size == 0
        f_zero = Chebfun.initconst(0.0, [-1, 1])
        assert f_zero.roots().size == 0

    def test_roots_multiple_intervals(self):
        f = Chebfun.initfun_adaptive(lambda x: sin(2 * pi * x), [-1, 0, 1])
        roots = f.roots()
        expected_roots = np.array([-1, -0.5, 0, 0.5, 1])
        assert roots.size == expected_roots.size
        assert np.allclose(np.sort(roots), expected_roots, atol=1e-10)

    def test_roots_high_frequency(self):
        f = Chebfun.initfun_adaptive(lambda x: sin(10 * pi * x), [-1, 1])
        roots = f.roots()
        expected_roots = np.linspace(-1, 1, 21)
        assert roots.size == expected_roots.size
        assert np.allclose(np.sort(roots), expected_roots, atol=1e-10)


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------


class TestPlotting:
    """Tests for Chebfun plotting methods."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.f1 = Chebfun.initfun_adaptive(sin, [-1, 1])
        self.f2 = Chebfun.initfun_adaptive(cos, [-1, 1])
        self.f3 = Chebfun.initfun_adaptive(exp, [-1, 1])
        self.f4 = Chebfun.initfun_adaptive(lambda x: np.exp(1j * np.pi * x), [-1, 1])

    def test_plot_multiple(self):
        fig, ax = plt.subplots()
        self.f1.plot(ax=ax)
        self.f2.plot(ax=ax, color="r")
        self.f3.plot(ax=ax, color="g")
        plt.close(fig)

    def test_plotcoeffs(self):
        fig, ax = plt.subplots()
        self.f1.plotcoeffs(ax=ax)
        plt.close(fig)
        fig, ax = plt.subplots()
        self.f2.plotcoeffs(ax=ax)
        plt.close(fig)
        fig, ax = plt.subplots()
        self.f3.plotcoeffs(ax=ax)

    def test_plotcoeffs_multiple(self):
        _fig, ax = plt.subplots()
        self.f1.plotcoeffs(ax=ax)
        self.f2.plotcoeffs(ax=ax, color="r")

    def test_plot_with_options(self):
        _fig, ax = plt.subplots()
        self.f1.plot(ax=ax, color="r", linestyle="--", linewidth=2, marker="o", markersize=5)

    def test_plotcoeffs_with_options(self):
        _fig, ax = plt.subplots()
        self.f1.plotcoeffs(ax=ax, color="g", marker="s", markersize=8, linestyle="-.")

    def test_plot_multipiece(self):
        domain = np.linspace(-1, 1, 5)
        f_multi = Chebfun.initfun_adaptive(sin, domain)
        _fig, ax = plt.subplots()
        f_multi.plot(ax=ax)


# ---------------------------------------------------------------------------
#  Ufuncs
# ---------------------------------------------------------------------------


class TestUfuncs:
    """Tests for Chebfun ufunc operations."""

    def test_abs_absolute_alias(self):
        assert Chebfun.abs == Chebfun.absolute

    def test_ufuncs(self):
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

    def test_sign_splitting(self):
        f = Chebfun.initfun_adaptive(lambda x: x, [-1, 1])
        g = f.sign()
        xx = np.linspace(-1, 1, 1000)
        assert np.max(np.abs(g(xx) - np.sign(xx))) < eps
        assert g(0.0) == 0.0

    def test_sign_no_root(self):
        f = Chebfun.initfun_adaptive(lambda x: x + 2, [-1, 1])
        g = f.sign()
        xx = np.linspace(-1, 1, 100)
        assert np.allclose(g(xx), 1.0)

    def test_sign_multiple_roots(self):
        f = Chebfun.initfun_adaptive(lambda x: np.sin(2 * np.pi * x), [-1, 1])
        g = f.sign()
        assert g(0.125) == 1.0
        assert g(-0.125) == -1.0
        assert g(0.625) == -1.0
        assert g(-0.625) == 1.0

    def test_ceil_splitting(self):
        f = Chebfun.initfun_adaptive(lambda x: 3 * x, [-1, 1])
        g = f.ceil()
        xx = np.linspace(-1, 1, 1000)
        bps = g.breakpoints
        mask = np.ones_like(xx, dtype=bool)
        for bp in bps:
            mask &= np.abs(xx - bp) > 1e-10
        xx_safe = xx[mask]
        assert np.max(np.abs(g(xx_safe) - np.ceil(3 * xx_safe))) < eps

    def test_ceil_no_crossing(self):
        f = Chebfun.initfun_adaptive(lambda x: 0.1 * x + 0.5, [-1, 1])
        g = f.ceil()
        xx = np.linspace(-1, 1, 100)
        assert np.allclose(g(xx), 1.0)

    def test_floor_splitting(self):
        f = Chebfun.initfun_adaptive(lambda x: 3 * x, [-1, 1])
        g = f.floor()
        xx = np.linspace(-1, 1, 1000)
        bps = g.breakpoints
        mask = np.ones_like(xx, dtype=bool)
        for bp in bps:
            mask &= np.abs(xx - bp) > 1e-10
        xx_safe = xx[mask]
        assert np.max(np.abs(g(xx_safe) - np.floor(3 * xx_safe))) < eps

    def test_floor_no_crossing(self):
        f = Chebfun.initfun_adaptive(lambda x: 0.1 * x + 0.5, [-1, 1])
        g = f.floor()
        xx = np.linspace(-1, 1, 100)
        assert np.allclose(g(xx), 0.0)


# ---------------------------------------------------------------------------
#  Private methods
# ---------------------------------------------------------------------------


class TestPrivateMethods:
    """Tests for Chebfun private methods (_break)."""

    def test_break_1(self):
        f1 = Chebfun.initfun_adaptive(lambda x: np.sin(x - 0.1), [-2, 0, 3])
        altdom = Domain([-2, -1, 1, 2, 3])
        newdom = f1.domain.union(altdom)
        f1_new = f1._break(newdom)
        assert f1_new.domain == newdom
        assert f1_new.domain != altdom
        assert f1_new.domain != f1.domain
        xx = np.linspace(-2, 3, 1000)
        assert np.max(np.abs(f1(xx) - f1_new(xx))) <= 3 * eps

    def test_break_2(self):
        f1 = Chebfun.initfun_adaptive(lambda x: np.sin(x - 0.1), [-2, 0, 3])
        altdom = Domain([-2, 3])
        newdom = f1.domain.union(altdom)
        f1_new = f1._break(newdom)
        assert f1_new.domain == newdom
        assert f1_new.domain != altdom
        xx = np.linspace(-2, 3, 1000)
        assert np.max(np.abs(f1(xx) - f1_new(xx))) <= 3 * eps

    def test_break_3(self):
        f2 = Chebfun.initfun_adaptive(lambda x: np.sin(x - 0.1), np.linspace(-2, 3, 5))
        altdom = Domain(np.linspace(-2, 3, 1000))
        newdom = f2.domain.union(altdom)
        f2_new = f2._break(newdom)
        assert f2_new.domain == newdom
        assert f2_new.domain != altdom
        assert f2_new.domain != f2.domain
        xx = np.linspace(-2, 3, 1000)
        assert np.max(np.abs(f2(xx) - f2_new(xx))) <= 3 * eps

    def test_break_identity(self):
        f = Chebfun.initfun_adaptive(np.sin, [-1, 1])
        f_new = f._break(f.domain)
        assert f_new.domain == f.domain
        xx = np.linspace(-1, 1, 1000)
        assert np.max(np.abs(f(xx) - f_new(xx))) <= eps
        assert f is not f_new

    def test_break_with_tolerance(self):
        f = Chebfun.initfun_adaptive(np.sin, [-1, 0, 1])
        tol = 0.8 * eps
        altdom = Domain([-1 - tol, 0 + tol, 1 - tol])
        newdom = f.domain.union(altdom)
        assert newdom == f.domain
        f_new = f._break(newdom)
        assert f_new.isempty

    def test_break_multipiece(self):
        f = Chebfun.initfun_adaptive(np.sin, [-2, -1, 0, 1, 2])
        newdom = Domain([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
        f_new = f._break(newdom)
        assert f_new.domain == newdom
        xx = np.linspace(-2, 2, 1000)
        assert np.max(np.abs(f(xx) - f_new(xx))) <= 3 * eps


# ---------------------------------------------------------------------------
#  Additional coverage
# ---------------------------------------------------------------------------


class TestAdditionalCoverage:
    """Additional tests to improve coverage for chebfun.py."""

    def test_eq_different_types(self):
        f = chebfun(lambda x: x**2)
        assert (f == "not a chebfun") is False
        assert (f == 5) is False
        assert (f == [1, 2, 3]) is False

    def test_eq_empty_chebfuns(self):
        f1 = Chebfun.initempty()
        f2 = Chebfun.initempty()
        assert f1 == f2

    def test_cumsum_multiple_funs(self):
        x = np.linspace(-1, 1, 5)
        f = chebfun(lambda t: np.abs(t), domain=x)
        F = f.cumsum()
        for i in range(1, len(x) - 1):
            left_val = F(x[i] - 1e-10)
            right_val = F(x[i] + 1e-10)
            assert np.abs(left_val - right_val) < 1e-9
        xx = np.linspace(-1, 1, 100)
        assert np.max(np.abs(F.diff()(xx) - f(xx))) < 1e-10

    def test_maximum_minimum_different_supports(self):
        f = chebfun(lambda x: x**2, domain=[-1, 1])
        g = chebfun(lambda x: 1 - x**2, domain=[0, 2])
        h_max = f.maximum(g)
        assert h_max.support[0] == 0
        assert h_max.support[1] == 1
        h_min = f.minimum(g)
        assert h_min.support[0] == 0
        assert h_min.support[1] == 1

    def test_maximum_minimum_no_intersection(self):
        f = chebfun(lambda x: x**2, domain=[-2, -1])
        g = chebfun(lambda x: 1 - x**2, domain=[1, 2])
        assert f.maximum(g).isempty
        assert f.minimum(g).isempty

    def test_maximum_minimum_empty_switch(self):
        f = chebfun(lambda x: x**2, domain=[-1, 1])
        g = chebfun(lambda x: x**2, domain=[-1, 1])
        assert not f.maximum(g).isempty
        assert not f.minimum(g).isempty

    def test_absolute_method(self):
        f = chebfun(lambda x: -(x**2) - 1, domain=[-1, 1])
        abs_f = abs(f)
        assert isinstance(abs_f, Chebfun)
        xx = np.array([-1, -0.5, 0, 0.5, 1])
        assert np.max(np.abs(abs_f(xx) - np.abs(-(xx**2) - 1))) < 1e-6
        g = chebfun(lambda x: x**2 - 0.5, domain=[-1, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            abs_g = abs(g)
        assert np.max(np.abs(abs_g(xx) - np.abs(xx**2 - 0.5))) < 1e-6

    def test_imag_complex_chebfun(self):
        f = chebfun(lambda x: np.exp(1j * np.pi * x), domain=[-1, 1])
        assert f.iscomplex
        imag_f = f.imag()
        assert isinstance(imag_f, Chebfun)
        xx = np.linspace(-1, 1, 100)
        assert np.max(np.abs(imag_f(xx) - np.imag(np.exp(1j * np.pi * xx)))) < 1e-10

    def test_diff_type_error(self):
        f = chebfun(lambda x: x**2)
        with pytest.raises(TypeError):
            f.diff(1.5)
        with pytest.raises(TypeError):
            f.diff("1")
