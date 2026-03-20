"""Unit-tests for the bndfun module (src/chebpy/bndfun.py).

This single file consolidates every test that exercises the Bndfun class,
including shared (generic) tests imported from ``tests._shared``.
"""

# ---------------------------------------------------------------------------
# Shared / generic tests
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Module-specific imports
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebpy.algorithms import standard_chop
from chebpy.bndfun import Bndfun
from chebpy.chebtech import Chebtech
from chebpy.utilities import Interval
from tests.generic.algebra import *  # noqa: F403
from tests.generic.calculus import test_cumsum_empty, test_diff_empty, test_sum_empty  # noqa: F401
from tests.generic.class_usage import test_constfun_value, test_copy, test_endvalues, test_support  # noqa: F401
from tests.generic.complex import (  # noqa: F401
    test_calculus,
    test_complexfun_properties,
    test_real_imag,
    test_rho_ellipse_construction,
    test_roots,
)
from tests.generic.plotting import test_plot, test_plot_complex  # noqa: F401
from tests.generic.ufuncs import test_emptycase, ufunc_parameter  # noqa: F401
from tests.utilities import cos, eps, exp, pi, sin

# Ensure reproducibility
rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
#  Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for Bndfun construction methods."""

    def test_onefun_construction(self):
        coeffs = rng.random(10)
        subinterval = Interval()
        onefun = Chebtech(coeffs)
        f = Bndfun(onefun, subinterval)
        assert isinstance(f, Bndfun)
        assert np.max(np.abs(f.coeffs - coeffs)) < eps

    def test_identity_construction(self):
        for a, b in [(-1, 1), (-10, -2), (-2.3, 1.24), (20, 2000)]:
            itvl = Interval(a, b)
            ff = Bndfun.initidentity(itvl)
            assert ff.size == 2
            xx = np.linspace(a, b, 1001)
            tol = eps * abs(itvl).max()
            assert np.max(np.abs(ff(xx) - xx)) <= tol

    # Test functions for adaptive and fixed-length construction
    _fun_details = [
        (lambda x: x**3 + x**2 + x + 1, "poly3(x)", [-2, 3], 4),
        (lambda x: exp(x), "exp(x)", [-2, 3], 20),
        (lambda x: sin(x), "sin(x)", [-2, 3], 20),
        (lambda x: cos(20 * x), "cos(20x)", [-2, 3], 90),
        (lambda x: 0.0 * x + 1.0, "constfun", [-2, 3], 1),
        (lambda x: 0.0 * x, "zerofun", [-2, 3], 1),
    ]

    @pytest.mark.parametrize(("fun", "name", "interval", "funlen"), _fun_details)
    def test_adaptive(self, fun, name, interval, funlen):
        subinterval = Interval(*interval)
        ff = Bndfun.initfun_adaptive(fun, subinterval)
        assert ff.size in {funlen - 1, funlen}

    @pytest.mark.parametrize(("fun", "name", "interval", "_"), _fun_details)
    def test_fixedlen(self, fun, name, interval, _):
        subinterval = Interval(*interval)
        n = 100
        ff = Bndfun.initfun_fixedlen(fun, subinterval, n)
        assert ff.size == n


# ---------------------------------------------------------------------------
#  Class usage
# ---------------------------------------------------------------------------


class TestClassUsage:
    """Tests for miscellaneous Bndfun class usage."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        subinterval = Interval(-2, 3)

        def f(x):
            return sin(30 * x)

        self.ff = Bndfun.initfun_adaptive(f, subinterval)
        self.f = f
        self.xx = subinterval(np.linspace(-1, 1, 100))
        self.subinterval = subinterval

    def test_size(self):
        cfs = rng.random(10)
        b2 = Bndfun(Chebtech(cfs), Interval())
        assert b2.size == cfs.size

    def test_call_bary(self):
        self.ff(self.xx, "bary")
        self.ff(self.xx, how="bary")

    def test_call_clenshaw(self):
        self.ff(self.xx, "clenshaw")
        self.ff(self.xx, how="clenshaw")

    def test_call_bary_vs_clenshaw(self):
        b = self.ff(self.xx, "clenshaw")
        c = self.ff(self.xx, "bary")
        assert np.max(np.abs(b - c)) <= 2e2 * eps

    def test_call_raises(self):
        with pytest.raises(ValueError, match="notamethod"):
            self.ff(self.xx, "notamethod")
        with pytest.raises(ValueError, match="notamethod"):
            self.ff(self.xx, how="notamethod")

    def test_restrict(self):
        i1 = Interval(-1, 1)
        gg = self.ff.restrict(i1)
        yy = np.linspace(-1, 1, 1000)
        assert np.max(np.abs(self.ff(yy) - gg(yy))) <= 1e2 * eps

    def test_simplify(self):
        interval = Interval(-2, 1)
        ff = Bndfun.initfun_fixedlen(self.f, interval, 1000)
        gg = ff.simplify()
        assert gg.size == standard_chop(ff.onefun.coeffs)
        assert np.max(np.abs(ff.coeffs[: gg.size] - gg.coeffs)) == 0
        assert ff.interval == gg.interval

    def test_translate(self):
        c = -1
        shifted_interval = self.ff.interval + c
        gg = self.ff.translate(c)
        hh = Bndfun.initfun_fixedlen(lambda x: self.ff(x - c), shifted_interval, gg.size)
        yk = shifted_interval(np.linspace(-1, 1, 100))
        assert gg.interval == hh.interval
        assert np.max(np.abs(gg.coeffs - hh.coeffs)) <= 2e1 * eps
        assert np.max(np.abs(gg(yk) - hh(yk))) <= 1e2 * eps

    # vscale estimates
    _vscales = [
        (lambda x: sin(4 * pi * x), [-2, 2], 1),
        (lambda x: cos(x), [-10, 1], 1),
        (lambda x: cos(4 * pi * x), [-100, 100], 1),
        (lambda x: exp(cos(4 * pi * x)), [-1, 1], exp(1)),
        (lambda x: cos(3244 * x), [-2, 0], 1),
        (lambda x: exp(x), [-1, 2], exp(2)),
        (lambda x: 1e10 * exp(x), [-1, 1], 1e10 * exp(1)),
        (lambda x: 0 * x + 1.0, [-1e5, 1e4], 1),
    ]

    @pytest.mark.parametrize(("fun", "interval", "vscale"), _vscales)
    def test_vscale(self, fun, interval, vscale):
        subinterval = Interval(*interval)
        ff = Bndfun.initfun_adaptive(fun, subinterval)
        assert abs(ff.vscale - vscale) <= 0.1 * vscale


# ---------------------------------------------------------------------------
#  Calculus
# ---------------------------------------------------------------------------


class TestCalculus:
    """Tests for Bndfun calculus operations."""

    _def_integrals = [
        (lambda x: sin(x), [-2, 2], 0.0, 2 * eps),
        (lambda x: sin(4 * pi * x), [-0.1, 0.7], 0.088970317927147, 1e1 * eps),
        (lambda x: cos(x), [-100, 203], 0.426944059057085, 5e2 * eps),
        (lambda x: cos(4 * pi * x), [-1e-1, -1e-3], 0.074682699182803, 2 * eps),
        (lambda x: exp(cos(4 * pi * x)), [-3, 1], 5.064263511008033, 4 * eps),
        (lambda x: cos(3244 * x), [0, 0.4], -3.758628487169980e-05, 5e2 * eps),
        (lambda x: exp(x), [-2, -1], exp(-1) - exp(-2), 2 * eps),
        (lambda x: 1e10 * exp(x), [-1, 2], 1e10 * (exp(2) - exp(-1)), 2e10 * eps),
        (lambda x: 0 * x + 1.0, [-100, 300], 400, eps),
    ]

    @pytest.mark.parametrize(("fun", "interval", "integral", "tol"), _def_integrals)
    def test_sum(self, fun, interval, integral, tol):
        subinterval = Interval(*interval)
        ff = Bndfun.initfun_adaptive(fun, subinterval)
        assert abs(ff.sum() - integral) <= tol

    _indef_integrals = [
        (lambda x: 0 * x + 1.0, lambda x: x, [-2, 3], eps),
        (lambda x: x, lambda x: 1 / 2 * x**2, [-5, 0], 4 * eps),
        (lambda x: x**2, lambda x: 1 / 3 * x**3, [1, 10], 2e2 * eps),
        (lambda x: x**3, lambda x: 1 / 4 * x**4, [-1e-2, 4e-1], 2 * eps),
        (lambda x: x**4, lambda x: 1 / 5 * x**5, [-3, -2], 3e2 * eps),
        (lambda x: x**5, lambda x: 1 / 6 * x**6, [-1e-10, 1], 4 * eps),
        (lambda x: sin(x), lambda x: -cos(x), [-10, 22], 3e1 * eps),
        (lambda x: cos(3 * x), lambda x: 1.0 / 3 * sin(3 * x), [-3, 4], 2 * eps),
        (lambda x: exp(x), lambda x: exp(x), [-60, 1], 1e1 * eps),
        (lambda x: 1e10 * exp(x), lambda x: 1e10 * exp(x), [-1, 1], 1e10 * (3 * eps)),
    ]

    @pytest.mark.parametrize(("fun", "ifn", "interval", "tol"), _indef_integrals)
    def test_cumsum(self, fun, ifn, interval, tol):
        subinterval = Interval(*interval)
        ff = Bndfun.initfun_adaptive(fun, subinterval)
        gg = Bndfun.initfun_fixedlen(ifn, subinterval, ff.size + 1)
        coeffs = gg.coeffs
        coeffs[0] = coeffs[0] - ifn(np.array([interval[0]]))[0]
        assert np.max(ff.cumsum().coeffs - coeffs) <= tol

    _derivatives = [
        (lambda x: 0 * x + 1.0, lambda x: 0 * x + 0, [-2, 3], eps),
        (lambda x: x, lambda x: 0 * x + 1, [-5, 0], 2e1 * eps),
        (lambda x: x**2, lambda x: 2 * x, [1, 10], 2e2 * eps),
        (lambda x: x**3, lambda x: 3 * x**2, [-1e-2, 4e-1], 3 * eps),
        (lambda x: x**4, lambda x: 4 * x**3, [-3, -2], 1e3 * eps),
        (lambda x: x**5, lambda x: 5 * x**4, [-1e-10, 1], 4e1 * eps),
        (lambda x: sin(x), lambda x: cos(x), [-10, 22], 5e2 * eps),
        (lambda x: cos(3 * x), lambda x: -3 * sin(3 * x), [-3, 4], 5e2 * eps),
        (lambda x: exp(x), lambda x: exp(x), [-60, 1], 2e2 * eps),
        (lambda x: 1e10 * exp(x), lambda x: 1e10 * exp(x), [-1, 1], 1e10 * 2e2 * eps),
    ]

    @pytest.mark.parametrize(("fun", "der", "interval", "tol"), _derivatives)
    def test_diff(self, fun, der, interval, tol):
        subinterval = Interval(*interval)
        ff = Bndfun.initfun_adaptive(fun, subinterval)
        gg = Bndfun.initfun_fixedlen(der, subinterval, max(ff.size - 1, 1))
        assert np.max(ff.diff().coeffs - gg.coeffs) <= tol


# ---------------------------------------------------------------------------
#  Roots
# ---------------------------------------------------------------------------


class TestRoots:
    """Tests for Bndfun roots functionality."""

    _roots_test_params = [
        (lambda x: 3 * x + 2.0, [-2, 3], np.array([-2 / 3]), eps),
        (lambda x: x**2 + 0.2 * x - 0.08, [-2, 5], np.array([-0.4, 0.2]), 3e1 * eps),
        (lambda x: sin(x), [-7, 7], pi * np.linspace(-2, 2, 5), 1e1 * eps),
        (lambda x: cos(2 * pi * x), [-20, 10], np.linspace(-19.75, 9.75, 60), 3e1 * eps),
        (lambda x: sin(100 * pi * x), [-0.5, 0.5], np.linspace(-0.5, 0.5, 101), eps),
        (lambda x: sin(5 * pi / 2 * x), [-1, 1], np.array([-0.8, -0.4, 0, 0.4, 0.8]), eps),
    ]

    @pytest.mark.parametrize(("f", "interval", "roots_expected", "tol"), _roots_test_params)
    def test_roots(self, f, interval, roots_expected, tol):  # noqa: F811
        subinterval = Interval(*interval)
        ff = Bndfun.initfun_adaptive(f, subinterval)
        rts = ff.roots()
        assert np.max(np.abs(rts - roots_expected)) <= tol


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------


class TestPlotting:
    """Tests for Bndfun plotting methods."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        def f(x):
            return sin(1 * x) + 5e-1 * cos(10 * x) + 5e-3 * sin(100 * x)

        subinterval = Interval(-6, 10)
        self.f0 = Bndfun.initfun_fixedlen(f, subinterval, 1000)
        self.f1 = Bndfun.initfun_adaptive(f, subinterval)

    def test_plotcoeffs(self):
        _fig, ax = plt.subplots()
        self.f0.plotcoeffs(ax=ax)
        self.f1.plotcoeffs(ax=ax, color="r")


# ---------------------------------------------------------------------------
#  Ufuncs
# ---------------------------------------------------------------------------


class TestUfuncs:
    """Tests for Bndfun numpy ufunc overloads."""

    def test_ufunc(self):
        yy = np.linspace(-1, 1, 1000)
        for ufunc, f, interval in ufunc_parameter():
            subinterval = Interval(*interval)
            ff = Bndfun.initfun_adaptive(f, subinterval)

            def gg(x, ufunc=ufunc, f=f):
                return ufunc(f(x))

            gg_result = getattr(ff, ufunc.__name__)()
            xx = subinterval(yy)
            vscl = gg_result.vscale
            lscl = gg_result.size
            assert np.max(np.abs(gg(xx) - gg_result(xx))) <= vscl * lscl * eps
