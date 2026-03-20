"""Unit-tests for the chebtech module (src/chebpy/chebtech.py).

This single file consolidates every test that exercises the Chebtech class,
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
from chebpy.chebtech import Chebtech
from tests.generic.algebra import *  # noqa: F403
from tests.generic.class_usage import test_constfun_value  # noqa: F401
from tests.generic.complex import (  # noqa: F401
    test_calculus,
    test_complexfun_properties,
    test_real_imag,
    test_rho_ellipse_construction,
    test_roots,
)
from tests.generic.plotting import test_plot, test_plot_complex  # noqa: F401
from tests.generic.roots import rootstestfuns, test_empty  # noqa: F401
from tests.utilities import cos, eps, exp, pi, scaled_tol, sin

# Ensure reproducibility
rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
#  Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for Chebtech construction methods."""

    def test_initvalues(self):
        vals = rng.random(0)
        fun = Chebtech.initvalues(vals)
        cfs = Chebtech._vals2coeffs(vals)
        assert fun.coeffs.size == cfs.size == 0
        for n in range(1, 10):
            vals = rng.random(n)
            fun = Chebtech.initvalues(vals)
            cfs = Chebtech._vals2coeffs(vals)
            assert np.max(np.abs(fun.coeffs - cfs)) == 0.0

    def test_initidentity(self):
        x = Chebtech.initidentity()
        s = -1 + 2 * rng.random(10000)
        assert np.max(np.abs(s - x(s))) == 0.0

    def test_coeff_construction(self):
        coeffs = rng.random(10)
        f = Chebtech(coeffs)
        assert isinstance(f, Chebtech)
        assert np.max(np.abs(f.coeffs - coeffs)) < eps

    def test_adaptive_construction(self, testfunctions):
        for fun, funlen, _ in testfunctions:
            ff = Chebtech.initfun_adaptive(fun)
            assert ff.size - funlen <= 2
            assert ff.size - funlen > -1

    @pytest.mark.parametrize("n", [50, 500])
    def test_fixedlen_construction(self, testfunctions, n):
        for fun, _, _ in testfunctions:
            ff = Chebtech.initfun_fixedlen(fun, n)
            assert ff.size == n

    def test_initconst_with_non_scalar(self):
        with pytest.raises(ValueError, match=r"\[1"):
            Chebtech.initconst([1, 2, 3])
        with pytest.raises(ValueError, match=r"\[1"):
            Chebtech.initconst(np.array([1, 2, 3]))

    def test_imag_complex_chebtech(self):
        f = Chebtech.initfun_adaptive(lambda x: np.exp(1j * np.pi * x))
        assert f.iscomplex
        imag_f = f.imag()
        assert isinstance(imag_f, Chebtech)
        xx = np.linspace(-1, 1, 100)
        assert np.max(np.abs(imag_f(xx) - np.imag(np.exp(1j * np.pi * xx)))) < 1e-10

    def test_chebtech_with_nan_coefficients(self):
        nan_coeffs = np.array([1.0, np.nan, 2.0])
        f = Chebtech(nan_coeffs)
        xx = np.array([-1.0, 0.0, 1.0])
        result = f(xx)
        assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
#  Chebyshev points
# ---------------------------------------------------------------------------


class TestChebyshevPoints:
    """Tests for Chebyshev points functionality."""

    def test_chebpts_0(self):
        assert Chebtech._chebpts(0).size == 0

    def test_vals2coeffs_empty(self):
        assert Chebtech._vals2coeffs(np.array([])).size == 0

    def test_coeffs2vals_empty(self):
        assert Chebtech._coeffs2vals(np.array([])).size == 0

    def test_vals2coeffs_size1(self):
        for k in np.arange(10):
            fk = np.array([k])
            assert np.max(np.abs(Chebtech._vals2coeffs(fk) - fk)) <= eps

    def test_coeffs2vals_size1(self):
        for k in np.arange(10):
            ak = np.array([k])
            assert np.max(np.abs(Chebtech._coeffs2vals(ak) - ak)) <= eps

    _test_sizes = 2 ** np.arange(2, 18, 2) + 1

    @pytest.mark.parametrize("n", _test_sizes)
    def test_vals2coeffs2vals(self, n):
        values = rng.random(n)
        coeffs = Chebtech._vals2coeffs(values)
        _values_ = Chebtech._coeffs2vals(coeffs)
        assert np.max(np.abs(values - _values_)) <= scaled_tol(n)

    @pytest.mark.parametrize("n", _test_sizes)
    def test_coeffs2vals2coeffs(self, n):
        coeffs = rng.random(n)
        values = Chebtech._coeffs2vals(coeffs)
        _coeffs_ = Chebtech._vals2coeffs(values)
        assert np.max(np.abs(coeffs - _coeffs_)) <= scaled_tol(n)

    _chebpts2_testlist = [
        (1, np.array([0.0]), eps),
        (2, np.array([-1.0, 1.0]), eps),
        (3, np.array([-1.0, 0.0, 1.0]), eps),
        (4, np.array([-1.0, -0.5, 0.5, 1.0]), 2 * eps),
        (5, np.array([-1.0, -(2.0 ** (-0.5)), 0.0, 2.0 ** (-0.5), 1.0]), eps),
    ]

    @pytest.mark.parametrize(("n", "expected", "tol"), _chebpts2_testlist)
    def test_chebpts_values(self, n, expected, tol):
        actual = Chebtech._chebpts(n)
        assert np.max(np.abs(actual - expected)) <= tol

    _chebpts_len_sizes = 2 ** np.arange(2, 18, 2) + 3

    @pytest.mark.parametrize("k", _chebpts_len_sizes)
    def test_chebpts_properties(self, k):
        pts = Chebtech._chebpts(k)
        assert pts.size == k
        assert pts[0] == -1.0
        assert pts[-1] == 1.0
        assert np.all(np.diff(pts) > 0)


# ---------------------------------------------------------------------------
#  Class usage
# ---------------------------------------------------------------------------


class TestClassUsage:
    """Tests for miscellaneous Chebtech class usage."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.ff = Chebtech.initfun_fixedlen(lambda x: np.sin(30 * x), 100)
        self.xx = -1 + 2 * rng.random(100)

    def test_size(self):
        cfs = rng.random(10)
        assert Chebtech(np.array([])).size == 0
        assert Chebtech(np.array([1.0])).size == 1
        assert Chebtech(cfs).size == cfs.size

    def test_call(self):
        self.ff(self.xx)

    def test_call_bary(self):
        self.ff(self.xx, "bary")
        self.ff(self.xx, how="bary")

    def test_call_clenshaw(self):
        self.ff(self.xx, "clenshaw")
        self.ff(self.xx, how="clenshaw")

    def test_call_bary_vs_clenshaw(self):
        b = self.ff(self.xx, "clenshaw")
        c = self.ff(self.xx, "bary")
        assert np.max(np.abs(b - c)) <= 5e1 * eps

    def test_call_raises(self):
        with pytest.raises(ValueError, match="notamethod"):
            self.ff(self.xx, "notamethod")
        with pytest.raises(ValueError, match="notamethod"):
            self.ff(self.xx, how="notamethod")

    def test_prolong(self):
        for k in [0, 1, 20, self.ff.size, 200]:
            assert self.ff.prolong(k).size == k

    def test_copy(self):
        gg = self.ff.copy()
        assert self.ff == self.ff
        assert gg == gg
        assert self.ff != gg
        assert np.max(np.abs(self.ff.coeffs - gg.coeffs)) == 0

    def test_simplify(self):
        gg = self.ff.simplify()
        assert gg.size == standard_chop(self.ff.coeffs)
        assert np.max(np.abs(self.ff.coeffs[: gg.size] - gg.coeffs)) == 0
        fcfs = self.ff.coeffs
        gcfs = gg.coeffs
        assert (fcfs[: gg.size] - gcfs).sum() == 0
        gg.coeffs[0] = 1
        assert (fcfs[: gg.size] - gcfs).sum() != 0

    _vscales = [
        (lambda x: sin(4 * pi * x), 40, 1),
        (lambda x: cos(x), 15, 1),
        (lambda x: cos(4 * pi * x), 39, 1),
        (lambda x: exp(cos(4 * pi * x)), 181, exp(1)),
        (lambda x: cos(3244 * x), 3389, 1),
        (lambda x: exp(x), 15, exp(1)),
        (lambda x: 1e10 * exp(x), 15, 1e10 * exp(1)),
        (lambda x: 0 * x + 1.0, 1, 1),
    ]

    @pytest.mark.parametrize(("fun", "n", "vscale"), _vscales)
    def test_vscale(self, fun, n, vscale):
        ff = Chebtech.initfun_fixedlen(fun, n)
        assert abs(ff.vscale - vscale) <= vscale


# ---------------------------------------------------------------------------
#  Calculus
# ---------------------------------------------------------------------------


class TestCalculus:
    """Tests for Chebtech calculus operations."""

    _def_integrals = [
        (lambda x: sin(x), 14, 0.0, eps),
        (lambda x: sin(4 * pi * x), 40, 0.0, 1e1 * eps),
        (lambda x: cos(x), 15, 1.682941969615793, 2 * eps),
        (lambda x: cos(4 * pi * x), 39, 0.0, 2 * eps),
        (lambda x: exp(cos(4 * pi * x)), 182, 2.532131755504016, 4 * eps),
        (lambda x: cos(3244 * x), 3389, 5.879599674161602e-04, 5e2 * eps),
        (lambda x: exp(x), 15, exp(1) - exp(-1), 2 * eps),
        (lambda x: 1e10 * exp(x), 15, 1e10 * (exp(1) - exp(-1)), 4e10 * eps),
        (lambda x: 0 * x + 1.0, 1, 2, eps),
    ]

    @pytest.mark.parametrize(("fun", "n", "integral", "tol"), _def_integrals)
    def test_definite_integral(self, fun, n, integral, tol):
        ff = Chebtech.initfun_fixedlen(fun, n)
        assert abs(ff.sum() - integral) <= tol

    _indef_integrals = [
        (lambda x: sin(x), lambda x: -cos(x), 15, 2 * eps),
        (lambda x: cos(x), lambda x: sin(x), 15, 2 * eps),
        (lambda x: exp(x), lambda x: exp(x), 15, 5 * eps),
        (lambda x: x**3, lambda x: 0.25 * x**4, 16, 10 * eps),
        (lambda x: 0 * x + 1, lambda x: x, 1, 3.0),
        (lambda x: 0 * x, lambda x: 0 * x, 1, eps),
    ]

    @pytest.mark.parametrize(("fun", "dfn", "n", "tol"), _indef_integrals)
    def test_indefinite_integral(self, fun, dfn, n, tol):
        ff = Chebtech.initfun_fixedlen(fun, n)
        gg = Chebtech.initfun_fixedlen(dfn, n)
        xx = np.linspace(-1, 1, 1000)
        assert np.max(np.abs(ff.cumsum()(xx) - (gg(xx) - gg(-1)))) <= 100 * tol

    _derivatives = [
        (lambda x: sin(x), lambda x: cos(x), 15, 30 * eps),
        (lambda x: cos(x), lambda x: -sin(x), 15, 30 * eps),
        (lambda x: exp(x), lambda x: exp(x), 15, 200 * eps),
        (lambda x: x**3, lambda x: 3 * x**2, 16, 30 * eps),
        (lambda x: 0 * x + 1, lambda x: 0 * x, 1, eps),
        (lambda x: 0 * x, lambda x: 0 * x, 1, eps),
    ]

    @pytest.mark.parametrize(("fun", "der", "n", "tol"), _derivatives)
    def test_derivative(self, fun, der, n, tol):
        ff = Chebtech.initfun_fixedlen(fun, n)
        gg = Chebtech.initfun_fixedlen(der, n)
        xx = np.linspace(-1, 1, 1000)
        assert np.max(np.abs(ff.diff()(xx) - gg(xx))) <= 10 * tol


# ---------------------------------------------------------------------------
#  Roots
# ---------------------------------------------------------------------------


class TestRoots:
    """Tests for Chebtech roots functionality."""

    @pytest.mark.parametrize(("f", "roots"), rootstestfuns)
    def test_roots(self, f, roots):  # noqa: F811
        ff = Chebtech.initfun_adaptive(f)
        rts = ff.roots()
        assert np.max(np.abs(rts - roots)) <= eps


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------


class TestPlotting:
    """Tests for Chebtech plotting methods."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        def f(x):
            return sin(3 * x) + 5e-1 * cos(30 * x)

        self.f0 = Chebtech.initfun_fixedlen(f, 100)
        self.f1 = Chebtech.initfun_adaptive(f)
        self.f2 = Chebtech.initfun_adaptive(lambda x: np.exp(2 * np.pi * 1j * x))

    def test_plotcoeffs(self):
        _fig, ax = plt.subplots()
        self.f0.plotcoeffs(ax=ax)
        self.f1.plotcoeffs(ax=ax, color="r")
