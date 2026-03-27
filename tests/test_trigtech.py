"""Unit tests for the Trigtech module (src/chebpy/trigtech.py).

Tests cover construction, evaluation, algebra, calculus, root-finding, plotting,
the `trigfun` public API, and integration with Bndfun/Chebfun via the `techdict`
mechanism.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebpy.settings import DefaultPreferences
from chebpy.trigtech import Trigtech
from chebpy.utilities import Interval

mpl.use("Agg")

rng = np.random.default_rng(42)

eps = DefaultPreferences.eps
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tol(n: int, scale: float = 1.0) -> float:
    return scale * max(50 * eps, np.log(n + 1) ** 2 * eps)


def _linspace(a: float = -1.0, b: float = 1.0, n: int = 500) -> np.ndarray:
    # Avoid endpoints to stay away from wrap-around artefacts
    return np.linspace(a + 0.01, b - 0.01, n)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_initempty(self):
        f = Trigtech.initempty()
        assert f.isempty
        assert f.size == 0

    def test_initconst_float(self):
        f = Trigtech.initconst(3.0)
        assert f.isconst
        assert not f.isempty
        assert abs(f(0.0) - 3.0) < eps

    def test_initconst_int(self):
        f = Trigtech.initconst(2)
        assert isinstance(f(0.0), float)

    def test_initconst_non_scalar_raises(self):
        with pytest.raises(ValueError):
            Trigtech.initconst([1, 2, 3])
        with pytest.raises(ValueError):
            Trigtech.initconst(np.array([1, 2, 3]))

    def test_initvalues_roundtrip(self):
        n = 32
        vals = rng.random(n)
        f = Trigtech.initvalues(vals)
        assert np.max(np.abs(f.values() - vals)) < _tol(n)

    def test_initvalues_empty(self):
        f = Trigtech.initvalues(np.array([]))
        assert f.isempty

    def test_coeff_construction(self):
        n = 16
        coeffs = Trigtech._vals2coeffs(rng.random(n))
        f = Trigtech(coeffs)
        assert isinstance(f, Trigtech)
        assert f.size == n

    def test_adaptive_cos(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        assert f.size == 3  # c_{-1}, c_0, c_{+1}
        x = _linspace()
        assert np.max(np.abs(f(x) - cos(pi * x))) < 10 * eps

    def test_adaptive_sin(self):
        f = Trigtech.initfun_adaptive(lambda x: sin(2 * pi * x))
        assert f.size == 5
        x = _linspace()
        assert np.max(np.abs(f(x) - sin(2 * pi * x))) < 10 * eps

    def test_adaptive_exp_cos(self):
        f = Trigtech.initfun_adaptive(lambda x: exp(cos(2 * pi * x)))
        assert f.size >= 17
        x = _linspace()
        assert np.max(np.abs(f(x) - exp(cos(2 * pi * x)))) < 1e-12

    @pytest.mark.parametrize("n", [16, 64, 256])
    def test_fixedlen_construction(self, n):
        f = Trigtech.initfun_fixedlen(lambda x: sin(pi * x), n)
        assert f.size == n

    def test_fixedlen_none_raises(self):
        with pytest.raises(ValueError, match="n parameter"):
            Trigtech.initfun_fixedlen(sin, n=None)

    def test_initfun_delegates_adaptive(self):
        f = Trigtech.initfun(lambda x: cos(pi * x))
        assert f.size == 3

    def test_initfun_delegates_fixedlen(self):
        f = Trigtech.initfun(lambda x: cos(pi * x), n=32)
        assert f.size == 32


# ---------------------------------------------------------------------------
# FFT helpers
# ---------------------------------------------------------------------------


class TestFFTHelpers:
    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
    def test_vals2coeffs_round_trip(self, n):
        vals = rng.random(n)
        coeffs = Trigtech._vals2coeffs(vals)
        vals_back = Trigtech._coeffs2vals(coeffs)
        assert np.max(np.abs(vals - vals_back)) < _tol(n)

    def test_vals2coeffs_empty(self):
        c = Trigtech._vals2coeffs(np.array([]))
        assert c.size == 0

    def test_coeffs2vals_empty(self):
        v = Trigtech._coeffs2vals(np.array([]))
        assert v.size == 0

    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_trigpts_properties(self, n):
        pts = Trigtech._trigpts(n)
        assert pts.size == n
        assert pts[0] == pytest.approx(-1.0)
        assert np.all(np.diff(pts) > 0)
        # Last point is strictly < 1 (open right endpoint)
        assert pts[-1] < 1.0

    def test_trigpts_empty(self):
        assert Trigtech._trigpts(0).size == 0


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_iscomplex_real_function(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        assert not f.iscomplex

    def test_iscomplex_sine_function(self):
        f = Trigtech.initfun_adaptive(lambda x: sin(pi * x))
        assert not f.iscomplex

    def test_iscomplex_complex_function(self):
        f = Trigtech.initfun_adaptive(lambda x: exp(1j * pi * x))
        assert f.iscomplex

    def test_isperiodic(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        assert f.isperiodic

    def test_vscale_cos(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        assert abs(f.vscale - 1.0) < 10 * eps

    def test_vscale_empty(self):
        assert Trigtech.initempty().vscale == 0.0

    def test_vscale_scaled(self):
        scale = 1e5
        f = Trigtech.initfun_adaptive(lambda x: scale * cos(pi * x))
        assert abs(f.vscale - scale) < 10 * eps * scale


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_eval_at_grid_points(self):
        """Evaluation at the original sample points should be exact."""
        n = 16
        f = Trigtech.initfun_fixedlen(lambda x: cos(4 * pi * x), n)
        pts = Trigtech._trigpts(n)
        vals = f(pts)
        expected = cos(4 * pi * pts)
        assert np.max(np.abs(vals - expected)) < _tol(n)

    def test_eval_scalar(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        assert isinstance(f(0.0), float)
        assert abs(f(0.0) - 1.0) < 10 * eps

    def test_eval_array(self):
        f = Trigtech.initfun_adaptive(lambda x: sin(2 * pi * x))
        x = np.array([0.0, 0.25, -0.25])
        out = f(x)
        assert out.shape == (3,)
        assert np.max(np.abs(out - sin(2 * pi * x))) < 10 * eps

    def test_eval_empty(self):
        f = Trigtech.initempty()
        out = f(np.array([0.0, 0.5]))
        assert out.size == 0

    def test_eval_const(self):
        c = 7.5
        f = Trigtech.initconst(c)
        x = _linspace()
        assert np.max(np.abs(f(x) - c)) < eps

    def test_eval_how_ignored(self):
        """The 'how' parameter is accepted but does not change results."""
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        x = _linspace()
        assert np.max(np.abs(f(x, "fft") - f(x))) < eps

    def test_eval_complex(self):
        f = Trigtech.initfun_adaptive(lambda x: exp(1j * pi * x))
        out = f(np.array([0.0]))
        assert np.iscomplex(out[0]) or abs(out[0].imag) < 1e-10


# ---------------------------------------------------------------------------
# Algebra
# ---------------------------------------------------------------------------


class TestAlgebra:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.xx = _linspace()
        self.f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        self.g = Trigtech.initfun_adaptive(lambda x: sin(2 * pi * x))

    @pytest.mark.parametrize("c", [-2.0, 0.0, 1.0, 1e5])
    def test_add_scalar(self, c):
        h = self.f + c
        tol = 50 * eps * max(abs(c), 1.0)
        assert np.max(np.abs(h(self.xx) - (cos(pi * self.xx) + c))) < tol

    def test_add_trigtech(self):
        h = self.f + self.g
        expected = cos(pi * self.xx) + sin(2 * pi * self.xx)
        assert np.max(np.abs(h(self.xx) - expected)) < 10 * eps

    def test_radd_scalar(self):
        h = 3.0 + self.f
        assert np.max(np.abs(h(self.xx) - (3.0 + cos(pi * self.xx)))) < 50 * eps

    def test_sub(self):
        h = self.f - self.g
        expected = cos(pi * self.xx) - sin(2 * pi * self.xx)
        assert np.max(np.abs(h(self.xx) - expected)) < 10 * eps

    def test_neg(self):
        h = -self.f
        assert np.max(np.abs(h(self.xx) + cos(pi * self.xx))) < 10 * eps

    def test_pos(self):
        assert self.f is +self.f

    @pytest.mark.parametrize("c", [-2.0, 0.5, 3.0])
    def test_mul_scalar(self, c):
        h = self.f * c
        assert np.max(np.abs(h(self.xx) - c * cos(pi * self.xx))) < 50 * eps * abs(c)

    def test_mul_trigtech(self):
        h = self.f * self.f
        expected = cos(pi * self.xx) ** 2
        assert np.max(np.abs(h(self.xx) - expected)) < 50 * eps

    @pytest.mark.parametrize("c", [2.0, 0.5])
    def test_div_scalar(self, c):
        h = self.f / c
        assert np.max(np.abs(h(self.xx) - cos(pi * self.xx) / c)) < 50 * eps

    def test_pow_int(self):
        h = self.f**2
        expected = cos(pi * self.xx) ** 2
        assert np.max(np.abs(h(self.xx) - expected)) < 50 * eps

    def test_add_empty(self):
        empty = Trigtech.initempty()
        result = self.f + empty
        assert result.isempty

    def test_sub_empty(self):
        empty = Trigtech.initempty()
        result = self.f - empty
        assert result.isempty


# ---------------------------------------------------------------------------
# Calculus
# ---------------------------------------------------------------------------


class TestCalculus:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.xx = _linspace()

    @pytest.mark.parametrize(
        ("fun", "expected_integral"),
        [
            (lambda x: cos(pi * x), 0.0),  # ∫cos(πx) = 0
            (lambda x: sin(2 * pi * x), 0.0),  # ∫sin(2πx) = 0
            (lambda x: 0 * x + 1.0, 2.0),  # ∫1 dx = 2
        ],
    )
    def test_definite_integral(self, fun, expected_integral):
        f = Trigtech.initfun_adaptive(fun)
        assert abs(f.sum() - expected_integral) < 100 * eps

    def test_sum_empty(self):
        assert Trigtech.initempty().sum() == 0.0

    def test_derivative_cos(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        df = f.diff()
        expected = -pi * sin(pi * self.xx)
        assert np.max(np.abs(df(self.xx) - expected)) < 100 * eps

    def test_derivative_sin(self):
        f = Trigtech.initfun_adaptive(lambda x: sin(2 * pi * x))
        df = f.diff()
        expected = 2 * pi * cos(2 * pi * self.xx)
        assert np.max(np.abs(df(self.xx) - expected)) < 100 * eps

    def test_derivative_const(self):
        f = Trigtech.initconst(3.0)
        df = f.diff()
        assert np.max(np.abs(df(self.xx))) < eps

    def test_derivative_empty(self):
        f = Trigtech.initempty()
        assert f.diff().isempty

    def test_cumsum_cos(self):
        """∫_{-1}^{x} cos(πt) dt = sin(πx)/π."""
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        cf = f.cumsum()
        assert abs(cf(-1.0)) < 1e-14
        x = self.xx
        expected = sin(pi * x) / pi
        assert np.max(np.abs(cf(x) - expected)) < 100 * eps

    def test_cumsum_sin(self):
        """∫_{-1}^{x} sin(2πt) dt = (-cos(2πx) + cos(-2π)) / (2π)."""
        f = Trigtech.initfun_adaptive(lambda x: sin(2 * pi * x))
        cf = f.cumsum()
        assert abs(cf(-1.0)) < 1e-14
        x = self.xx
        expected = (-cos(2 * pi * x) + cos(-2 * pi)) / (2 * pi)
        assert np.max(np.abs(cf(x) - expected)) < 100 * eps

    def test_cumsum_empty(self):
        f = Trigtech.initempty()
        assert f.cumsum().isempty

    def test_diff_cumsum_roundtrip(self):
        """d/dx ∫_{-1}^{x} f ≈ f (up to machine precision)."""
        f = Trigtech.initfun_adaptive(lambda x: sin(2 * pi * x))
        h = f.cumsum().diff()
        assert np.max(np.abs(h(self.xx) - f(self.xx))) < 1e-10


# ---------------------------------------------------------------------------
# Root-finding
# ---------------------------------------------------------------------------


class TestRoots:
    def test_roots_cos(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        rts = f.roots()
        expected = np.array([-0.5, 0.5])
        assert rts.size == 2
        assert np.max(np.abs(np.sort(rts) - expected)) < 1e-10

    def test_roots_sin(self):
        f = Trigtech.initfun_adaptive(lambda x: sin(pi * x))
        rts = f.roots()
        expected = np.array([-1.0, 0.0, 1.0])
        assert rts.size >= 2  # at least the interior roots
        assert any(np.abs(rts - e).min() < 1e-10 for e in expected[1:-1])

    def test_roots_empty(self):
        f = Trigtech.initempty()
        assert f.roots().size == 0

    def test_roots_sort_kwarg(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        rts = f.roots(sort=True)
        assert np.all(np.diff(rts) >= 0)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_copy(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        g = f.copy()
        assert f is not g
        assert np.max(np.abs(f.coeffs - g.coeffs)) == 0.0
        # Mutating the copy doesn't affect the original
        g._coeffs[0] += 1.0
        assert np.max(np.abs(f.coeffs - g.coeffs)) != 0.0

    def test_real_of_real(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        assert f.real() is f

    def test_real_of_complex(self):
        f = Trigtech.initfun_adaptive(lambda x: exp(1j * pi * x))
        gr = f.real()
        x = _linspace()
        assert np.max(np.abs(gr(x) - np.real(exp(1j * pi * x)))) < 10 * eps

    def test_imag_of_real(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        gi = f.imag()
        assert gi.isconst
        x = _linspace()
        assert np.max(np.abs(gi(x))) < eps

    def test_imag_of_complex(self):
        f = Trigtech.initfun_adaptive(lambda x: exp(1j * pi * x))
        gi = f.imag()
        x = _linspace()
        assert np.max(np.abs(gi(x) - np.imag(exp(1j * pi * x)))) < 10 * eps

    def test_prolong_increase(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        g = f.prolong(32)
        assert g.size == 32
        x = _linspace()
        assert np.max(np.abs(g(x) - f(x))) < 10 * eps

    def test_prolong_decrease(self):
        f = Trigtech.initfun_fixedlen(lambda x: cos(pi * x), 32)
        g = f.prolong(3)
        assert g.size == 3

    def test_prolong_same(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        g = f.prolong(f.size)
        assert g is not f
        assert g.size == f.size

    def test_simplify(self):
        f = Trigtech.initfun_fixedlen(lambda x: cos(pi * x), 64)
        g = f.simplify()
        assert g.size <= f.size
        x = _linspace()
        assert np.max(np.abs(g(x) - f(x))) < 1e-12

    def test_values(self):
        n = 16
        f = Trigtech.initfun_fixedlen(lambda x: cos(4 * pi * x), n)
        pts = Trigtech._trigpts(n)
        assert np.max(np.abs(f.values() - cos(4 * pi * pts))) < _tol(n)

    def test_coeffs_to_plotorder(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        dc_centered = f._coeffs_to_plotorder()
        assert np.max(np.abs(dc_centered - np.fft.fftshift(f.coeffs))) == 0.0
        # DC should be in the middle
        assert abs(dc_centered[len(dc_centered) // 2]) < 10 * eps  # DC ≈ 0 for cos


# ---------------------------------------------------------------------------
# Integration with classicfun / trigfun API
# ---------------------------------------------------------------------------


class TestClassicfunIntegration:
    def test_bndfun_with_trigtech(self):
        """Bndfun with Trigtech technology should function correctly."""
        from chebpy.bndfun import Bndfun
        from chebpy.settings import _preferences as prefs

        with prefs:
            prefs.tech = "Trigtech"
            interval = Interval(-1.0, 1.0)
            bf = Bndfun.initfun_adaptive(lambda x: sin(2 * pi * x), interval)
            x = np.linspace(-0.9, 0.9, 50)
            assert np.max(np.abs(bf(x) - sin(2 * pi * x))) < 1e-12

    def test_trigfun_api(self):
        from chebpy import trigfun

        f = trigfun(lambda x: cos(pi * x), [-1, 1])
        from chebpy.chebfun import Chebfun

        assert isinstance(f, Chebfun)
        x = np.linspace(-0.9, 0.9, 50)
        assert np.max(np.abs(f(x) - cos(pi * x))) < 1e-12

    def test_trigfun_empty(self):
        from chebpy import trigfun

        f = trigfun()
        from chebpy.chebfun import Chebfun

        assert isinstance(f, Chebfun)
        assert f.isempty

    def test_trigfun_constant(self):
        from chebpy import trigfun

        f = trigfun(3.14)
        assert abs(f(0.0) - 3.14) < eps

    def test_trigfun_fixedlen(self):
        from chebpy import trigfun

        f = trigfun(lambda x: sin(pi * x), n=16)
        from chebpy.chebfun import Chebfun

        assert isinstance(f, Chebfun)

    def test_trigtech_exported_from_package(self):
        import chebpy

        assert hasattr(chebpy, "Trigtech")
        assert chebpy.Trigtech is Trigtech

    def test_trigfun_exported_from_package(self):
        import chebpy

        assert hasattr(chebpy, "trigfun")


# ---------------------------------------------------------------------------
# Plotting (smoke tests only, no visual assertions)
# ---------------------------------------------------------------------------


class TestPlotting:
    def test_plot(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        fig, ax = plt.subplots()
        f.plot(ax=ax)
        plt.close(fig)

    def test_plotcoeffs(self):
        f = Trigtech.initfun_adaptive(lambda x: cos(pi * x))
        fig, ax = plt.subplots()
        f.plotcoeffs(ax=ax)
        plt.close(fig)
