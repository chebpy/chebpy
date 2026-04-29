"""Unit tests for :class:`chebpy.singfun.Singfun` (Phase 3, v1)."""

import numpy as np
import pytest

from chebpy.maps import DoubleSlitMap, SingleSlitMap
from chebpy.singfun import Singfun


# ---------------------------------------------------------------------------
#  construction
# ---------------------------------------------------------------------------
class TestConstruction:
    """Construction from sample functions, constants, identity, and empty."""

    def test_initempty(self):
        """Empty :class:`Singfun` round-trips through the empty :class:`Onefun` API."""
        f = Singfun.initempty()
        assert f.isempty

    def test_initconst(self):
        """A constant :class:`Singfun` evaluates to the constant everywhere on ``[a, b]``."""
        f = Singfun.initconst(2.5, [0.0, 1.0], sing="left")
        x = np.linspace(0.0, 1.0, 11)
        assert np.allclose(f(x), 2.5)

    def test_initidentity(self):
        """The identity :class:`Singfun` matches ``f(x) = x`` to machine precision."""
        f = Singfun.initidentity([0.0, 1.0], sing="left")
        x = np.linspace(0.05, 0.95, 19)
        assert np.allclose(f(x), x, atol=1e-12)

    def test_initfun_adaptive_resolves_sqrt(self):
        """``sqrt(x)`` on ``[0, 1]`` is resolved to machine precision under ``sing='left'``."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.0)
        x = np.linspace(0.001, 0.999, 41)
        assert np.allclose(f(x), np.sqrt(x), atol=1e-14)

    def test_initfun_adaptive_resolves_sqrt_right(self):
        """``sqrt(1 - x)`` on ``[0, 1]`` is resolved under ``sing='right'``.

        Note: pointwise accuracy is bounded by ``ulp(x) / sqrt(1-x)`` at the
        clustered samples near ``x=1``, giving a floor around ``1e-10``.
        """
        g = Singfun.initfun_adaptive(lambda x: np.sqrt(1.0 - x), [0.0, 1.0], sing="right", alpha=1.0)
        x = np.linspace(0.001, 0.999, 41)
        assert np.allclose(g(x), np.sqrt(1.0 - x), atol=1e-10)

    def test_initfun_adaptive_resolves_two_sided_singularity(self):
        """``sqrt(x(1-x))`` resolves under the symmetric double-slit map.

        Note: the right-endpoint is ulp-limited like ``sqrt(1-x)``; pointwise
        accuracy floors around ``1e-10``.
        """
        f = Singfun.initfun_adaptive(
            lambda x: np.sqrt(x * (1.0 - x)),
            [0.0, 1.0],
            sing="both",
            alpha=1.0,
        )
        x = np.linspace(0.01, 0.99, 41)
        assert np.allclose(f(x), np.sqrt(x * (1.0 - x)), atol=1e-10)

    def test_initfun_fixedlen(self):
        """Fixed-length construction respects the requested coefficient count."""
        n = 32
        f = Singfun.initfun_fixedlen(np.sqrt, [0.0, 1.0], n, sing="left")
        assert f.size == n

    def test_invalid_sing_raises(self):
        """An unrecognised ``sing`` keyword raises :class:`ValueError`."""
        with pytest.raises(ValueError):
            Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="middle")


# ---------------------------------------------------------------------------
#  map property and inherited behaviour
# ---------------------------------------------------------------------------
class TestMapProperty:
    """The ``map`` override is the conduit for non-affine evaluation and rootfinding."""

    def test_map_is_single_slit(self):
        """``sing='left'`` produces a :class:`SingleSlitMap` with matching parameters."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.5)
        assert isinstance(f.map, SingleSlitMap)
        assert f.map.alpha == 1.5
        assert f.map.side == "left"

    def test_map_is_double_slit(self):
        """``sing='both'`` produces a :class:`DoubleSlitMap`."""
        f = Singfun.initconst(0.0, [0.0, 1.0], sing="both", alpha=2.0)
        assert isinstance(f.map, DoubleSlitMap)
        assert f.map.alpha == 2.0

    def test_evaluation_at_endpoints(self):
        """``__call__`` at the support endpoints returns the limiting values."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.0)
        # sqrt(0) = 0 (clustered endpoint -> Onefun(t=-1) ~= 0)
        assert abs(float(f(0.0))) < 1e-14
        # sqrt(1) = 1 (smooth endpoint)
        assert abs(float(f(1.0)) - 1.0) < 1e-14

    def test_support(self):
        """``support`` returns the logical interval as ``(a, b)``."""
        f = Singfun.initconst(1.0, [2.0, 5.0], sing="left")
        a, b = f.support
        assert a == pytest.approx(2.0)
        assert b == pytest.approx(5.0)


# ---------------------------------------------------------------------------
#  calculus
# ---------------------------------------------------------------------------
class TestCalculus:
    """Definite and indefinite integration via the change-of-variables Jacobian."""

    def test_sum_sqrt(self):
        r"""``\int_0^1 \sqrt{x}\,dx = 2/3``."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.0)
        assert float(f.sum()) == pytest.approx(2.0 / 3.0, abs=1e-14)

    def test_sum_two_sided(self):
        r"""``\int_0^1 \sqrt{x(1-x)}\,dx = \pi/8``.

        The integral converges to machine precision even though pointwise
        evaluation is ulp-limited near ``x=1``: the noisy samples are
        weighted by ``m'(t) -> 0`` at the clustered endpoint.
        """
        f = Singfun.initfun_adaptive(
            lambda x: np.sqrt(x * (1.0 - x)),
            [0.0, 1.0],
            sing="both",
            alpha=1.0,
        )
        assert float(f.sum()) == pytest.approx(np.pi / 8.0, abs=1e-13)

    def test_sum_constant(self):
        """The integral of a constant over ``[a, b]`` is ``c * (b - a)``."""
        f = Singfun.initconst(3.0, [1.0, 4.0], sing="left")
        assert float(f.sum()) == pytest.approx(9.0, abs=1e-12)

    def test_sum_empty_returns_zero(self):
        """The integral of an empty :class:`Singfun` is ``0``."""
        f = Singfun.initempty()
        assert float(f.sum()) == 0.0

    def test_cumsum_endpoint_value(self):
        r"""``cumsum(sqrt)(1) = \int_0^1 \sqrt{x}\,dx = 2/3``."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.0)
        F = f.cumsum()
        assert float(F(1.0)) == pytest.approx(2.0 / 3.0, abs=1e-14)

    def test_cumsum_left_endpoint_is_zero(self):
        """``cumsum(f)(a) = 0`` (constant of integration chosen so ``F(a) = 0``)."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.0)
        F = f.cumsum()
        assert abs(float(F(0.0))) < 1e-14

    def test_cumsum_interior_values(self):
        r"""``cumsum(\sqrt{\cdot})(x) = (2/3) x^{3/2}`` on the interior."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.0)
        F = f.cumsum()
        x = np.linspace(0.05, 0.95, 19)
        expected = (2.0 / 3.0) * x**1.5
        assert np.allclose(F(x), expected, atol=1e-14)

    def test_diff_raises_not_implemented(self):
        """``diff`` is deferred to a later phase and raises :class:`NotImplementedError`."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left")
        with pytest.raises(NotImplementedError):
            f.diff()


# ---------------------------------------------------------------------------
#  arithmetic and rebuild
# ---------------------------------------------------------------------------
class TestArithmetic:
    """Same-map arithmetic preserves the :class:`Singfun` representation."""

    def test_add_same_map(self):
        """Adding two :class:`Singfun` instances with the same map gives a :class:`Singfun`."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.0)
        g = Singfun.initfun_adaptive(lambda x: 2.0 * np.sqrt(x), [0.0, 1.0], sing="left", alpha=1.0)
        h = f + g
        assert isinstance(h, Singfun)
        x = np.linspace(0.01, 0.99, 21)
        assert np.allclose(h(x), 3.0 * np.sqrt(x), atol=1e-9)

    def test_scalar_multiply(self):
        """Scalar multiplication preserves the :class:`Singfun` representation."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.0)
        h = 2.0 * f
        assert isinstance(h, Singfun)
        x = np.linspace(0.01, 0.99, 21)
        assert np.allclose(h(x), 2.0 * np.sqrt(x), atol=1e-9)

    def test_negate(self):
        """Unary negation preserves the :class:`Singfun` representation."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.0)
        h = -f
        assert isinstance(h, Singfun)
        x = np.linspace(0.01, 0.99, 21)
        assert np.allclose(h(x), -np.sqrt(x), atol=1e-9)


# ---------------------------------------------------------------------------
#  utility methods
# ---------------------------------------------------------------------------
class TestUtilities:
    """Translation, restriction, and the explicit ``to_bndfun`` opt-in."""

    def test_translate(self):
        """Translation shifts the support and rebuilds the map for the new endpoints."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.0)
        g = f.translate(2.0)
        assert g.support[0] == pytest.approx(2.0)
        assert g.support[1] == pytest.approx(3.0)
        x = np.linspace(2.001, 2.999, 21)
        assert np.allclose(g(x), np.sqrt(x - 2.0), atol=1e-9)

    def test_restrict_identity(self):
        """Restricting to the same support is a no-op."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left")
        g = f.restrict(f.interval)
        assert g is f

    def test_to_bndfun(self):
        """``to_bndfun`` returns a :class:`~chebpy.bndfun.Bndfun` representation.

        Note: the :class:`Bndfun` adaptive constructor will not converge to
        machine precision for a true endpoint branch singularity; we only
        check that the resulting object has the right type and matches in
        the smooth interior.
        """
        from chebpy.bndfun import Bndfun

        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", alpha=1.0)
        with pytest.warns(UserWarning, match="did not converge"):
            b = f.to_bndfun()
        assert isinstance(b, Bndfun)
        # Match in the smooth interior, away from the singular endpoint.
        x = np.linspace(0.5, 0.95, 11)
        assert np.allclose(b(x), np.sqrt(x), atol=1e-3)
