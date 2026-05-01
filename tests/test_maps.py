"""Unit tests for the non-affine slit-strip maps in :mod:`chebpy.maps`."""

import numpy as np
import pytest

from chebpy.maps import DoubleSlitMap, MapParams, SingleSlitMap
from chebpy.utilities import IntervalMap

# Sample grid avoiding the open-endpoint singular limits.
INTERIOR = np.linspace(-0.99, 0.99, 41)
# Tighter grid for round-trip tests: well clear of the clustered endpoint, where
# super-exponential decay underflows to 0 in float64 and the inverse is unrecoverable.
ROUNDTRIP = np.linspace(-0.9, 0.9, 37)


# ---------------------------------------------------------------------------
#  MapParams
# ---------------------------------------------------------------------------
class TestMapParams:
    """Behavioural tests for the :class:`MapParams` dataclass."""

    def test_defaults(self):
        """Default ``L`` and ``alpha`` are positive."""
        p = MapParams()
        assert p.L > 0
        assert p.alpha > 0

    @pytest.mark.parametrize("kwargs", [{"L": 0.0}, {"L": -1.0}, {"alpha": 0.0}, {"alpha": -1.0}])
    def test_rejects_non_positive(self, kwargs):
        """Non-positive ``L`` or ``alpha`` raises :class:`ValueError`."""
        with pytest.raises(ValueError):
            MapParams(**kwargs)

    def test_frozen(self):
        """The dataclass is frozen (immutable)."""
        p = MapParams()
        with pytest.raises((AttributeError, TypeError)):
            p.L = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
#  protocol conformance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ctor",
    [
        lambda: SingleSlitMap(0.0, 1.0, MapParams(alpha=1.0), side="left"),
        lambda: SingleSlitMap(0.0, 1.0, MapParams(alpha=2.5), side="right"),
        lambda: DoubleSlitMap(-1.0, 1.0, MapParams(alpha=1.0)),
    ],
)
def test_implements_intervalmap_protocol(ctor):
    """Both clustering maps satisfy the structural :class:`IntervalMap` protocol."""
    m = ctor()
    assert isinstance(m, IntervalMap)


# ---------------------------------------------------------------------------
#  constructor validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ctor",
    [
        lambda: SingleSlitMap(1.0, 1.0),  # a == b
        lambda: SingleSlitMap(2.0, 1.0),  # a > b
        lambda: SingleSlitMap(0.0, 1.0, side="middle"),
        lambda: DoubleSlitMap(1.0, 0.0),
    ],
)
def test_constructor_rejects_bad_arguments(ctor):
    """Invalid constructor arguments raise :class:`ValueError`."""
    with pytest.raises(ValueError):
        ctor()


# ---------------------------------------------------------------------------
#  SingleSlitMap
# ---------------------------------------------------------------------------
class TestSingleSlitMap:
    """Behavioural tests for :class:`~chebpy.maps.SingleSlitMap`."""

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_endpoint_values(self, side):
        """``formap`` agrees with the logical support at ``t = pm 1`` up to the gap."""
        m = SingleSlitMap(2.0, 5.0, MapParams(alpha=1.5), side=side)
        # The clustered endpoint matches up to the (small) endpoint gap; the
        # opposite endpoint is exact by construction (gamma normalisation).
        if side == "left":
            assert m.formap(-1.0) == pytest.approx(2.0, abs=2.0 * m.gap)
            assert m.formap(1.0) == pytest.approx(5.0, abs=1e-12)
        else:
            assert m.formap(-1.0) == pytest.approx(2.0, abs=1e-12)
            assert m.formap(1.0) == pytest.approx(5.0, abs=2.0 * m.gap)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_round_trip(self, side):
        """``invmap o formap = id`` on the numerically representable interior."""
        m = SingleSlitMap(-1.0, 3.0, side=side)
        x = m.formap(ROUNDTRIP)
        t_back = m.invmap(x)
        assert np.allclose(t_back, ROUNDTRIP, atol=1e-10)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_monotonicity(self, side):
        """``formap`` is non-decreasing on the interior (strict in exact arithmetic)."""
        m = SingleSlitMap(0.0, 10.0, MapParams(alpha=1.2), side=side)
        x = m.formap(INTERIOR)
        assert np.all(np.diff(x) >= 0)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_drvmap_matches_finite_difference(self, side):
        """``drvmap`` agrees with a centred finite difference of ``formap``."""
        m = SingleSlitMap(-2.0, 4.0, side=side)
        # Stay clear of the clustered endpoint where the FD step is ill-conditioned.
        t = np.linspace(-0.8, 0.8, 21)
        h = 1e-6
        fd = (m.formap(t + h) - m.formap(t - h)) / (2 * h)
        analytic = m.drvmap(t)
        assert np.allclose(analytic, fd, rtol=1e-5, atol=1e-7)

    def test_left_clusters_at_a(self):
        """A ``side='left'`` map's derivative is tiny at ``t = -1`` (vanishes as L -> inf)."""
        m = SingleSlitMap(0.0, 1.0, side="left")
        # drvmap(-1) ~ L/2 * sigmoid(pi*(-L+gamma)/alpha) -> 0 super-exponentially in L.
        assert m.drvmap(-1.0) == pytest.approx(0.0, abs=1e-7)
        assert m.drvmap(1.0) > 0.0

    def test_right_clusters_at_b(self):
        """A ``side='right'`` map's derivative is tiny at ``t = 1`` (vanishes as L -> inf)."""
        m = SingleSlitMap(0.0, 1.0, side="right")
        assert m.drvmap(1.0) == pytest.approx(0.0, abs=1e-7)
        assert m.drvmap(-1.0) > 0.0

    def test_default_params(self):
        """Default ``params`` is :class:`MapParams` with documented defaults."""
        m = SingleSlitMap(0.0, 1.0)
        assert m.params == MapParams()
        assert m.alpha == m.params.alpha
        assert m.L == m.params.L

    def test_gap_smaller_for_larger_L(self):
        """Larger ``L`` produces a smaller endpoint gap."""
        m_small = SingleSlitMap(0.0, 1.0, MapParams(L=2.0))
        m_large = SingleSlitMap(0.0, 1.0, MapParams(L=8.0))
        assert m_small.gap > m_large.gap > 0
        assert m_large.gap < 1e-9

    def test_phi_S_normalisation(self):
        """``u(s=0) = 1`` to working precision: ``formap(1) = b`` exactly (left side)."""
        m = SingleSlitMap(0.0, 1.0, MapParams(alpha=0.7, L=5.0), side="left")
        # At y = 1, s = 0, u = 1 by construction (gamma chosen for this normalisation).
        assert m.formap(1.0) == pytest.approx(1.0, abs=1e-12)

    def test_scalar_returns_scalar(self):
        """Scalar inputs yield Python ``float`` outputs (no 0-d arrays)."""
        m = SingleSlitMap(0.0, 1.0)
        assert isinstance(m.formap(0.0), float)
        assert isinstance(m.invmap(0.5), float)
        assert isinstance(m.drvmap(0.0), float)


# ---------------------------------------------------------------------------
#  DoubleSlitMap
# ---------------------------------------------------------------------------
class TestDoubleSlitMap:
    """Behavioural tests for :class:`~chebpy.maps.DoubleSlitMap`."""

    def test_endpoint_values(self):
        """``formap`` agrees with the logical support at ``t = pm 1`` up to the gap."""
        m = DoubleSlitMap(-2.0, 3.0)
        assert m.formap(-1.0) == pytest.approx(-2.0, abs=2.0 * m.gap)
        assert m.formap(1.0) == pytest.approx(3.0, abs=2.0 * m.gap)

    def test_centre_value(self):
        """``formap(0)`` lands at the midpoint of the logical interval."""
        m = DoubleSlitMap(-2.0, 3.0)
        assert m.formap(0.0) == pytest.approx(0.5)

    def test_round_trip(self):
        """``invmap o formap = id`` on the numerically representable interior."""
        m = DoubleSlitMap(0.0, 1.0)
        x = m.formap(ROUNDTRIP)
        t_back = m.invmap(x)
        assert np.allclose(t_back, ROUNDTRIP, atol=1e-10)

    def test_monotonicity(self):
        """``formap`` is non-decreasing on the interior (strict in exact arithmetic)."""
        m = DoubleSlitMap(0.0, 1.0, MapParams(alpha=2.0))
        x = m.formap(INTERIOR)
        assert np.all(np.diff(x) >= 0)

    def test_symmetry_about_midpoint(self):
        """The map is symmetric: ``formap(-t) + formap(t) = a + b``."""
        m = DoubleSlitMap(-1.0, 1.0)
        t = np.linspace(0.05, 0.95, 19)
        assert np.allclose(m.formap(-t) + m.formap(t), 0.0, atol=1e-12)

    def test_drvmap_matches_finite_difference(self):
        """``drvmap`` agrees with a centred finite difference of ``formap``."""
        m = DoubleSlitMap(-1.0, 1.0)
        t = np.linspace(-0.7, 0.7, 21)
        h = 1e-6
        fd = (m.formap(t + h) - m.formap(t - h)) / (2 * h)
        analytic = m.drvmap(t)
        assert np.allclose(analytic, fd, rtol=1e-5, atol=1e-7)

    def test_endpoint_derivative_vanishes(self):
        """The map's derivative is tiny at both endpoints (vanishes as L -> inf)."""
        m = DoubleSlitMap(-1.0, 1.0)
        # drvmap(pm 1) ~ L * (sigmoid_+ - sigmoid_-) -> 0 super-exponentially in L.
        assert m.drvmap(-1.0) == pytest.approx(0.0, abs=1e-7)
        assert m.drvmap(1.0) == pytest.approx(0.0, abs=1e-7)
        assert m.drvmap(0.0) > 0.0

    def test_default_params(self):
        """Default ``params`` is :class:`MapParams` with documented defaults."""
        m = DoubleSlitMap(0.0, 1.0)
        assert m.params == MapParams()
        assert m.alpha == m.params.alpha
        assert m.L == m.params.L

    def test_gap_smaller_for_larger_L(self):
        """Larger ``L`` produces a smaller endpoint gap."""
        m_small = DoubleSlitMap(0.0, 1.0, MapParams(L=2.0))
        m_large = DoubleSlitMap(0.0, 1.0, MapParams(L=8.0))
        assert m_small.gap > m_large.gap > 0
        assert m_large.gap < 1e-9

    def test_scalar_returns_scalar(self):
        """Scalar inputs yield Python ``float`` outputs (no 0-d arrays)."""
        m = DoubleSlitMap(0.0, 1.0)
        assert isinstance(m.formap(0.0), float)
        assert isinstance(m.invmap(0.5), float)
        assert isinstance(m.drvmap(0.0), float)
