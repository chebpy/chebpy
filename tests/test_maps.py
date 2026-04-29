"""Unit tests for the non-affine clustering maps in :mod:`chebpy.maps`."""

import numpy as np
import pytest

from chebpy.maps import DoubleSlitMap, SingleSlitMap
from chebpy.utilities import IntervalMap

# Sample grid avoiding the open-endpoint singular limits of ``q``.
INTERIOR = np.linspace(-0.99, 0.99, 41)
# Tighter grid for round-trip tests: well clear of the clustered endpoint, where
# ``exp(-alpha * q)`` underflows to 0 in float64 and the inverse is unrecoverable.
ROUNDTRIP = np.linspace(-0.9, 0.9, 37)


# ---------------------------------------------------------------------------
#  protocol conformance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ctor",
    [
        lambda: SingleSlitMap(0.0, 1.0, alpha=1.0, side="left"),
        lambda: SingleSlitMap(0.0, 1.0, alpha=2.5, side="right"),
        lambda: DoubleSlitMap(-1.0, 1.0, alpha=1.0),
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
        lambda: SingleSlitMap(0.0, 1.0, alpha=0.0),
        lambda: SingleSlitMap(0.0, 1.0, alpha=-1.0),
        lambda: SingleSlitMap(0.0, 1.0, side="middle"),
        lambda: DoubleSlitMap(1.0, 0.0),
        lambda: DoubleSlitMap(0.0, 1.0, alpha=0.0),
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
        """``formap`` agrees with the logical support at ``t = ±1``."""
        m = SingleSlitMap(2.0, 5.0, alpha=1.5, side=side)
        assert m.formap(-1.0) == pytest.approx(2.0)
        assert m.formap(1.0) == pytest.approx(5.0)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_round_trip(self, side):
        """``invmap ∘ formap = id`` on the numerically representable interior."""
        m = SingleSlitMap(-1.0, 3.0, alpha=1.0, side=side)
        x = m.formap(ROUNDTRIP)
        t_back = m.invmap(x)
        assert np.allclose(t_back, ROUNDTRIP, atol=1e-10)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_monotonicity(self, side):
        """``formap`` is non-decreasing on the interior (strict in exact arithmetic)."""
        m = SingleSlitMap(0.0, 10.0, alpha=1.2, side=side)
        x = m.formap(INTERIOR)
        assert np.all(np.diff(x) >= 0)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_drvmap_matches_finite_difference(self, side):
        """``drvmap`` agrees with a centred finite difference of ``formap``."""
        m = SingleSlitMap(-2.0, 4.0, alpha=1.0, side=side)
        # Stay clear of the clustered endpoint where the FD step is ill-conditioned.
        t = np.linspace(-0.8, 0.8, 21)
        h = 1e-6
        fd = (m.formap(t + h) - m.formap(t - h)) / (2 * h)
        analytic = m.drvmap(t)
        assert np.allclose(analytic, fd, rtol=1e-5, atol=1e-7)

    def test_left_clusters_at_a(self):
        """A ``side='left'`` map's derivative vanishes at ``t = -1``."""
        m = SingleSlitMap(0.0, 1.0, alpha=1.0, side="left")
        assert m.drvmap(-1.0) == pytest.approx(0.0)
        assert m.drvmap(1.0) > 0.0

    def test_right_clusters_at_b(self):
        """A ``side='right'`` map's derivative vanishes at ``t = 1``."""
        m = SingleSlitMap(0.0, 1.0, alpha=1.0, side="right")
        assert m.drvmap(1.0) == pytest.approx(0.0)
        assert m.drvmap(-1.0) > 0.0

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
        """``formap`` agrees with the logical support at ``t = ±1``."""
        m = DoubleSlitMap(-2.0, 3.0, alpha=1.0)
        assert m.formap(-1.0) == pytest.approx(-2.0)
        assert m.formap(1.0) == pytest.approx(3.0)

    def test_centre_value(self):
        """``formap(0)`` lands at the midpoint of the logical interval."""
        m = DoubleSlitMap(-2.0, 3.0, alpha=1.0)
        assert m.formap(0.0) == pytest.approx(0.5)

    def test_round_trip(self):
        """``invmap ∘ formap = id`` on the numerically representable interior."""
        m = DoubleSlitMap(0.0, 1.0, alpha=1.0)
        x = m.formap(ROUNDTRIP)
        t_back = m.invmap(x)
        assert np.allclose(t_back, ROUNDTRIP, atol=1e-10)

    def test_monotonicity(self):
        """``formap`` is non-decreasing on the interior (strict in exact arithmetic)."""
        m = DoubleSlitMap(0.0, 1.0, alpha=2.0)
        x = m.formap(INTERIOR)
        assert np.all(np.diff(x) >= 0)

    def test_symmetry_about_midpoint(self):
        """The map is symmetric: ``formap(-t) + formap(t) = a + b``."""
        m = DoubleSlitMap(-1.0, 1.0, alpha=1.0)
        t = np.linspace(0.05, 0.95, 19)
        assert np.allclose(m.formap(-t) + m.formap(t), 0.0, atol=1e-12)

    def test_drvmap_matches_finite_difference(self):
        """``drvmap`` agrees with a centred finite difference of ``formap``."""
        m = DoubleSlitMap(-1.0, 1.0, alpha=1.0)
        t = np.linspace(-0.7, 0.7, 21)
        h = 1e-6
        fd = (m.formap(t + h) - m.formap(t - h)) / (2 * h)
        analytic = m.drvmap(t)
        assert np.allclose(analytic, fd, rtol=1e-5, atol=1e-7)

    def test_endpoint_derivative_vanishes(self):
        """The map's derivative vanishes at both endpoints."""
        m = DoubleSlitMap(-1.0, 1.0, alpha=1.0)
        assert m.drvmap(-1.0) == pytest.approx(0.0)
        assert m.drvmap(1.0) == pytest.approx(0.0)
        assert m.drvmap(0.0) > 0.0

    def test_scalar_returns_scalar(self):
        """Scalar inputs yield Python ``float`` outputs (no 0-d arrays)."""
        m = DoubleSlitMap(0.0, 1.0)
        assert isinstance(m.formap(0.0), float)
        assert isinstance(m.invmap(0.5), float)
        assert isinstance(m.drvmap(0.0), float)
