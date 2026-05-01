"""Integration tests for :class:`Singfun` at the :class:`Chebfun` level (plan 03 phase 4)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import chebpy
from chebpy import chebfun
from chebpy.bndfun import Bndfun
from chebpy.maps import MapParams
from chebpy.singfun import Singfun


# ---------------------------------------------------------------------------
#  chebfun(..., sing=...) constructor
# ---------------------------------------------------------------------------
class TestChebfunSingKwarg:
    """``chebfun(f, [a, b], sing=...)`` builds Singfun-backed pieces."""

    def test_single_piece_left(self):
        """A single-piece domain with ``sing='left'`` produces one :class:`Singfun`."""
        f = chebfun(np.sqrt, [0.0, 1.0], sing="left")
        assert len(f.funs) == 1
        assert isinstance(f.funs[0], Singfun)
        x = np.linspace(0.001, 0.999, 21)
        assert np.allclose(f(x), np.sqrt(x), atol=1e-14)

    def test_single_piece_right(self):
        """A single-piece domain with ``sing='right'`` produces one :class:`Singfun`."""
        f = chebfun(lambda x: np.sqrt(1.0 - x), [0.0, 1.0], sing="right")
        assert len(f.funs) == 1
        assert isinstance(f.funs[0], Singfun)
        x = np.linspace(0.001, 0.999, 21)
        assert np.allclose(f(x), np.sqrt(1.0 - x), atol=1e-10)

    def test_single_piece_both(self):
        """A single-piece domain with ``sing='both'`` integrates to ``pi/8``."""
        f = chebfun(lambda x: np.sqrt(x * (1.0 - x)), [0.0, 1.0], sing="both")
        assert len(f.funs) == 1
        assert isinstance(f.funs[0], Singfun)
        assert float(f.sum()) == pytest.approx(np.pi / 8.0, abs=1e-13)

    def test_multipiece_left(self):
        """``sing='left'`` with multiple pieces makes only the leftmost a :class:`Singfun`."""
        f = chebfun(np.sqrt, [0.0, 0.3, 1.0], sing="left")
        assert isinstance(f.funs[0], Singfun)
        assert isinstance(f.funs[1], Bndfun)
        assert not isinstance(f.funs[1], Singfun)
        x = np.linspace(0.001, 0.999, 41)
        assert np.allclose(f(x), np.sqrt(x), atol=1e-13)

    def test_multipiece_both(self):
        """``sing='both'`` with multiple pieces makes the boundary pieces :class:`Singfun`."""
        f = chebfun(lambda x: np.sqrt(x * (1.0 - x)), [0.0, 0.5, 1.0], sing="both")
        assert isinstance(f.funs[0], Singfun)
        assert f.funs[0].map.side == "left"
        assert isinstance(f.funs[-1], Singfun)
        assert f.funs[-1].map.side == "right"
        assert float(f.sum()) == pytest.approx(np.pi / 8.0, abs=1e-13)

    def test_invalid_sing_raises(self):
        """An unrecognised ``sing`` keyword raises :class:`ValueError`."""
        with pytest.raises(ValueError):
            chebfun(np.sqrt, [0.0, 1.0], sing="middle")

    def test_fixedlen_with_sing_raises(self):
        """Mixing ``n=`` with ``sing=`` is not supported in v1."""
        with pytest.raises(NotImplementedError):
            chebfun(np.sqrt, [0.0, 1.0], n=32, sing="left")


# ---------------------------------------------------------------------------
#  mixed-piece arithmetic between Singfun and Bndfun
# ---------------------------------------------------------------------------
class TestMixedArithmetic:
    """Arithmetic across :class:`Singfun` and :class:`Bndfun` pieces of the same interval."""

    @pytest.fixture
    def s(self):
        """Square-root :class:`Singfun` on ``[0, 1]``."""
        return Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", params=MapParams(alpha=1.0))

    @pytest.fixture
    def b(self):
        """``x**2`` :class:`Bndfun` on ``[0, 1]``."""
        from chebpy.utilities import Interval

        return Bndfun.initfun_adaptive(lambda x: x * x, Interval(0.0, 1.0))

    def test_singfun_plus_bndfun_returns_singfun(self, s, b):
        """Adding a :class:`Bndfun` to a :class:`Singfun` rebuilds as :class:`Singfun`."""
        r = s + b
        assert isinstance(r, Singfun)
        x = np.linspace(0.05, 0.95, 11)
        assert np.allclose(r(x), np.sqrt(x) + x * x, atol=1e-13)

    def test_bndfun_plus_singfun_returns_singfun(self, s, b):
        """Adding a :class:`Singfun` to a :class:`Bndfun` rebuilds as :class:`Singfun`."""
        r = b + s
        assert isinstance(r, Singfun)
        x = np.linspace(0.05, 0.95, 11)
        assert np.allclose(r(x), np.sqrt(x) + x * x, atol=1e-13)

    def test_singfun_times_bndfun(self, s, b):
        """Multiplication across mixed types preserves the singular representation."""
        r = s * b
        assert isinstance(r, Singfun)
        x = np.linspace(0.05, 0.95, 11)
        assert np.allclose(r(x), np.sqrt(x) * x * x, atol=1e-13)

    def test_singfun_minus_bndfun(self, s, b):
        """Subtraction across mixed types preserves the singular representation."""
        r = s - b
        assert isinstance(r, Singfun)
        x = np.linspace(0.05, 0.95, 11)
        assert np.allclose(r(x), np.sqrt(x) - x * x, atol=1e-13)

    def test_singfun_plus_singfun_same_map_fast_path(self, s):
        """Two :class:`Singfun` instances with identical maps share their t-grid."""
        s2 = Singfun.initfun_adaptive(lambda x: 1.0 / (1.0 + x), [0.0, 1.0], sing="left", params=MapParams(alpha=1.0))
        assert s._can_share_onefun_with(s2)
        r = s + s2
        assert isinstance(r, Singfun)
        x = np.linspace(0.05, 0.95, 11)
        assert np.allclose(r(x), np.sqrt(x) + 1.0 / (1.0 + x), atol=1e-13)

    def test_singfun_plus_singfun_different_alpha_rebuilds(self, s):
        """Two :class:`Singfun` instances with different alpha rebuild adaptively."""
        s3 = Singfun.initfun_adaptive(lambda x: 1.0 / (1.0 + x), [0.0, 1.0], sing="left", params=MapParams(alpha=2.0))
        assert not s._can_share_onefun_with(s3)
        r = s + s3
        assert isinstance(r, Singfun)
        x = np.linspace(0.05, 0.95, 11)
        assert np.allclose(r(x), np.sqrt(x) + 1.0 / (1.0 + x), atol=1e-13)


# ---------------------------------------------------------------------------
#  Singfun.restrict semantics
# ---------------------------------------------------------------------------
class TestRestrict:
    """``Singfun.restrict`` falls back to :class:`Bndfun` on interior subintervals."""

    def test_trivial_restrict_returns_self(self):
        """Restricting to the full interval returns ``self``."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", params=MapParams(alpha=1.0))
        r = f.restrict([0.0, 1.0])
        assert r is f

    def test_restrict_to_clustered_endpoint(self):
        """A subinterval sharing the clustered endpoint stays :class:`Singfun`."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", params=MapParams(alpha=1.0))
        r = f.restrict([0.0, 0.5])
        assert isinstance(r, Singfun)
        assert tuple(r.support) == (0.0, 0.5)
        x = np.linspace(0.001, 0.499, 21)
        assert np.allclose(r(x), np.sqrt(x), atol=1e-14)

    def test_restrict_to_interior_returns_bndfun(self):
        """A purely interior subinterval drops to a :class:`Bndfun`."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", params=MapParams(alpha=1.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = f.restrict([0.2, 0.8])
        assert isinstance(r, Bndfun)
        assert not isinstance(r, Singfun)
        x = np.linspace(0.21, 0.79, 21)
        assert np.allclose(r(x), np.sqrt(x), atol=1e-14)

    def test_restrict_to_opposite_endpoint_returns_bndfun(self):
        """Restricting a left-singular fun to a right-side range drops to :class:`Bndfun`."""
        f = Singfun.initfun_adaptive(np.sqrt, [0.0, 1.0], sing="left", params=MapParams(alpha=1.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = f.restrict([0.5, 1.0])
        assert isinstance(r, Bndfun)
        assert not isinstance(r, Singfun)

    def test_double_slit_restrict_to_left_half_becomes_left_singfun(self):
        """A two-sided :class:`Singfun` restricted to the left half becomes left-singular."""
        f = Singfun.initfun_adaptive(
            lambda x: np.sqrt(x * (1.0 - x)),
            [0.0, 1.0],
            sing="both",
            params=MapParams(alpha=1.0),
        )
        r = f.restrict([0.0, 0.5])
        assert isinstance(r, Singfun)
        assert r.map.side == "left"

    def test_double_slit_restrict_to_right_half_becomes_right_singfun(self):
        """A two-sided :class:`Singfun` restricted to the right half becomes right-singular."""
        f = Singfun.initfun_adaptive(
            lambda x: np.sqrt(x * (1.0 - x)),
            [0.0, 1.0],
            sing="both",
            params=MapParams(alpha=1.0),
        )
        r = f.restrict([0.5, 1.0])
        assert isinstance(r, Singfun)
        assert r.map.side == "right"


# ---------------------------------------------------------------------------
#  conv refusal
# ---------------------------------------------------------------------------
class TestConvRefusesSingfun:
    """``Chebfun.conv`` cleanly refuses Singfun pieces."""

    def test_conv_refuses_singfun_lhs(self):
        """``conv`` on a Singfun-backed Chebfun raises :class:`NotImplementedError`."""
        f = chebfun(np.sqrt, [0.0, 1.0], sing="left")
        g = chebfun(lambda x: 1.0 + 0 * x, [0.0, 1.0])
        with pytest.raises(NotImplementedError, match="Singfun"):
            f.conv(g)

    def test_conv_refuses_singfun_rhs(self):
        """``conv`` refuses when the right-hand operand contains a Singfun piece."""
        f = chebfun(np.sqrt, [0.0, 1.0], sing="left")
        g = chebfun(lambda x: 1.0 + 0 * x, [0.0, 1.0])
        with pytest.raises(NotImplementedError, match="Singfun"):
            g.conv(f)


# ---------------------------------------------------------------------------
#  package-level export
# ---------------------------------------------------------------------------
class TestExports:
    """Phase-4 exports are available from the top-level ``chebpy`` package."""

    def test_singfun_exported(self):
        """:class:`Singfun` is reachable via ``chebpy.Singfun``."""
        assert chebpy.Singfun is Singfun
