"""Implementation of :class:`Singfun` for functions with endpoint singularities.

A :class:`Singfun` represents a function on a bounded interval ``[a, b]``
that is analytic on the open interval ``(a, b)`` but may have algebraic or
logarithmic branch-type singularities at one or both endpoints.  It is a
sibling of :class:`~chebpy.bndfun.Bndfun` and
:class:`~chebpy.compactfun.CompactFun` under
:class:`~chebpy.classicfun.Classicfun`: the only structural novelty is that
the bijective map between the storage variable ``t in [-1, 1]`` and the
logical variable ``x in [a, b]`` is a non-affine, endpoint-clustering
exponential transform (see :mod:`chebpy.maps`) rather than the affine
:class:`~chebpy.utilities.Interval` map.

For a function ``f`` with branch-type endpoint behaviour, the composition
``f(m(t))`` is analytic in a Bernstein ellipse around ``[-1, 1]`` and is
therefore resolved by ordinary Chebyshev interpolation in ``t``.  All the
existing :class:`~chebpy.classicfun.Classicfun` plumbing (``__call__``,
``roots``, the binary operators) is reused unchanged via the ``map``
property override; only the calculus operations (``sum``, ``cumsum``,
``diff``) need bespoke implementations because the affine-Jacobian
shortcuts in :class:`~chebpy.classicfun.Classicfun` no longer apply.

This is a v1 implementation (Phase 3 of the singfun plan); see
``docs/plans/03-singfun-mapped-integration.md`` for the broader design and
the remaining closure / fallback work.
"""

from __future__ import annotations

from typing import Any

from .classicfun import Classicfun, techdict
from .maps import DoubleSlitMap, SingleSlitMap
from .settings import _preferences as prefs
from .utilities import Interval, IntervalMap


def _build_map(a: float, b: float, sing: str, alpha: float) -> IntervalMap:
    """Construct the appropriate non-affine map for the requested singularity pattern.

    Args:
        a: Left endpoint of the logical interval.
        b: Right endpoint of the logical interval.
        sing: One of ``"left"``, ``"right"``, or ``"both"``.
        alpha: Positive clustering strength.

    Returns:
        IntervalMap: A :class:`SingleSlitMap` (for ``"left"`` / ``"right"``)
            or :class:`DoubleSlitMap` (for ``"both"``).

    Raises:
        ValueError: If ``sing`` is not one of the recognised values.
    """
    if sing in ("left", "right"):
        return SingleSlitMap(a, b, alpha=alpha, side=sing)
    if sing == "both":
        return DoubleSlitMap(a, b, alpha=alpha)
    msg = f"sing must be 'left', 'right', or 'both'; got {sing!r}"
    raise ValueError(msg)


class Singfun(Classicfun):
    """Functions with branch-type endpoint singularities on a bounded interval.

    A :class:`Singfun` stores:

    - ``self.onefun`` (inherited): a standard :class:`~chebpy.onefun.Onefun`
      (typically a :class:`~chebpy.chebtech.Chebtech`) on ``[-1, 1]``
      representing ``f(m(t))``, which is analytic by construction.
    - ``self._interval`` (inherited): an :class:`~chebpy.utilities.Interval`
      ``Interval(a, b)`` carrying the logical support endpoints.
    - ``self._map``: a non-affine :class:`~chebpy.utilities.IntervalMap`
      (a :class:`~chebpy.maps.SingleSlitMap` or
      :class:`~chebpy.maps.DoubleSlitMap`) — the actual bijection between
      the reference and logical variables.  Returned by the
      :attr:`map` override so that
      :meth:`~chebpy.classicfun.Classicfun.__call__` and
      :meth:`~chebpy.classicfun.Classicfun.roots` route through the
      non-affine map without further changes.
    """

    # Mixed-subclass binary ops (Singfun + Bndfun, etc.) reconstruct on the
    # operand with the highest priority.  Singfun outranks Bndfun/CompactFun
    # because the singularity must be preserved in the result.
    _singularity_priority: int = 10

    def __init__(self, onefun: Any, interval: Any, map_: IntervalMap) -> None:
        """Create a new :class:`Singfun`.

        Args:
            onefun: The :class:`~chebpy.onefun.Onefun` on ``[-1, 1]``
                representing ``f(m(t))``.
            interval: The logical support :class:`~chebpy.utilities.Interval`
                ``Interval(a, b)`` (always finite for v1).
            map_: The non-affine :class:`~chebpy.utilities.IntervalMap` between
                ``[-1, 1]`` and ``[a, b]``.
        """
        super().__init__(onefun, interval)
        self._map = map_

    def _rebuild(self, onefun: Any) -> Singfun:
        """Construct a new :class:`Singfun` preserving the map.

        Used by every operation in :class:`~chebpy.classicfun.Classicfun`
        that produces a new instance from a replacement ``onefun``
        (``copy``, ``simplify``, the unary operators, the binary operators
        between two same-map :class:`Singfun` instances).
        """
        return type(self)(onefun, self._interval, self._map)

    def _can_share_onefun_with(self, other: Any) -> bool:
        """Two :class:`Singfun` instances share a t-grid only when their maps match.

        The ``Onefun`` coefficients of a :class:`Singfun` represent ``f(m(t))``
        sampled at Chebyshev nodes in ``t``-space; if the maps differ, those
        nodes encode different logical points and onefun-level arithmetic is
        no longer correct.  In that case the parent class falls back to
        rebuilding the result adaptively on the dominant operand's map.
        """
        if not super()._can_share_onefun_with(other):
            return False
        return self._maps_equal(self._map, other._map)

    def _rebuild_from_callable(self, f: Any) -> Singfun:
        """Adaptively rebuild a :class:`Singfun` evaluating callable ``f`` on this map."""
        m = self._map
        if isinstance(m, SingleSlitMap):
            return type(self).initfun_adaptive(f, self._interval, sing=m.side, alpha=m.alpha)
        if isinstance(m, DoubleSlitMap):
            return type(self).initfun_adaptive(f, self._interval, sing="both", alpha=m.alpha)
        msg = "Singfun._rebuild_from_callable: unknown map type"
        raise NotImplementedError(msg)  # pragma: no cover

    @staticmethod
    def _maps_equal(m1: IntervalMap, m2: IntervalMap) -> bool:
        """Structural equality check for the maps used by :class:`Singfun`."""
        if isinstance(m1, SingleSlitMap) and isinstance(m2, SingleSlitMap):
            return m1.side == m2.side and m1.alpha == m2.alpha and m1.support == m2.support
        if isinstance(m1, DoubleSlitMap) and isinstance(m2, DoubleSlitMap):
            return m1.alpha == m2.alpha and m1.support == m2.support
        return False

    # ------------
    #  properties
    # ------------
    @property
    def map(self) -> IntervalMap:
        """Return the non-affine clustering map used by this :class:`Singfun`."""
        return self._map

    # --------------------------
    #  alternative constructors
    # --------------------------
    @classmethod
    def initempty(cls) -> Singfun:
        """Initialise an empty :class:`Singfun` with a default left-clustered map."""
        iv = Interval(-1.0, 1.0)
        m = SingleSlitMap(-1.0, 1.0, alpha=1.0, side="left")
        onefun = techdict[prefs.tech].initempty(interval=iv)
        return cls(onefun, iv, m)

    @classmethod
    def initconst(cls, c: Any, interval: Any, *, sing: str = "left", alpha: float = 1.0) -> Singfun:
        """Initialise a constant :class:`Singfun`.

        Args:
            c: The constant value.
            interval: The bounded logical interval.
            sing: Which endpoint(s) to cluster (``"left"`` / ``"right"`` / ``"both"``).
                Default ``"left"``.
            alpha: Positive clustering strength for the underlying map.
                Default ``1.0``.
        """
        a, b = float(interval[0]), float(interval[1])
        iv = Interval(a, b)
        m = _build_map(a, b, sing, alpha)
        onefun = techdict[prefs.tech].initconst(c, interval=iv)
        return cls(onefun, iv, m)

    @classmethod
    def initidentity(cls, interval: Any, *, sing: str = "left", alpha: float = 1.0) -> Singfun:
        """Initialise the identity ``f(x) = x`` as a :class:`Singfun`.

        Note that ``f(x) = x`` is itself analytic on ``[a, b]`` and does not
        require a clustering map; this constructor exists primarily for
        symmetry with the :class:`Bndfun` API and for testing.
        """
        a, b = float(interval[0]), float(interval[1])
        iv = Interval(a, b)
        m = _build_map(a, b, sing, alpha)
        # Sample x = m(t) at the t-Chebyshev nodes so the Onefun encodes f(m(t)) = m(t).
        onefun = techdict[prefs.tech].initfun(lambda t: m.formap(t), interval=iv)
        return cls(onefun, iv, m)

    @classmethod
    def initfun_adaptive(
        cls,
        f: Any,
        interval: Any,
        *,
        sing: str = "left",
        alpha: float = 1.0,
    ) -> Singfun:
        """Adaptive constructor for a :class:`Singfun`.

        Builds the underlying :class:`~chebpy.chebtech.Chebtech` (or
        :class:`~chebpy.trigtech.Trigtech`) by adaptively sampling ``f``
        composed with the chosen clustering map.

        Args:
            f: Callable accepting a NumPy array of logical points and returning
                an array of function values.
            interval: The bounded logical interval ``(a, b)``.
            sing: Which endpoint(s) of the interval carry a branch-type
                singularity.  One of ``"left"``, ``"right"``, ``"both"``.
            alpha: Positive clustering strength.

        Returns:
            Singfun: The newly constructed :class:`Singfun`.
        """
        a, b = float(interval[0]), float(interval[1])
        iv = Interval(a, b)
        m = _build_map(a, b, sing, alpha)
        onefun = techdict[prefs.tech].initfun(lambda t: f(m.formap(t)), interval=iv)
        return cls(onefun, iv, m)

    @classmethod
    def initfun_fixedlen(
        cls,
        f: Any,
        interval: Any,
        n: int,
        *,
        sing: str = "left",
        alpha: float = 1.0,
    ) -> Singfun:
        """Fixed-length constructor for a :class:`Singfun` (``n`` Chebyshev coefficients)."""
        a, b = float(interval[0]), float(interval[1])
        iv = Interval(a, b)
        m = _build_map(a, b, sing, alpha)
        onefun = techdict[prefs.tech].initfun(lambda t: f(m.formap(t)), n, interval=iv)
        return cls(onefun, iv, m)

    # ----------
    #  calculus
    # ----------
    def sum(self) -> Any:
        r"""Definite integral of the function over ``[a, b]``.

        Computed in the reference variable via the change-of-variables

        .. math::

            \int_a^b f(x)\,dx \;=\; \int_{-1}^{1} (f \circ m)(t)\, m'(t)\, dt.

        The integrand ``onefun(t) * m'(t)`` is built adaptively as a
        standard :class:`~chebpy.chebtech.Chebtech` on ``[-1, 1]``; ``m'(t)``
        vanishes at the clustered endpoint(s), exactly absorbing the
        integrable singularity of ``f``.
        """
        if self.onefun.isempty:
            return 0.0
        iv = Interval(-1.0, 1.0)
        m = self._map
        onefun = self.onefun
        integrand = techdict[prefs.tech].initfun(lambda t: onefun(t) * m.drvmap(t), interval=iv)
        return integrand.sum()

    def cumsum(self) -> Singfun:
        r"""Indefinite integral ``F(x) = \int_a^x f(s)\,ds`` as a :class:`Singfun`.

        The chain rule gives ``F(m(t)) = \int_{-1}^t (f \circ m)(s)\, m'(s)\, ds``,
        which is the cumulative sum of ``onefun(t) * m'(t)`` and is even more
        regular than ``f`` itself.  The result re-uses the same map.
        """
        if self.onefun.isempty:
            return self._rebuild(self.onefun.copy())
        iv = Interval(-1.0, 1.0)
        m = self._map
        onefun = self.onefun
        integrand = techdict[prefs.tech].initfun(lambda t: onefun(t) * m.drvmap(t), interval=iv)
        return self._rebuild(integrand.cumsum())

    def diff(self) -> Singfun:  # pragma: no cover - not yet implemented
        r"""Differentiation is not yet implemented for :class:`Singfun` (Phase 3 v1).

        ``f'(x) = (f \\circ m)'(t) / m'(t)`` introduces a stronger
        endpoint singularity (``1/m'`` blows up at the clustered endpoint),
        which the current map cannot resolve.  See plan 03 for the planned
        treatment.
        """
        msg = (
            "Singfun.diff is not implemented yet; differentiation introduces a "
            "stronger endpoint singularity that the current map cannot resolve."
        )
        raise NotImplementedError(msg)

    # -----------
    #  utilities
    # -----------
    def restrict(self, subinterval: Any) -> Classicfun:
        """Restrict to a subinterval.

        Behaviour depends on the relationship between ``subinterval`` and the
        clustered endpoint(s):

        * Trivial restriction (``subinterval == self.interval``) returns
          ``self`` unchanged.
        * A subinterval that **shares** the clustered endpoint (e.g. for
          ``sing="left"`` a sub-range ``[a, c]``, or for ``sing="both"``
          ``[a, c]`` / ``[c, b]``) is rebuilt as a :class:`Singfun` with a
          rescaled map that retains the singular endpoint.  Two-sided maps
          are restricted to a one-sided map of the appropriate side.
        * A purely interior subinterval (one that excludes the clustered
          endpoint(s)) is returned as a :class:`~chebpy.bndfun.Bndfun`,
          since the function is analytic there and the affine map suffices.

        This is the closure-fallback described in plan 03 phase 4: the
        result remains a usable :class:`~chebpy.classicfun.Classicfun` but
        may change subclass.
        """
        from .bndfun import Bndfun

        if subinterval not in self.interval:
            from .exceptions import NotSubinterval

            raise NotSubinterval(self.interval, subinterval)
        a, b = float(self._interval[0]), float(self._interval[1])
        sa, sb = float(subinterval[0]), float(subinterval[1])
        if sa == a and sb == b:
            return self

        m = self._map
        new_iv = Interval(sa, sb)

        # Decide which clustered endpoints (if any) the subinterval still touches.
        touches_left = sa == a
        touches_right = sb == b

        if isinstance(m, SingleSlitMap):
            if (m.side == "left" and touches_left) or (m.side == "right" and touches_right):
                return type(self).initfun_adaptive(self, new_iv, sing=m.side, alpha=m.alpha)
            # Interior or opposite-end restriction: drop to Bndfun.
            return Bndfun.initfun_adaptive(self, new_iv)

        if isinstance(m, DoubleSlitMap):
            if touches_left and touches_right:
                # Subinterval == self.interval handled above; this branch is
                # therefore unreachable in normal usage.
                return self  # pragma: no cover
            if touches_left:
                return type(self).initfun_adaptive(self, new_iv, sing="left", alpha=m.alpha)
            if touches_right:
                return type(self).initfun_adaptive(self, new_iv, sing="right", alpha=m.alpha)
            return Bndfun.initfun_adaptive(self, new_iv)

        # Unknown map type — conservative fallback.
        return Bndfun.initfun_adaptive(self, new_iv)  # pragma: no cover

    def translate(self, c: float) -> Singfun:
        """Translate the function: ``g(x) = f(x - c)``.

        The map is rebuilt for the shifted support; the underlying
        ``onefun`` (which lives on ``[-1, 1]``) is unchanged.
        """
        a, b = float(self._interval[0]) + float(c), float(self._interval[1]) + float(c)
        new_iv = Interval(a, b)
        new_map = self._rebuild_map_for(a, b)
        return type(self)(self.onefun, new_iv, new_map)

    # -----------------
    #  internal helpers
    # -----------------
    def _rebuild_map_for(self, a: float, b: float) -> IntervalMap:
        """Return a copy of ``self._map`` rescaled to a new logical interval."""
        m = self._map
        if isinstance(m, SingleSlitMap):
            return SingleSlitMap(a, b, alpha=m.alpha, side=m.side)
        if isinstance(m, DoubleSlitMap):
            return DoubleSlitMap(a, b, alpha=m.alpha)
        # Unknown map type — fall back to leaving the map unchanged; callers
        # of translate that need a non-trivial rebuild should override.
        return m  # pragma: no cover
