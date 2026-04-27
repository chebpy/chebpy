"""Implementation of functions on (semi-)infinite intervals via numerical-support truncation.

This module provides the :class:`CompactFun` class, which sits next to
:class:`~chebpy.bndfun.Bndfun` under :class:`~chebpy.classicfun.Classicfun`.
It represents functions whose user-facing logical interval has one or both
endpoints at ``±inf`` but whose **numerical support** — the set of points
where the function exceeds a configured tolerance — is finite.  Internally,
a :class:`CompactFun` stores a standard :class:`~chebpy.onefun.Onefun`
(Chebtech) on the discovered finite storage interval; outside that interval
the function is reported as identically zero.

This approach is a deliberate departure from MATLAB Chebfun's ``@unbndfun``
(which uses a rational change of variables to map ``(-inf, inf)`` onto
``[-1, 1]``).  See ``docs/plans/02-compactfun-integration.md`` for the design
rationale and scope.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .classicfun import Classicfun, techdict
from .exceptions import CompactFunConstructionError
from .settings import _preferences as prefs
from .utilities import Interval


def _ensure_endpoints(interval: Any) -> tuple[float, float]:
    """Return ``(a, b)`` floats from any 2-element interval-like object.

    Accepts :class:`Interval`, ``numpy.ndarray``, list, or tuple.  Both
    endpoints may be ``±inf``.
    """
    a, b = interval[0], interval[1]
    return float(a), float(b)


def _discover_one_side(
    f: Any, anchor: float, sign: int, tol: float, max_width: float, max_probes: int
) -> tuple[float, float]:
    """Discover the numerical-support boundary on one infinite side.

    Probes ``f`` at ``anchor + sign * 2**k`` for ``k = 0, 1, 2, ...`` up to
    the configured budget.  Returns the discovered finite boundary and the
    observed vertical scale on this side.

    Args:
        f: Callable being approximated.
        anchor: Finite anchor point (the bounded endpoint of a semi-infinite
            interval, or ``0.0`` for the doubly-infinite case).
        sign: ``+1`` for the rightward (toward ``+inf``) side, ``-1`` for the
            leftward side.
        tol: Relative tolerance threshold; the boundary is the smallest
            ``r`` for which ``|f(anchor + sign * r)| < tol * vscale`` for
            all probes farther out.
        max_width: Maximum permitted boundary distance from ``anchor``.
        max_probes: Maximum number of geometric probes.

    Returns:
        Tuple ``(boundary, vscale)`` where ``boundary`` is a finite ``float``
        and ``vscale`` is the largest absolute probed value on this side.

    Raises:
        CompactFunConstructionError: If the function does not decay below
            ``tol * vscale`` within the probing budget or ``max_width``.
    """
    radii: list[float] = []
    values: list[float] = []
    r = 1.0
    for _ in range(max_probes):
        if r > max_width:
            break
        x = anchor + sign * r
        try:
            v = float(np.abs(f(x)))
        except (FloatingPointError, OverflowError, ZeroDivisionError) as err:  # pragma: no cover
            raise CompactFunConstructionError(  # noqa: TRY003
                f"Could not evaluate f at probe x = {x:g} during numerical-support discovery"
            ) from err
        if not np.isfinite(v):
            raise CompactFunConstructionError(  # noqa: TRY003
                f"f returned non-finite value {v} at probe x = {x:g}; CompactFun "
                f"requires the function to be finite at all sampled points."
            )
        radii.append(r)
        values.append(v)
        r *= 2.0
    if not radii:
        return anchor + sign * 1.0, 0.0

    vscale = max(values) if values else 0.0
    threshold = tol * max(vscale, 1.0)

    # Largest radius at which f is still "active" (above threshold).
    active_r = 0.0
    for ri, vi in zip(radii, values, strict=False):
        if vi > threshold:
            active_r = ri

    # If the farthest probe is still active, the function does not decay
    # within the probing budget.
    if values[-1] > threshold:
        raise CompactFunConstructionError(  # noqa: TRY003
            f"Function does not decay below tolerance {tol:g} within "
            f"{radii[-1]:g} of anchor {anchor:g}; heavy-tailed inputs are not "
            f"supported in this release."
        )

    boundary_r = max(2.0 * active_r, 1.0)
    if boundary_r > max_width:
        raise CompactFunConstructionError(  # noqa: TRY003
            f"Discovered numerical support exceeds max_width = {max_width:g}; "
            f"heavy-tailed inputs are not supported in this release."
        )
    return anchor + sign * boundary_r, vscale


def _discover_numsupp(f: Any, a: float, b: float, tol: float, max_width: float, max_probes: int) -> tuple[float, float]:
    """Discover the storage interval ``[a', b']`` for ``f``.

    Args:
        f: Callable being approximated.
        a: Left endpoint of the logical interval (may be ``-inf``).
        b: Right endpoint of the logical interval (may be ``+inf``).
        tol: Relative tolerance for support detection.
        max_width: Maximum permitted storage interval width.
        max_probes: Maximum probes per unbounded side.

    Returns:
        Tuple ``(a', b')`` of finite floats with ``a' < b'``.

    Raises:
        CompactFunConstructionError: If support cannot be discovered.
    """
    left_inf = not np.isfinite(a)
    right_inf = not np.isfinite(b)

    if not (left_inf or right_inf):
        return a, b

    # Anchor: the finite endpoint of a semi-infinite interval, else 0.
    if left_inf and right_inf:
        anchor = 0.0
    elif left_inf:
        anchor = b
    else:
        anchor = a

    if left_inf:
        a_storage, _ = _discover_one_side(f, anchor, -1, tol, max_width, max_probes)
    else:
        a_storage = a

    if right_inf:
        b_storage, _ = _discover_one_side(f, anchor, +1, tol, max_width, max_probes)
    else:
        b_storage = b

    if b_storage - a_storage > max_width:
        raise CompactFunConstructionError(  # noqa: TRY003
            f"Discovered numerical support [{a_storage:g}, {b_storage:g}] exceeds "
            f"max_width = {max_width:g}; heavy-tailed inputs are not supported "
            f"in this release."
        )
    if b_storage <= a_storage:
        # Function appears identically zero; pick a small default storage.
        a_storage, b_storage = anchor - 1.0, anchor + 1.0
    return a_storage, b_storage


class CompactFun(Classicfun):
    """Functions on (semi-)infinite intervals with finite numerical support.

    A :class:`CompactFun` represents a function whose user-facing logical
    interval has one or both endpoints at ``±inf`` but whose numerical
    support — the set where the function exceeds a configured tolerance —
    is finite.  Internally it inherits from :class:`Classicfun` and stores a
    standard :class:`Onefun` on the discovered finite storage interval; the
    function is reported as identically zero outside that interval.

    Two intervals are tracked:

    - ``self._interval`` (inherited): the finite storage interval where the
      underlying ``Onefun`` lives.
    - ``self._logical_interval``: the user-facing interval, which may have
      ``±inf`` endpoints; returned by :attr:`support`.

    For finite logical intervals the two coincide and a :class:`CompactFun`
    behaves identically to :class:`~chebpy.bndfun.Bndfun`.

    Attributes:
        onefun: Inherited; the standard :class:`Onefun` on ``[-1, 1]``.
        support: The logical interval (possibly with ``±inf`` endpoints).
        numerical_support: The finite storage interval.
    """

    def __init__(self, onefun: Any, interval: Any, logical_interval: Any = None) -> None:
        """Create a new :class:`CompactFun` instance.

        Args:
            onefun: The :class:`Onefun` representing the function on ``[-1, 1]``.
            interval: The finite storage :class:`Interval` (always finite).
            logical_interval: The user-facing interval (possibly with ``±inf``
                endpoints).  Defaults to ``interval`` if omitted.
        """
        super().__init__(onefun, interval)
        if logical_interval is None:
            self._logical_interval = np.asarray(interval, dtype=float)
        else:
            self._logical_interval = np.asarray((float(logical_interval[0]), float(logical_interval[1])), dtype=float)

    def _rebuild(self, onefun: Any) -> CompactFun:
        """Construct a new :class:`CompactFun` preserving the logical interval."""
        return self.__class__(onefun, self._interval, logical_interval=self._logical_interval)

    # --------------------------
    #  alternative constructors
    # --------------------------
    @classmethod
    def initempty(cls) -> CompactFun:
        """Initialise an empty CompactFun on ``(-inf, +inf)``."""
        storage = Interval(-1.0, 1.0)
        onefun = techdict[prefs.tech].initempty(interval=storage)
        return cls(onefun, storage, logical_interval=(-np.inf, np.inf))

    @classmethod
    def initconst(cls, c: Any, interval: Any) -> CompactFun:
        """Initialise a constant function.

        Non-zero constants on unbounded intervals are not representable as
        :class:`CompactFun` because the function does not decay to zero at
        ``±inf``.
        """
        a, b = _ensure_endpoints(interval)
        unbounded = (not np.isfinite(a)) or (not np.isfinite(b))
        if unbounded and float(c) != 0.0:
            raise CompactFunConstructionError(  # noqa: TRY003
                "Non-zero constants are not representable as a CompactFun on an "
                "unbounded interval (a CompactFun must decay to zero at ±inf)."
            )
        if not np.isfinite(a) and not np.isfinite(b):
            storage = Interval(-1.0, 1.0)
        elif not np.isfinite(a):
            storage = Interval(b - 1.0, b)
        elif not np.isfinite(b):
            storage = Interval(a, a + 1.0)
        else:
            storage = Interval(a, b)
        onefun = techdict[prefs.tech].initconst(c, interval=storage)
        return cls(onefun, storage, logical_interval=(a, b))

    @classmethod
    def initidentity(cls, interval: Any) -> CompactFun:
        """Initialise the identity function ``f(x) = x``.

        The identity function is unbounded and so cannot be represented as a
        :class:`CompactFun` on an unbounded interval.  This method is provided
        only for completeness and refuses any infinite endpoint.
        """
        a, b = _ensure_endpoints(interval)
        if not (np.isfinite(a) and np.isfinite(b)):
            raise CompactFunConstructionError(  # noqa: TRY003
                "The identity function f(x) = x cannot be represented as a CompactFun on an unbounded interval."
            )
        storage = Interval(a, b)
        onefun = techdict[prefs.tech].initvalues(np.asarray(storage), interval=storage)
        return cls(onefun, storage, logical_interval=(a, b))

    @classmethod
    def initfun_adaptive(cls, f: Any, interval: Any) -> CompactFun:
        """Initialise from a callable using adaptive sampling.

        Discovers the numerical support of ``f`` on the (possibly unbounded)
        logical interval, then builds a standard adaptive :class:`Onefun` on
        that finite storage interval.

        Raises:
            CompactFunConstructionError: If the numerical support cannot be
                discovered within the configured tolerance and width budget.
        """
        a, b = _ensure_endpoints(interval)
        a_s, b_s = _discover_numsupp(
            f,
            a,
            b,
            prefs.numsupp_tol,
            prefs.numsupp_max_width,
            prefs.numsupp_max_probes,
        )
        storage = Interval(a_s, b_s)
        onefun = techdict[prefs.tech].initfun(lambda y: f(storage(y)), interval=storage)
        return cls(onefun, storage, logical_interval=(a, b))

    @classmethod
    def initfun_fixedlen(cls, f: Any, interval: Any, n: int) -> CompactFun:
        """Initialise from a callable using a fixed number of points.

        Discovers numerical support as in :meth:`initfun_adaptive`, then
        builds a fixed-length :class:`Onefun` on the storage interval.
        """
        a, b = _ensure_endpoints(interval)
        a_s, b_s = _discover_numsupp(
            f,
            a,
            b,
            prefs.numsupp_tol,
            prefs.numsupp_max_width,
            prefs.numsupp_max_probes,
        )
        storage = Interval(a_s, b_s)
        onefun = techdict[prefs.tech].initfun(lambda y: f(storage(y)), n, interval=storage)
        return cls(onefun, storage, logical_interval=(a, b))

    # -------------------
    #  evaluation
    # -------------------
    def __call__(self, x: Any, how: str = "clenshaw") -> Any:
        """Evaluate the function at ``x``; returns ``0`` outside the storage interval."""
        scalar_input = np.isscalar(x) or np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x))
        is_complex = bool(getattr(self.onefun, "iscomplex", False))
        result = np.zeros(x_arr.shape, dtype=complex if is_complex else float)
        a_s, b_s = self._interval
        mask = (x_arr >= a_s) & (x_arr <= b_s)
        if mask.any():
            y = self._interval.invmap(x_arr[mask])
            result[mask] = self.onefun(y, how)
        if scalar_input:
            return result.item()
        return result

    # ------------
    #  properties
    # ------------
    @property
    def support(self) -> Any:
        """Return the logical (user-facing) interval, possibly with ``±inf`` endpoints."""
        return self._logical_interval

    @property
    def numerical_support(self) -> Any:
        """Return the finite storage interval ``[a, b]`` discovered at construction."""
        return np.asarray(self._interval)

    @property
    def endvalues(self) -> Any:
        """Return values at the logical endpoints; ``0`` at any ``±inf`` endpoint."""
        a_log, b_log = float(self._logical_interval[0]), float(self._logical_interval[1])
        yl = 0.0 if not np.isfinite(a_log) else self.__call__(a_log)
        yr = 0.0 if not np.isfinite(b_log) else self.__call__(b_log)
        return np.array([yl, yr])

    def __repr__(self) -> str:  # pragma: no cover
        """Return a string representation showing the logical interval and size."""
        a_log, b_log = self._logical_interval
        return f"{self.__class__.__name__}([{a_log}, {b_log}], {self.size})"

    # ----------
    #  calculus
    # ----------
    def cumsum(self) -> Any:
        """Indefinite integral.

        ``cumsum`` does not close in :class:`CompactFun` because the
        antiderivative of an integrable function on ``(-inf, inf)`` does not
        decay to zero at ``+inf`` (it tends to ``∫f``).  Future releases will
        return a bounded :class:`Chebfun` representation; for now this is a
        documented limitation.
        """
        raise NotImplementedError(
            "CompactFun.cumsum() is not implemented in this release: the "
            "antiderivative of an integrable function on (-inf, inf) tends "
            "to a non-zero constant at +inf and so cannot itself be a "
            "CompactFun.  This is a documented v1 limitation; a future "
            "release will return a bounded Chebfun representation."
        )

    # -----------
    #  utilities
    # -----------
    def restrict(self, subinterval: Any) -> Any:
        """Restrict to a finite subinterval, returning a :class:`Bndfun`."""
        from .bndfun import Bndfun

        sub_a, sub_b = _ensure_endpoints(subinterval)
        if not (np.isfinite(sub_a) and np.isfinite(sub_b)):
            raise NotImplementedError(
                "CompactFun.restrict() requires a finite subinterval; "
                "restriction to unbounded subintervals is not supported."
            )
        return Bndfun.initfun_adaptive(self, Interval(sub_a, sub_b))

    def translate(self, c: float) -> CompactFun:
        """Translate by ``c`` along the real line, preserving both intervals."""
        new_storage = Interval(float(self._interval[0]) + c, float(self._interval[1]) + c)
        a_log, b_log = float(self._logical_interval[0]), float(self._logical_interval[1])
        new_logical = (a_log + c, b_log + c)
        return self.__class__(self.onefun, new_storage, logical_interval=new_logical)

    # ----------
    #  plotting
    # ----------
    @property
    def plot_support(self) -> tuple[float, float]:
        """Return a finite ``[a, b]`` plotting window.

        Replaces any ``±inf`` logical endpoint with the corresponding
        numerical-support endpoint padded by 10% of the storage width
        (minimum padding of 1.0) so the decay-to-zero region is visible.
        """
        a_s, b_s = float(self._interval[0]), float(self._interval[1])
        a_log, b_log = float(self._logical_interval[0]), float(self._logical_interval[1])
        pad = max(0.1 * (b_s - a_s), 1.0)
        a = a_log if np.isfinite(a_log) else a_s - pad
        b = b_log if np.isfinite(b_log) else b_s + pad
        return (a, b)

    def plot(self, ax: Any = None, **kwds: Any) -> Any:
        """Plot the function over a finite window derived from its numerical support.

        For doubly- or singly-infinite logical intervals, the plotting window
        defaults to the numerical-support interval padded by 10% on each
        unbounded side. Pass an explicit ``support=(a, b)`` keyword to override.
        """
        from .plotting import plotfun

        support = kwds.pop("support", self.plot_support)
        return plotfun(self, support, ax=ax, **kwds)
