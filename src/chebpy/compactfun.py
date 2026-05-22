"""Implementation of functions on (semi-)infinite intervals via numerical-support truncation.

This module provides the :class:`CompactFun` class, which sits next to
:class:`~chebpy.bndfun.Bndfun` under :class:`~chebpy.classicfun.Classicfun`.
It represents functions whose user-facing logical interval has one or both
endpoints at ``±inf`` but whose **numerical support** — the set of points
where the function differs from its asymptotic limit by more than a
configured tolerance — is finite.  Internally, a :class:`CompactFun` stores
a standard :class:`~chebpy.onefun.Onefun` (Chebtech) on the discovered
finite storage interval; outside that interval the function is reported as
the corresponding asymptotic constant (``tail_left`` or ``tail_right``,
default ``0``).

This approach is a deliberate departure from MATLAB Chebfun's ``@unbndfun``
(which uses a rational change of variables to map ``(-inf, inf)`` onto
``[-1, 1]``).  See ``docs/plans/02-compactfun-integration.md`` for the
zero-tail design and ``docs/plans/02b-compactfun-tail-constants.md`` for
the non-zero asymptote extension.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from .classicfun import Classicfun, techdict
from .exceptions import CompactFunConstructionError, DivergentIntegralError
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
) -> tuple[float, float, float]:
    """Discover the numerical-support boundary on one infinite side.

    Probes ``f`` at ``anchor + sign * 2**k`` for ``k = 0, 1, 2, ...`` up to
    the configured budget.  Detects the asymptotic limit ``L`` of ``f`` on
    this side (which may be zero or non-zero) and returns the smallest
    finite boundary beyond which ``|f - L| < tol * scale``.

    Args:
        f: Callable being approximated.
        anchor: Finite anchor point (the bounded endpoint of a semi-infinite
            interval, or ``0.0`` for the doubly-infinite case).
        sign: ``+1`` for the rightward (toward ``+inf``) side, ``-1`` for the
            leftward side.
        tol: Relative tolerance threshold for both convergence detection and
            boundary placement.
        max_width: Maximum permitted boundary distance from ``anchor``.
        max_probes: Maximum number of geometric probes.

    Returns:
        Tuple ``(boundary, tail, vscale)`` where ``boundary`` is the finite
        boundary, ``tail`` is the detected asymptotic constant (``0.0`` if
        the function decays to zero), and ``vscale`` is the largest
        absolute probed value on this side.

    Raises:
        CompactFunConstructionError: If ``f`` does not converge to a
            constant within the probing budget or ``max_width``.
    """
    radii: list[float] = []
    values: list[float] = []  # signed values
    r = 1.0
    for _ in range(max_probes):
        if r > max_width:
            break
        x = anchor + sign * r
        try:
            v = float(f(x))
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
        return anchor + sign * 1.0, 0.0, 0.0

    abs_values = [abs(v) for v in values]
    vscale = max(abs_values) if abs_values else 0.0

    # Need at least three probes to verify convergence to a constant.
    last_n = 3
    if len(values) < last_n:
        raise CompactFunConstructionError(  # noqa: TRY003
            f"Too few probes ({len(values)}) to determine the asymptotic "
            f"behaviour of f near {'+' if sign > 0 else '-'}inf; "
            f"increase numsupp_max_probes or numsupp_max_width."
        )

    # Convergence test: the last few signed probes must agree to tol*scale.
    tail_window = values[-last_n:]
    conv_threshold = tol * max(vscale, 1.0)
    spread = max(tail_window) - min(tail_window)
    if spread > conv_threshold:
        # Function does not settle to a constant — heavy tail or oscillation.
        raise CompactFunConstructionError(  # noqa: TRY003
            f"Function does not converge to a constant within "
            f"{radii[-1]:g} of anchor {anchor:g} on the "
            f"{'+' if sign > 0 else '-'}inf side (last {last_n} probes "
            f"spread by {spread:g} > {conv_threshold:g}); heavy-tailed or "
            f"oscillating inputs are not supported in this release."
        )
    tail = float(np.mean(tail_window))
    if abs(tail) < conv_threshold:
        tail = 0.0

    # Find the largest radius at which f is still "active" (above threshold
    # relative to the tail).
    threshold = tol * max(abs(tail), vscale, 1.0)
    active_r = 0.0
    for ri, vi in zip(radii, values, strict=False):
        if abs(vi - tail) > threshold:
            active_r = ri

    boundary_r = max(2.0 * active_r, 1.0)
    if boundary_r > max_width:
        raise CompactFunConstructionError(  # noqa: TRY003
            f"Discovered numerical support exceeds max_width = {max_width:g}; "
            f"heavy-tailed inputs are not supported in this release."
        )
    return anchor + sign * boundary_r, tail, vscale


def _discover_numsupp(
    f: Any, a: float, b: float, tol: float, max_width: float, max_probes: int
) -> tuple[float, float, float, float]:
    """Discover the storage interval and tail constants for ``f``.

    Args:
        f: Callable being approximated.
        a: Left endpoint of the logical interval (may be ``-inf``).
        b: Right endpoint of the logical interval (may be ``+inf``).
        tol: Relative tolerance for support detection.
        max_width: Maximum permitted storage interval width.
        max_probes: Maximum probes per unbounded side.

    Returns:
        Tuple ``(a', b', tail_left, tail_right)`` where ``a' < b'`` are
        finite floats and the tails are the detected asymptotic constants
        (``0.0`` on any side whose logical endpoint is finite).

    Raises:
        CompactFunConstructionError: If support cannot be discovered.
    """
    left_inf = not np.isfinite(a)
    right_inf = not np.isfinite(b)

    if not (left_inf or right_inf):
        return a, b, 0.0, 0.0

    # Anchor: the finite endpoint of a semi-infinite interval, else 0.
    if left_inf and right_inf:
        anchor = 0.0
    elif left_inf:
        anchor = b
    else:
        anchor = a

    if left_inf:
        a_storage, tail_left, _ = _discover_one_side(f, anchor, -1, tol, max_width, max_probes)
    else:
        a_storage, tail_left = a, 0.0

    if right_inf:
        b_storage, tail_right, _ = _discover_one_side(f, anchor, +1, tol, max_width, max_probes)
    else:
        b_storage, tail_right = b, 0.0

    if b_storage - a_storage > max_width:
        raise CompactFunConstructionError(  # noqa: TRY003
            f"Discovered numerical support [{a_storage:g}, {b_storage:g}] exceeds "
            f"max_width = {max_width:g}; heavy-tailed inputs are not supported "
            f"in this release."
        )
    if b_storage <= a_storage:
        # Function appears identically constant on both sides; pick a small
        # default storage interval.
        a_storage, b_storage = anchor - 1.0, anchor + 1.0
    return a_storage, b_storage, tail_left, tail_right


class CompactFun(Classicfun):
    """Functions on (semi-)infinite intervals with finite numerical support.

    A :class:`CompactFun` represents a function whose user-facing logical
    interval has one or both endpoints at ``±inf`` but whose numerical
    support — the set where the function differs from its asymptotic limit
    by more than a configured tolerance — is finite.  Internally it
    inherits from :class:`Classicfun` and stores a standard
    :class:`Onefun` on the discovered finite storage interval; outside that
    interval the function is reported as the corresponding asymptotic
    constant ``tail_left`` or ``tail_right`` (default ``0.0``).

    Two intervals are tracked:

    - ``self._interval`` (inherited): the finite storage interval where the
      underlying ``Onefun`` lives.
    - ``self._logical_interval``: the user-facing interval, which may have
      ``±inf`` endpoints; returned by :attr:`support`.

    Two scalar tail constants are tracked:

    - ``tail_left``: the value reported for ``x < a_storage`` when the
      logical-left endpoint is ``-inf``.
    - ``tail_right``: the value reported for ``x > b_storage`` when the
      logical-right endpoint is ``+inf``.

    For finite logical intervals the storage and logical intervals coincide
    and the tails are ignored, so a :class:`CompactFun` behaves identically
    to :class:`~chebpy.bndfun.Bndfun`.

    Attributes:
        onefun: Inherited; the standard :class:`Onefun` on ``[-1, 1]``.
        support: The logical interval (possibly with ``±inf`` endpoints).
        numerical_support: The finite storage interval.
        tail_left: Asymptotic value at ``-inf`` (``0.0`` if logical-left is finite).
        tail_right: Asymptotic value at ``+inf`` (``0.0`` if logical-right is finite).
    """

    def __init__(
        self,
        onefun: Any,
        interval: Any,
        logical_interval: Any = None,
        tail_left: float = 0.0,
        tail_right: float = 0.0,
    ) -> None:
        """Create a new :class:`CompactFun` instance.

        Args:
            onefun: The :class:`Onefun` representing the function on ``[-1, 1]``.
            interval: The finite storage :class:`Interval` (always finite).
            logical_interval: The user-facing interval (possibly with ``±inf``
                endpoints).  Defaults to ``interval`` if omitted.
            tail_left: Asymptotic value at ``-inf``.  Default ``0.0``.
            tail_right: Asymptotic value at ``+inf``.  Default ``0.0``.
        """
        super().__init__(onefun, interval)
        if logical_interval is None:
            self._logical_interval = np.asarray(interval, dtype=float)
        else:
            self._logical_interval = np.asarray((float(logical_interval[0]), float(logical_interval[1])), dtype=float)
        self._tail_left = float(tail_left)
        self._tail_right = float(tail_right)

    def _rebuild(self, onefun: Any, *, tail_left: float | None = None, tail_right: float | None = None) -> CompactFun:
        """Construct a new :class:`CompactFun` preserving logical interval and tails.

        Args:
            onefun: Replacement :class:`Onefun` for the new instance.
            tail_left: Optional override for the new instance's left tail.
                Defaults to ``self.tail_left``.
            tail_right: Optional override for the new instance's right tail.
                Defaults to ``self.tail_right``.
        """
        new_tl = self._tail_left if tail_left is None else float(tail_left)
        new_tr = self._tail_right if tail_right is None else float(tail_right)
        return self.__class__(
            onefun,
            self._interval,
            logical_interval=self._logical_interval,
            tail_left=new_tl,
            tail_right=new_tr,
        )

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

        On an unbounded interval the constant ``c`` becomes the asymptotic
        value on each unbounded side: ``tail_left = tail_right = c``.  This
        makes ``initconst`` total — every constant is representable on every
        interval — but note that integrating a non-zero constant over an
        unbounded logical interval will (correctly) raise
        :class:`~chebpy.exceptions.DivergentIntegralError`.
        """
        a, b = _ensure_endpoints(interval)
        c_val = float(c)
        if not np.isfinite(a) and not np.isfinite(b):
            storage = Interval(-1.0, 1.0)
        elif not np.isfinite(a):
            storage = Interval(b - 1.0, b)
        elif not np.isfinite(b):
            storage = Interval(a, a + 1.0)
        else:
            storage = Interval(a, b)
        onefun = techdict[prefs.tech].initconst(c_val, interval=storage)
        tail_left = c_val if not np.isfinite(a) else 0.0
        tail_right = c_val if not np.isfinite(b) else 0.0
        return cls(onefun, storage, logical_interval=(a, b), tail_left=tail_left, tail_right=tail_right)

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

        Discovers the numerical support and asymptotic tail constants of
        ``f`` on the (possibly unbounded) logical interval, then builds a
        standard adaptive :class:`Onefun` on that finite storage interval.

        Raises:
            CompactFunConstructionError: If the numerical support cannot be
                discovered within the configured tolerance and width budget,
                or if ``f`` does not converge to a constant at ``±inf``.
        """
        a, b = _ensure_endpoints(interval)
        a_s, b_s, tl, tr = _discover_numsupp(
            f,
            a,
            b,
            prefs.numsupp_tol,
            prefs.numsupp_max_width,
            prefs.numsupp_max_probes,
        )
        storage = Interval(a_s, b_s)
        onefun = techdict[prefs.tech].initfun(lambda y: f(storage(y)), interval=storage)
        return cls(onefun, storage, logical_interval=(a, b), tail_left=tl, tail_right=tr)

    @classmethod
    def initfun_fixedlen(cls, f: Any, interval: Any, n: int) -> CompactFun:
        """Initialise from a callable using a fixed number of points.

        Discovers numerical support and tails as in :meth:`initfun_adaptive`,
        then builds a fixed-length :class:`Onefun` on the storage interval.
        """
        a, b = _ensure_endpoints(interval)
        a_s, b_s, tl, tr = _discover_numsupp(
            f,
            a,
            b,
            prefs.numsupp_tol,
            prefs.numsupp_max_width,
            prefs.numsupp_max_probes,
        )
        storage = Interval(a_s, b_s)
        onefun = techdict[prefs.tech].initfun(lambda y: f(storage(y)), n, interval=storage)
        return cls(onefun, storage, logical_interval=(a, b), tail_left=tl, tail_right=tr)

    # -------------------
    #  evaluation
    # -------------------
    def __call__(self, x: Any, how: str = "clenshaw") -> Any:
        """Evaluate the function at ``x``.

        Outside the storage interval, returns the corresponding tail constant
        when the matching logical endpoint is ``±inf`` (default ``0.0``), or
        ``0.0`` when the logical endpoint is finite.
        """
        scalar_input = np.isscalar(x) or np.ndim(x) == 0
        x_arr = np.atleast_1d(np.asarray(x))
        is_complex = bool(getattr(self.onefun, "iscomplex", False))
        result = np.zeros(x_arr.shape, dtype=complex if is_complex else float)
        a_s, b_s = self._interval
        a_log, b_log = float(self._logical_interval[0]), float(self._logical_interval[1])
        # Outside-storage values: tail constants where the logical edge is ±inf.
        left_mask = x_arr < a_s
        right_mask = x_arr > b_s
        if not np.isfinite(a_log) and self._tail_left != 0.0:
            result[left_mask] = self._tail_left
        if not np.isfinite(b_log) and self._tail_right != 0.0:
            result[right_mask] = self._tail_right
        # Inside-storage values: standard onefun evaluation.
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
    def tail_left(self) -> float:
        """Asymptotic value of the function as ``x → -inf``.

        Always ``0.0`` when the logical-left endpoint is finite.
        """
        return self._tail_left

    @property
    def tail_right(self) -> float:
        """Asymptotic value of the function as ``x → +inf``.

        Always ``0.0`` when the logical-right endpoint is finite.
        """
        return self._tail_right

    @property
    def endvalues(self) -> Any:
        """Return values at the logical endpoints; tails at any ``±inf`` endpoint."""
        a_log, b_log = float(self._logical_interval[0]), float(self._logical_interval[1])
        yl = self._tail_left if not np.isfinite(a_log) else self.__call__(a_log)
        yr = self._tail_right if not np.isfinite(b_log) else self.__call__(b_log)
        return np.array([yl, yr])

    def __repr__(self) -> str:  # pragma: no cover
        """Return a string representation showing the logical interval, size, and tails."""
        a_log, b_log = self._logical_interval
        if self._tail_left != 0.0 or self._tail_right != 0.0:
            return (
                f"{self.__class__.__name__}([{a_log}, {b_log}], {self.size}, "
                f"tails=({self._tail_left}, {self._tail_right}))"
            )
        return f"{self.__class__.__name__}([{a_log}, {b_log}], {self.size})"

    # ----------
    #  calculus
    # ----------
    def sum(self) -> Any:
        """Compute the definite integral over the logical interval.

        Raises:
            DivergentIntegralError: If the logical interval is unbounded on
                a side where the corresponding tail is non-zero (the integral
                of a non-decaying function over a half-line diverges).
        """
        a_log, b_log = float(self._logical_interval[0]), float(self._logical_interval[1])
        if (not np.isfinite(a_log)) and self._tail_left != 0.0:
            raise DivergentIntegralError(  # noqa: TRY003
                f"Integrand has non-zero left asymptote tail_left={self._tail_left}; "
                f"integral over (-inf, ...) diverges."
            )
        if (not np.isfinite(b_log)) and self._tail_right != 0.0:
            raise DivergentIntegralError(  # noqa: TRY003
                f"Integrand has non-zero right asymptote tail_right={self._tail_right}; "
                f"integral over (..., +inf) diverges."
            )
        return super().sum()

    def cumsum(self) -> CompactFun:
        """Compute the indefinite integral.

        For a :class:`CompactFun` with zero asymptote on the unbounded
        left/right side, the antiderivative is well-defined; it is itself a
        :class:`CompactFun` whose right-tail equals ``∫f`` and whose
        left-tail is ``0`` (anchored so ``F(-inf) = 0``).

        Raises:
            DivergentIntegralError: If the logical interval is unbounded on
                a side where the corresponding tail is non-zero, in which
                case the antiderivative diverges.
        """
        a_log, b_log = float(self._logical_interval[0]), float(self._logical_interval[1])
        if (not np.isfinite(a_log)) and self._tail_left != 0.0:
            raise DivergentIntegralError(  # noqa: TRY003
                f"Antiderivative diverges at -inf because tail_left={self._tail_left} != 0."
            )
        if (not np.isfinite(b_log)) and self._tail_right != 0.0:
            raise DivergentIntegralError(  # noqa: TRY003
                f"Antiderivative diverges at +inf because tail_right={self._tail_right} != 0."
            )
        # Standard cumsum on the storage interval anchors F(a_storage) = 0.
        # When logical-left is -inf with tail_left=0, this approximates
        # F(-inf) = 0 (since f is below tolerance below a_storage).
        inner = super().cumsum()
        # The right-tail of F is the total integral.
        total = float(super().sum())
        # The left-tail is 0 when logical-left is -inf (anchor at -inf).
        new_tail_left = 0.0
        new_tail_right = total
        return self.__class__(
            inner.onefun,
            inner._interval,
            logical_interval=self._logical_interval,
            tail_left=new_tail_left,
            tail_right=new_tail_right,
        )

    def diff(self) -> CompactFun:
        """Compute the derivative.

        The derivative of a function with constant asymptotic limits has
        zero asymptotes, so the result has ``tail_left = tail_right = 0``.
        """
        result = cast(CompactFun, super().diff())
        result._tail_left = 0.0
        result._tail_right = 0.0
        return result

    # -------------
    #  rootfinding
    # -------------
    def roots(self) -> Any:
        """Find the roots, filtering out spurious roots in numerical-noise regions.

        The underlying polynomial approximation can produce many spurious
        roots in regions where the function has decayed to numerical noise
        (typically near the boundary of the storage interval).  We keep a
        candidate root ``r`` only if both:

        - ``f(r - δ)`` and ``f(r + δ)`` have opposite signs (the function
          actually crosses zero), **and**
        - ``max(|f(r - δ)|, |f(r + δ)|)`` exceeds ``numsupp_tol * vscale``
          (the values are above numerical noise).

        Here ``delta = 1e-3 * storage_width``.  This heuristic does not preserve
        double roots; that is a documented limitation since double roots are
        uncommon in the decay-to-zero functions that :class:`CompactFun` is
        designed for.
        """
        raw = super().roots()
        if raw.size == 0:
            return raw
        a_s, b_s = float(self._interval[0]), float(self._interval[1])
        vals = np.abs(np.atleast_1d(self.onefun.values()))
        vscale = float(vals.max()) if vals.size else 1.0
        threshold = prefs.numsupp_tol * max(vscale, 1.0)
        delta = 1e-3 * (b_s - a_s)
        left = np.clip(raw - delta, a_s, b_s)
        right = np.clip(raw + delta, a_s, b_s)
        f_left = np.atleast_1d(self.__call__(left))
        f_right = np.atleast_1d(self.__call__(right))
        sign_flip = np.sign(f_left) != np.sign(f_right)
        above_noise = np.maximum(np.abs(f_left), np.abs(f_right)) > threshold
        keep = sign_flip & above_noise
        return np.sort(np.unique(raw[keep]))

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
        """Translate by ``c`` along the real line, preserving both intervals and tails."""
        new_storage = Interval(float(self._interval[0]) + c, float(self._interval[1]) + c)
        a_log, b_log = float(self._logical_interval[0]), float(self._logical_interval[1])
        new_logical = (a_log + c, b_log + c)
        return self.__class__(
            self.onefun,
            new_storage,
            logical_interval=new_logical,
            tail_left=self._tail_left,
            tail_right=self._tail_right,
        )

    # ------------
    #  arithmetic
    # ------------
    def __neg__(self) -> CompactFun:
        """Return ``-f``; negates both tail constants."""
        result = cast(CompactFun, super().__neg__())
        result._tail_left = -self._tail_left
        result._tail_right = -self._tail_right
        return result

    def __add__(self, other: Any) -> Any:
        """Pointwise addition; combines tail constants additively."""
        result = super().__add__(other)
        if isinstance(result, CompactFun):
            other_tl, other_tr = self._other_tails(other)
            result._tail_left = self._tail_left + other_tl
            result._tail_right = self._tail_right + other_tr
        return result

    def __radd__(self, other: Any) -> Any:
        """Right-hand addition for scalar + CompactFun."""
        result = super().__radd__(other)
        if isinstance(result, CompactFun):
            other_tl, other_tr = self._other_tails(other)
            result._tail_left = self._tail_left + other_tl
            result._tail_right = self._tail_right + other_tr
        return result

    def __sub__(self, other: Any) -> Any:
        """Pointwise subtraction; combines tail constants additively."""
        result = super().__sub__(other)
        if isinstance(result, CompactFun):
            other_tl, other_tr = self._other_tails(other)
            result._tail_left = self._tail_left - other_tl
            result._tail_right = self._tail_right - other_tr
        return result

    def __rsub__(self, other: Any) -> Any:
        """Right-hand subtraction for scalar - CompactFun."""
        result = super().__rsub__(other)
        if isinstance(result, CompactFun):
            other_tl, other_tr = self._other_tails(other)
            result._tail_left = other_tl - self._tail_left
            result._tail_right = other_tr - self._tail_right
        return result

    def __mul__(self, other: Any) -> Any:
        """Pointwise multiplication; combines tail constants multiplicatively."""
        result = super().__mul__(other)
        if isinstance(result, CompactFun):
            other_tl, other_tr = self._other_tails(other)
            result._tail_left = self._tail_left * other_tl
            result._tail_right = self._tail_right * other_tr
        return result

    def __rmul__(self, other: Any) -> Any:
        """Right-hand multiplication for scalar * CompactFun."""
        result = super().__rmul__(other)
        if isinstance(result, CompactFun):
            other_tl, other_tr = self._other_tails(other)
            result._tail_left = self._tail_left * other_tl
            result._tail_right = self._tail_right * other_tr
        return result

    def _other_tails(self, other: Any) -> tuple[float, float]:
        """Extract ``(tail_left, tail_right)`` from a binary-op operand.

        For a :class:`CompactFun` operand, returns its tail attributes; for
        a scalar, returns ``(scalar, scalar)``.
        """
        if isinstance(other, CompactFun):
            return other._tail_left, other._tail_right
        if np.isscalar(other):
            v = float(other)
            return v, v
        # Anything else (e.g. a different Classicfun subclass) is treated as
        # zero-tailed; tail propagation may be inexact in that case.
        return 0.0, 0.0

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
