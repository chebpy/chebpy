"""Non-affine bijective maps between [-1, 1] and a logical interval [a, b].

These maps are concrete implementations of the
:class:`~chebpy.utilities.IntervalMap` protocol that cluster grid points
exponentially towards one or both endpoints. They are intended for
representing functions with branch-type / logarithmic endpoint
singularities, following the spirit of Adcock & Richardson,
*New exponential variable transform methods for functions with endpoint
singularities* (arXiv:1305.2643).

Two families are provided:

* :class:`SingleSlitMap` — exponential clustering at one chosen endpoint
  (``side='left'`` or ``side='right'``). The map's derivative vanishes
  super-algebraically at the chosen endpoint, so a function with an
  algebraic or logarithmic singularity at that endpoint is rendered
  analytic in the reference variable ``t`` and is resolved by ordinary
  Chebyshev interpolation.

* :class:`DoubleSlitMap` — exponential clustering at both endpoints. The
  derivative vanishes super-algebraically at ``t = ±1``.

Both classes are pure-Python, NumPy-vectorised, and stateless apart from
their constructor parameters; they carry no `Onefun` payload.
"""

from typing import Any

import numpy as np


def _as_array(x: float | np.ndarray) -> tuple[np.ndarray, bool]:
    """Coerce ``x`` to a numpy array, remembering whether the input was scalar.

    Returns:
        tuple[numpy.ndarray, bool]: A ``(array, was_scalar)`` pair. ``was_scalar``
            is ``True`` if ``x`` is a 0-d / Python scalar input, in which case
            callers should ``.item()`` the result before returning.
    """
    arr = np.asarray(x, dtype=float)
    return arr, arr.ndim == 0


class SingleSlitMap:
    """Exponential clustering map with a single clustered endpoint.

    Maps the reference interval ``[-1, 1]`` to a logical interval ``[a, b]``
    via an exponential transform that crowds grid points towards one chosen
    endpoint (``side='left'`` clusters near ``a``; ``side='right'`` clusters
    near ``b``). The map's derivative vanishes to all orders at the
    clustered endpoint, so a function with an algebraic or logarithmic
    singularity at that endpoint becomes analytic in ``t``.

    Mathematical form (with ``side='left'``)::

        q(t) = (1 - t) / (1 + t)        # (-1, 1) -> (∞, 0)
        g(t) = exp(-alpha * q(t))           # (-1, 1) -> (0, 1)
        x    = a + (b - a) * g(t)

    For ``side='right'`` the analogous form ``q(t) = (1 + t) / (1 - t)``
    and ``g(t) = 1 - exp(-alpha * q(t))`` is used.

    Args:
        a (float): Left endpoint of the logical interval.
        b (float): Right endpoint of the logical interval (must satisfy ``b > a``).
        alpha (float): Positive clustering strength. Larger ``alpha`` gives
            faster decay of the map's derivative at the clustered endpoint.
        side (str): Either ``"left"`` or ``"right"``.

    Raises:
        ValueError: If ``b <= a``, ``alpha <= 0``, or ``side`` is not one of
            the supported values.
    """

    def __init__(self, a: float, b: float, alpha: float = 1.0, side: str = "left") -> None:
        """Initialise a single-sided exponential clustering map.

        See the class docstring for parameter descriptions.
        """
        if not (b > a):
            msg = "require b > a"
            raise ValueError(msg)
        if not (alpha > 0):
            msg = "require alpha > 0"
            raise ValueError(msg)
        if side not in ("left", "right"):
            msg = "side must be 'left' or 'right'"
            raise ValueError(msg)
        self.a = float(a)
        self.b = float(b)
        self.alpha = float(alpha)
        self.side = side

    # ------------
    #  properties
    # ------------
    @property
    def support(self) -> tuple[float, float]:
        """Return the logical support ``(a, b)`` as a plain tuple."""
        return (self.a, self.b)

    # --------------
    #  IntervalMap
    # --------------
    def formap(self, y: float | np.ndarray) -> Any:
        """Map ``y ∈ [-1, 1]`` to ``x ∈ [a, b]`` with exponential endpoint clustering."""
        t, scalar = _as_array(y)
        # Clamp the endpoint that would produce 0/0 in ``q``; the limit is well-defined.
        if self.side == "left":
            # q = (1 - t) / (1 + t); at t = -1, q = +inf -> g = 0 -> x = a.
            tt = np.where(t <= -1.0, -1.0, t)
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where(tt <= -1.0, np.inf, (1.0 - tt) / (1.0 + tt))
            g = np.exp(-self.alpha * q)
        else:  # side == "right"
            # q = (1 + t) / (1 - t); at t = 1, q = +inf -> g = 1 -> x = b.
            tt = np.where(t >= 1.0, 1.0, t)
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where(tt >= 1.0, np.inf, (1.0 + tt) / (1.0 - tt))
            g = 1.0 - np.exp(-self.alpha * q)
        x = self.a + (self.b - self.a) * g
        if scalar:
            return float(x)
        return x

    def invmap(self, x: float | np.ndarray) -> Any:
        """Map ``x ∈ [a, b]`` back to ``y ∈ [-1, 1]`` (analytical inverse)."""
        xa, scalar = _as_array(x)
        u = (xa - self.a) / (self.b - self.a)
        # Clip ``u`` strictly into (0, 1) at the open boundary that produces a
        # log of 0; the corresponding ``t`` is the matching ±1 endpoint.
        if self.side == "left":
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where(u <= 0.0, np.inf, -np.log(np.where(u <= 0.0, 1.0, u)) / self.alpha)
            t = np.where(np.isinf(q), -1.0, (1.0 - q) / (1.0 + q))
        else:  # side == "right"
            with np.errstate(divide="ignore", invalid="ignore"):
                one_minus_u = 1.0 - u
                q = np.where(
                    one_minus_u <= 0.0,
                    np.inf,
                    -np.log(np.where(one_minus_u <= 0.0, 1.0, one_minus_u)) / self.alpha,
                )
            t = np.where(np.isinf(q), 1.0, (q - 1.0) / (q + 1.0))
        if scalar:
            return float(t)
        return t

    def drvmap(self, y: float | np.ndarray) -> Any:
        """Return ``dx/dy`` of :meth:`formap` evaluated at ``y``."""
        t, scalar = _as_array(y)
        scale = self.b - self.a
        if self.side == "left":
            # x' = (b-a) * g(t) * alpha * 2 / (1 + t)^2
            tt = np.where(t <= -1.0, -1.0, t)
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where(tt <= -1.0, np.inf, (1.0 - tt) / (1.0 + tt))
                g = np.exp(-self.alpha * q)
                dq = np.where(tt <= -1.0, 0.0, -2.0 / (1.0 + tt) ** 2)
            # g * (-alpha * dq) is finite at t = -1 because g * q = 0 there (exp wins).
            dxdy = scale * g * (-self.alpha * dq)
            # Tidy up the limit at the clustered endpoint: g → 0 dominates dq → ∞.
            dxdy = np.where(tt <= -1.0, 0.0, dxdy)
        else:  # side == "right"
            tt = np.where(t >= 1.0, 1.0, t)
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.where(tt >= 1.0, np.inf, (1.0 + tt) / (1.0 - tt))
                expmaq = np.exp(-self.alpha * q)
                dq = np.where(tt >= 1.0, 0.0, 2.0 / (1.0 - tt) ** 2)
            dxdy = scale * expmaq * (self.alpha * dq)
            dxdy = np.where(tt >= 1.0, 0.0, dxdy)
        if scalar:
            return float(dxdy)
        return dxdy

    def __repr__(self) -> str:  # pragma: no cover - trivial
        """Return a developer-friendly representation."""
        return f"SingleSlitMap(a={self.a!r}, b={self.b!r}, alpha={self.alpha!r}, side={self.side!r})"


class DoubleSlitMap:
    """Exponential clustering map with both endpoints clustered.

    Maps the reference interval ``[-1, 1]`` to a logical interval ``[a, b]``
    via a symmetric tanh-style exponential transform whose derivative
    vanishes to all orders at both endpoints ``t = ±1``. Suitable for
    representing functions with simultaneous branch-type singularities at
    ``a`` and ``b`` (e.g. ``√((x - a)(b - x))``, ``log((x - a)(b - x))``).

    Mathematical form::

        s(t) = t / √(1 - t^2)            # (-1, 1) -> R
        g(t) = (1 + tanh(alpha * s(t))) / 2  # (-1, 1) -> (0, 1)
        x    = a + (b - a) * g(t)

    Args:
        a (float): Left endpoint of the logical interval.
        b (float): Right endpoint of the logical interval (must satisfy ``b > a``).
        alpha (float): Positive clustering strength.

    Raises:
        ValueError: If ``b <= a`` or ``alpha <= 0``.
    """

    def __init__(self, a: float, b: float, alpha: float = 1.0) -> None:
        """Initialise a symmetric double-sided exponential clustering map.

        See the class docstring for parameter descriptions.
        """
        if not (b > a):
            msg = "require b > a"
            raise ValueError(msg)
        if not (alpha > 0):
            msg = "require alpha > 0"
            raise ValueError(msg)
        self.a = float(a)
        self.b = float(b)
        self.alpha = float(alpha)

    @property
    def support(self) -> tuple[float, float]:
        """Return the logical support ``(a, b)`` as a plain tuple."""
        return (self.a, self.b)

    def formap(self, y: float | np.ndarray) -> Any:
        """Map ``y ∈ [-1, 1]`` to ``x ∈ [a, b]`` with both endpoints clustered."""
        t, scalar = _as_array(y)
        tt = np.clip(t, -1.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.sqrt(np.maximum(1.0 - tt * tt, 0.0))
            s = np.where(denom > 0.0, tt / np.where(denom > 0.0, denom, 1.0), np.inf * np.sign(tt))
        g = 0.5 * (1.0 + np.tanh(self.alpha * s))
        # Endpoint cleanup: t = -1 -> g = 0; t = +1 -> g = 1.
        g = np.where(tt <= -1.0, 0.0, g)
        g = np.where(tt >= 1.0, 1.0, g)
        x = self.a + (self.b - self.a) * g
        if scalar:
            return float(x)
        return x

    def invmap(self, x: float | np.ndarray) -> Any:
        """Map ``x ∈ [a, b]`` back to ``y ∈ [-1, 1]`` (analytical inverse)."""
        xa, scalar = _as_array(x)
        u = (xa - self.a) / (self.b - self.a)
        # 2u - 1 = tanh(alpha * s); s = (1 / (2alpha)) * log(u / (1 - u))
        with np.errstate(divide="ignore", invalid="ignore"):
            num = np.where(u <= 0.0, 0.0, u)
            den = np.where(u >= 1.0, 1.0, 1.0 - u)
            s = np.where(
                (u <= 0.0) | (u >= 1.0),
                np.where(u <= 0.0, -np.inf, np.inf),
                np.log(num / den) / (2.0 * self.alpha),
            )
        # t = s / sqrt(1 + s^2); finite even when s -> ±∞ (gives ±1).
        with np.errstate(invalid="ignore"):
            t = np.where(np.isfinite(s), s / np.sqrt(1.0 + np.where(np.isfinite(s), s * s, 0.0)), np.sign(s))
        if scalar:
            return float(t)
        return t

    def drvmap(self, y: float | np.ndarray) -> Any:
        """Return ``dx/dy`` of :meth:`formap` evaluated at ``y``."""
        t, scalar = _as_array(y)
        tt = np.clip(t, -1.0, 1.0)
        # s'(t) = 1 / (1 - t^2)^(3/2); g'(t) = (alpha / 2) * sech^2(alpha * s) * s'(t)
        one_minus_t2 = np.maximum(1.0 - tt * tt, 0.0)
        denom = one_minus_t2**1.5
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(one_minus_t2 > 0.0, tt / np.sqrt(np.where(one_minus_t2 > 0.0, one_minus_t2, 1.0)), 0.0)
            sech2 = 1.0 / np.cosh(self.alpha * s) ** 2
            sprime = np.where(denom > 0.0, 1.0 / np.where(denom > 0.0, denom, 1.0), 0.0)
        dxdy = (self.b - self.a) * 0.5 * self.alpha * sech2 * sprime
        # At t = ±1 the product sech^2(alpha*s) * s'(t) is 0*∞; sech^2 decays double-exponentially
        # in s and dominates the algebraic blow-up of s', so the limit is 0.
        dxdy = np.where(one_minus_t2 <= 0.0, 0.0, dxdy)
        if scalar:
            return float(dxdy)
        return dxdy

    def __repr__(self) -> str:  # pragma: no cover - trivial
        """Return a developer-friendly representation."""
        return f"DoubleSlitMap(a={self.a!r}, b={self.b!r}, alpha={self.alpha!r})"
