"""Non-affine bijective maps between [-1, 1] and a logical interval [a, b].

These maps are concrete implementations of the
:class:`~chebpy.utilities.IntervalMap` protocol that cluster grid points
exponentially towards one or both endpoints, following the slit-strip
constructions of Adcock & Richardson, *New exponential variable transform
methods for functions with endpoint singularities*, SIAM J. Numer. Anal.
52(4), 1887–1912, 2014 (doi:10.1137/130920460; arXiv:1305.2643).

Two families are provided:

* :class:`SingleSlitMap` — semi-infinite slit-strip map ``phi_S`` of
  Adcock & Richardson. Exponential clustering at one chosen endpoint
  (``side='left'`` clusters near ``a``; ``side='right'`` near ``b``).

* :class:`DoubleSlitMap` — infinite two-slit-strip map ``psi_S`` of
  Adcock & Richardson. Exponential clustering at both endpoints.

Each map is parameterised by a :class:`MapParams` ``(L, alpha)`` pair,
where ``alpha > 0`` is the strip half-width and ``L > 0`` is the
truncation length applied to the underlying conformal map. The forward
map composes an affine scaling with the truncated paper map; with
finite ``L`` the image of ``[-1, 1]`` under :meth:`formap` falls a
distance ``gap`` short of the clustered endpoint(s). With the default
``L = 8.0`` the gap is below ``1e-10`` and is invisible at working
precision; smaller ``L`` (closer to the paper's empirical optimum
``L ~ 1``) shrinks the resolved interval visibly but improves the
convergence rate of the mapped Chebyshev expansion. See
:attr:`SingleSlitMap.gap` / :attr:`DoubleSlitMap.gap` for the exact
shortfall.

Both classes are pure-Python, NumPy-vectorised, and stateless apart from
their constructor parameters; they carry no ``Onefun`` payload.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class MapParams:
    """Parameters for the slit-strip clustering maps.

    Args:
        L: Positive truncation length. With finite ``L`` the map's image
            falls short of the clustered endpoint(s) by a tiny ``gap``;
            larger ``L`` shrinks the gap (default ``L = 8.0`` gives
            ``gap < 1e-10`` at ``alpha = 1.0``).
        alpha: Positive strip half-width. Controls the clustering
            strength; smaller ``alpha`` clusters more aggressively.
    """

    L: float = 8.0
    alpha: float = 1.0

    def __post_init__(self) -> None:
        """Validate that ``L`` and ``alpha`` are strictly positive."""
        if not (self.L > 0):
            msg = "require L > 0"
            raise ValueError(msg)
        if not (self.alpha > 0):
            msg = "require alpha > 0"
            raise ValueError(msg)


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
    """Paper-faithful semi-infinite slit-strip map ``phi_S``.

    Maps the reference interval ``[-1, 1]`` to (approximately) the
    logical interval ``[a, b]`` via the composition of an affine scaling
    and the inverse semi-infinite slit-strip map of Adcock & Richardson
    (arXiv:1305.2643). The map's derivative vanishes
    super-algebraically at the chosen clustered endpoint, so a function
    with an algebraic or logarithmic singularity at that endpoint becomes
    analytic in the reference variable ``t``.

    Mathematical form (with ``side='left'``)::

        s(y)        = L * (y - 1) / 2                      # [-1, 1] -> [-L, 0]
        gamma       = (alpha / pi) * log(exp(pi / alpha) - 1)
        u(s)        = (alpha / pi) * log(1 + exp(pi * (s + gamma) / alpha))
        x           = a + (b - a) * u(s(y))

    The shift ``gamma`` is chosen so that ``u(0) = 1``; ``u(-L)`` is a
    small positive number :attr:`gap_unit`, equal to the fraction of the
    interval ``[a, b]`` that is *not* covered by ``formap([-1, 1])``.

    For ``side='right'`` the analogous reflection ``x = b - (b - a) *
    u(-s(y))`` is used so the cluster lies at ``b``.

    Args:
        a: Left endpoint of the logical interval.
        b: Right endpoint of the logical interval (must satisfy ``b > a``).
        params: A :class:`MapParams` instance carrying ``(L, alpha)``.
            If ``None``, ``MapParams()`` defaults are used.
        side: Either ``"left"`` or ``"right"``.

    Raises:
        ValueError: If ``b <= a`` or ``side`` is not one of the supported
            values.
    """

    def __init__(
        self,
        a: float,
        b: float,
        params: MapParams | None = None,
        *,
        side: str = "left",
    ) -> None:
        """Initialise a semi-infinite slit-strip clustering map.

        See the class docstring for parameter descriptions.
        """
        if not (b > a):
            msg = "require b > a"
            raise ValueError(msg)
        if side not in ("left", "right"):
            msg = "side must be 'left' or 'right'"
            raise ValueError(msg)
        self.a = float(a)
        self.b = float(b)
        self.params = params if params is not None else MapParams()
        self.side = side
        # Pre-compute the shift so that u(0) = 1 exactly.
        a_p = self.params.alpha
        # gamma = (alpha/pi) * log(exp(pi/alpha) - 1); log1p(-exp(-pi/alpha)) is the
        # numerically stable form of log(exp(pi/alpha) - 1) - pi/alpha.
        pi_over_alpha = np.pi / a_p
        self._gamma = (a_p / np.pi) * (pi_over_alpha + np.log1p(-np.exp(-pi_over_alpha)))

    # Conveniences ------------------------------------------------------
    @property
    def alpha(self) -> float:
        """Strip half-width ``alpha`` (equal to ``self.params.alpha``)."""
        return self.params.alpha

    @property
    def L(self) -> float:
        """Truncation length ``L`` (equal to ``self.params.L``)."""
        return self.params.L

    @property
    def support(self) -> tuple[float, float]:
        """Return the *nominal* logical support ``(a, b)`` as a plain tuple.

        This is the interval the user requested; the actual image of
        :meth:`formap` falls short by :attr:`gap` at the clustered
        endpoint (negligible at the default ``L = 8``).
        """
        return (self.a, self.b)

    @property
    def gap_unit(self) -> float:
        """Fraction of ``[0, 1]`` not covered by the underlying truncated map.

        Equal to ``u(-L) = (alpha / pi) * log(1 + exp(pi * (-L + gamma) / alpha))``;
        this is the unit-interval shortfall before the affine scaling to
        ``[a, b]``.
        """
        a_p = self.params.alpha
        z = np.pi * (-self.params.L + self._gamma) / a_p
        return float((a_p / np.pi) * np.logaddexp(0.0, z))

    @property
    def gap(self) -> float:
        """Distance between :meth:`formap` ``(-1)`` (or ``(+1)``) and the clustered endpoint."""
        return (self.b - self.a) * self.gap_unit

    # Internal helpers --------------------------------------------------
    def _u_of_s(self, s: np.ndarray) -> np.ndarray:
        """Compute ``u(s) = (alpha/pi) * log(1 + exp(pi(s+gamma)/alpha))`` stably."""
        a_p = self.params.alpha
        z = np.pi * (s + self._gamma) / a_p
        return (a_p / np.pi) * np.logaddexp(0.0, z)

    def _du_ds(self, s: np.ndarray) -> np.ndarray:
        """Derivative ``du/ds = sigmoid(pi(s+gamma)/alpha)``."""
        a_p = self.params.alpha
        z = np.pi * (s + self._gamma) / a_p
        # 1 / (1 + exp(-z)), numerically stable.
        return np.where(z >= 0.0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    def _s_of_u(self, u: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`_u_of_s`. Maps ``u in (0, 1]`` to ``s in (-inf, 0]``.

        Uses the identity ``log(exp(w) - 1) = w + log1p(-exp(-w))`` for
        ``w = pi * u / alpha`` to retain precision near ``u = 1``.
        """
        a_p = self.params.alpha
        w = np.pi * u / a_p
        # log(exp(w) - 1) = w + log1p(-exp(-w)).
        log_em1 = w + np.log1p(-np.exp(-w))
        return (a_p / np.pi) * log_em1 - self._gamma

    # IntervalMap protocol ---------------------------------------------
    def formap(self, y: float | np.ndarray) -> Any:
        """Map ``y in [-1, 1]`` to ``x`` clustered near the chosen endpoint."""
        t, scalar = _as_array(y)
        L = self.params.L
        if self.side == "left":
            s = L * (t - 1.0) * 0.5  # [-1, 1] -> [-L, 0]
            u = self._u_of_s(s)
            x = self.a + (self.b - self.a) * u
        else:  # side == "right": reflect.
            s = L * (-t - 1.0) * 0.5  # [-1, 1] -> [0, -L]
            u = self._u_of_s(s)
            x = self.b - (self.b - self.a) * u
        if scalar:
            return float(x)
        return x

    def invmap(self, x: float | np.ndarray) -> Any:
        """Map ``x in [a, b]`` back to ``y in [-1, 1]`` (analytical inverse)."""
        xa, scalar = _as_array(x)
        L = self.params.L
        gap_u = self.gap_unit
        if self.side == "left":
            u = (xa - self.a) / (self.b - self.a)
            # Points within the (tiny) gap near the clustered endpoint are
            # mapped to t = -1; points at/beyond b map to t = +1. This keeps
            # evaluation inside [-1, 1] so the onefun is not extrapolated.
            u_safe = np.clip(u, gap_u, 1.0)
            s = self._s_of_u(u_safe)
            t = 2.0 * s / L + 1.0
        else:  # side == "right"
            u = (self.b - xa) / (self.b - self.a)
            u_safe = np.clip(u, gap_u, 1.0)
            s = self._s_of_u(u_safe)
            t = -(2.0 * s / L + 1.0)
        if scalar:
            return float(t)
        return t

    def drvmap(self, y: float | np.ndarray) -> Any:
        """Return ``dx/dy`` of :meth:`formap` evaluated at ``y``."""
        t, scalar = _as_array(y)
        L = self.params.L
        scale = self.b - self.a
        if self.side == "left":
            s = L * (t - 1.0) * 0.5
            dxdy = scale * self._du_ds(s) * (L * 0.5)
        else:
            s = L * (-t - 1.0) * 0.5
            # Reflection introduces a sign flip in ds/dy, plus the outer x = b - (b-a)*u
            # introduces another, so they cancel.
            dxdy = scale * self._du_ds(s) * (L * 0.5)
        if scalar:
            return float(dxdy)
        return dxdy

    def __repr__(self) -> str:  # pragma: no cover - trivial
        """Return a developer-friendly representation."""
        return f"SingleSlitMap(a={self.a!r}, b={self.b!r}, params={self.params!r}, side={self.side!r})"


class DoubleSlitMap:
    """Paper-faithful infinite two-slit-strip map ``psi_S``.

    Maps the reference interval ``[-1, 1]`` to (approximately) the
    logical interval ``[a, b]`` via the composition of an affine scaling
    and the inverse infinite two-slit-strip map of Adcock & Richardson
    (arXiv:1305.2643). The map's derivative vanishes
    super-algebraically at *both* endpoints ``t = ±1``, so functions
    with simultaneous endpoint singularities at ``a`` and ``b`` (e.g.
    ``sqrt((x - a)(b - x))``) become analytic in ``t``.

    Mathematical form::

        s(y) = L * y                                      # [-1, 1] -> [-L, L]
        u(s) = (alpha/pi) * [logaddexp(0, pi(s+1/2)/alpha)
                              - logaddexp(0, pi(s-1/2)/alpha)]
        x    = a + (b - a) * u(s(y))

    The construction satisfies ``u(0) = 1/2`` and ``u(±inf) = (1±1)/2``;
    with finite ``L`` the image of ``[-1, 1]`` is short of both endpoints
    by :attr:`gap` (negligible at the default ``L = 8``).

    Args:
        a: Left endpoint of the logical interval.
        b: Right endpoint of the logical interval (must satisfy ``b > a``).
        params: A :class:`MapParams` instance carrying ``(L, alpha)``.
            If ``None``, ``MapParams()`` defaults are used.

    Raises:
        ValueError: If ``b <= a``.
    """

    def __init__(
        self,
        a: float,
        b: float,
        params: MapParams | None = None,
    ) -> None:
        """Initialise a symmetric two-slit-strip clustering map.

        See the class docstring for parameter descriptions.
        """
        if not (b > a):
            msg = "require b > a"
            raise ValueError(msg)
        self.a = float(a)
        self.b = float(b)
        self.params = params if params is not None else MapParams()

    # Conveniences ------------------------------------------------------
    @property
    def alpha(self) -> float:
        """Strip half-width ``alpha`` (equal to ``self.params.alpha``)."""
        return self.params.alpha

    @property
    def L(self) -> float:
        """Truncation length ``L`` (equal to ``self.params.L``)."""
        return self.params.L

    @property
    def support(self) -> tuple[float, float]:
        """Return the *nominal* logical support ``(a, b)`` as a plain tuple."""
        return (self.a, self.b)

    @property
    def gap_unit(self) -> float:
        """Unit-interval shortfall at each clustered endpoint (equal at both ends by symmetry)."""
        a_p = self.params.alpha
        L = self.params.L
        # u(-L) = (alpha/pi)*[logaddexp(0, pi(-L+1/2)/alpha) - logaddexp(0, pi(-L-1/2)/alpha)]
        z_plus = np.pi * (-L + 0.5) / a_p
        z_minus = np.pi * (-L - 0.5) / a_p
        return float((a_p / np.pi) * (np.logaddexp(0.0, z_plus) - np.logaddexp(0.0, z_minus)))

    @property
    def gap(self) -> float:
        """Distance between :meth:`formap` ``(-1)`` and ``a`` (and symmetrically at ``b``)."""
        return (self.b - self.a) * self.gap_unit

    # Internal helpers --------------------------------------------------
    def _u_of_s(self, s: np.ndarray) -> np.ndarray:
        """Compute ``u(s)`` stably via ``logaddexp``."""
        a_p = self.params.alpha
        z_plus = np.pi * (s + 0.5) / a_p
        z_minus = np.pi * (s - 0.5) / a_p
        return (a_p / np.pi) * (np.logaddexp(0.0, z_plus) - np.logaddexp(0.0, z_minus))

    def _du_ds(self, s: np.ndarray) -> np.ndarray:
        """Derivative ``du/ds = sigmoid(z_plus) - sigmoid(z_minus)``."""
        a_p = self.params.alpha
        z_plus = np.pi * (s + 0.5) / a_p
        z_minus = np.pi * (s - 0.5) / a_p
        sig_plus = np.where(z_plus >= 0.0, 1.0 / (1.0 + np.exp(-z_plus)), np.exp(z_plus) / (1.0 + np.exp(z_plus)))
        sig_minus = np.where(z_minus >= 0.0, 1.0 / (1.0 + np.exp(-z_minus)), np.exp(z_minus) / (1.0 + np.exp(z_minus)))
        return sig_plus - sig_minus

    # IntervalMap protocol ---------------------------------------------
    def formap(self, y: float | np.ndarray) -> Any:
        """Map ``y in [-1, 1]`` to ``x`` clustered near both endpoints."""
        t, scalar = _as_array(y)
        s = self.params.L * t
        u = self._u_of_s(s)
        x = self.a + (self.b - self.a) * u
        if scalar:
            return float(x)
        return x

    def invmap(self, x: float | np.ndarray) -> Any:
        """Map ``x in [a, b]`` back to ``y in [-1, 1]`` by Newton iteration.

        ``u(s)`` has no closed-form inverse in elementary functions, so we
        solve ``u(s) = u_target`` by a few Newton steps starting from a
        well-conditioned initial guess based on the asymptotic behaviour
        ``u(s) ~ 1/2 + s/(2*alpha)`` near ``s = 0`` and the saturating
        single-slit limit elsewhere.
        """
        xa, scalar = _as_array(x)
        u_target = (xa - self.a) / (self.b - self.a)
        u_target = np.clip(u_target, np.finfo(float).tiny, 1.0 - np.finfo(float).tiny)
        # Initial guess: invert the dominant single-slit branch.
        # For u < 1/2 use the left slit; for u > 1/2 use the right slit's reflection.
        a_p = self.params.alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            # Left-branch guess: u ≈ (alpha/pi)*log(1 + exp(pi*(s+1/2)/alpha)) for s << 1/2.
            # Solve: s_guess_L = (alpha/pi)*log(exp(pi*u/alpha) - 1) - 1/2.
            w = np.pi * u_target / a_p
            s_guess_L = (a_p / np.pi) * (w + np.log1p(-np.exp(-np.clip(w, 1e-30, None)))) - 0.5
            # Right-branch guess: by symmetry u(s) = 1 - u(-s), so s_guess_R = -s_for_(1-u).
            w_r = np.pi * (1.0 - u_target) / a_p
            s_guess_R = -((a_p / np.pi) * (w_r + np.log1p(-np.exp(-np.clip(w_r, 1e-30, None)))) - 0.5)
        s = np.where(u_target < 0.5, s_guess_L, s_guess_R)
        # A handful of Newton iterations are enough for double precision across [-L, L].
        for _ in range(40):
            f_val = self._u_of_s(s) - u_target
            df = self._du_ds(s)
            # Guard against vanishing derivative at extreme s; these points won't move further.
            step = np.where(df > 0.0, f_val / np.where(df > 0.0, df, 1.0), 0.0)
            s = s - step
            if np.all(np.abs(step) < 1e-15 * (1.0 + np.abs(s))):
                break
        t = s / self.params.L
        if scalar:
            return float(t)
        return t

    def drvmap(self, y: float | np.ndarray) -> Any:
        """Return ``dx/dy`` of :meth:`formap` evaluated at ``y``."""
        t, scalar = _as_array(y)
        L = self.params.L
        s = L * t
        dxdy = (self.b - self.a) * self._du_ds(s) * L
        if scalar:
            return float(dxdy)
        return dxdy

    def __repr__(self) -> str:  # pragma: no cover - trivial
        """Return a developer-friendly representation."""
        return f"DoubleSlitMap(a={self.a!r}, b={self.b!r}, params={self.params!r})"
