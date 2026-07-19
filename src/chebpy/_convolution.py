"""Convolution of :class:`~chebpy.chebfun.Chebfun` objects.

This module implements ``h = f ★ g`` for piecewise Chebfuns.  It is kept
separate from :mod:`chebpy.chebfun` so the (sizeable) convolution algorithm
does not bloat the main class; :meth:`chebpy.chebfun.Chebfun.conv` is a thin
wrapper around :func:`convolve`.

Two strategies are used:

* When both inputs are single-piece :class:`~chebpy.bndfun.Bndfun` funs of
  equal width, the fast Hale-Townsend Legendre convolution is used
  (:func:`_equal_width_pair`).
* Otherwise each output sub-interval is built adaptively using
  Gauss-Legendre quadrature (:func:`_piecewise`).

The algorithm is based on:
    N. Hale and A. Townsend, "An algorithm for the convolution of Legendre
    series", SIAM J. Sci. Comput., 36(3), A1207-A1220, 2014.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from .algorithms import _conv_legendre, cheb2leg, leg2cheb
from .bndfun import Bndfun
from .chebtech import Chebtech
from .fun import Fun
from .trigtech import Trigtech
from .utilities import Interval

if TYPE_CHECKING:
    from .chebfun import Chebfun


def convolve(f: Chebfun, g: Chebfun) -> Chebfun:
    """Return the convolution ``h = f ★ g`` as a piecewise Chebfun.

    See :meth:`chebpy.chebfun.Chebfun.conv` for the full description; this is
    the implementation behind that method.
    """
    if f.isempty or g.isempty:
        return f.__class__.initempty()

    _reject_unsupported(f, g)
    _reject_nonzero_tails(f, g)

    # Fast path: both single-piece with equal-width finite domains.
    if _use_equal_width_fast_path(f, g):
        return _equal_width_pair(f, f.funs[0], g.funs[0])

    # General piecewise convolution.
    return _piecewise(f, g)


def _reject_unsupported(f: Chebfun, g: Chebfun) -> None:
    """Reject convolution operands the algorithms cannot handle.

    Both the Hale-Townsend Legendre algorithm and the Gauss-Legendre fallback
    assume Chebyshev coefficients on an affine map, so :class:`Trigtech`-backed
    pieces (Fourier coefficients; periodic convolution is a distinct
    ``circconv`` operation) and :class:`~chebpy.singfun.Singfun` pieces
    (non-affine clustering map) are refused with a clear error.

    Raises:
        NotImplementedError: If either operand contains a Trigtech- or
            Singfun-backed piece.
    """
    from .singfun import Singfun

    pieces = (*f.funs, *g.funs)
    if any(isinstance(piece.onefun, Trigtech) for piece in pieces):
        raise NotImplementedError(
            "conv() is not supported for trigfun (Trigtech-backed) inputs. "
            "Aperiodic convolution and periodic (circular) convolution are "
            "distinct operations; a dedicated circconv() for trigfuns is "
            "not yet implemented."
        )
    if any(isinstance(piece, Singfun) for piece in pieces):
        raise NotImplementedError(
            "conv() is not supported for Chebfuns containing Singfun pieces "
            "(functions with endpoint singularities represented by a non-affine "
            "clustering map)."
        )


def _reject_nonzero_tails(f: Chebfun, g: Chebfun) -> None:
    """Reject convolution when a :class:`CompactFun` piece has a non-zero tail.

    Convolution of a function with a non-zero asymptotic limit diverges on an
    unbounded interval, so refuse early with a clear error pointing the user at
    the algebraic-closure escape hatch.

    Raises:
        DivergentIntegralError: If any CompactFun piece of either operand has a
            non-zero ``tail_left`` or ``tail_right``.
    """
    from .compactfun import CompactFun
    from .exceptions import DivergentIntegralError

    for label, h in (("self", f), ("other", g)):
        for piece in h.funs:
            if isinstance(piece, CompactFun) and (piece.tail_left != 0.0 or piece.tail_right != 0.0):
                raise DivergentIntegralError(  # noqa: TRY003
                    f"Convolution requires both operands to decay to zero at "
                    f"±inf; got tail_left={piece.tail_left}, "
                    f"tail_right={piece.tail_right} for {label}. Consider "
                    f"subtracting a matched sigmoid first so the residual "
                    f"has zero tails, then convolving the residual."
                )


def _use_equal_width_fast_path(f: Chebfun, g: Chebfun) -> bool:
    """Return True if the fast equal-width Legendre path applies.

    The fast path requires both operands to be single finite :class:`Bndfun`
    pieces of equal width.  :class:`CompactFun` pieces (with possibly infinite
    logical support) always take the general piecewise path so the output is
    wrapped correctly.
    """
    from .compactfun import CompactFun

    if any(isinstance(piece, CompactFun) for piece in (*f.funs, *g.funs)):
        return False
    if f.funs.size != 1 or g.funs.size != 1:
        return False
    f_fun, g_fun = f.funs[0], g.funs[0]
    f_w = float(f_fun.support[1]) - float(f_fun.support[0])
    g_w = float(g_fun.support[1]) - float(g_fun.support[0])
    return bool(np.isclose(f_w, g_w))


def _equal_width_pair(f: Chebfun, f_fun: Any, g_fun: Any) -> Chebfun:
    """Convolve two single Bndfuns of equal width using the fast algorithm.

    Uses the Hale-Townsend Legendre convolution.  The two funs may be on
    different intervals as long as they have the same width.
    """
    a = float(f_fun.support[0])
    b = float(f_fun.support[1])
    c = float(g_fun.support[0])
    d = float(g_fun.support[1])

    h = (b - a) / 2.0  # half-width (same for both funs)

    leg_f = cheb2leg(f_fun.coeffs)
    leg_g = cheb2leg(g_fun.coeffs)

    gamma_left, gamma_right = _conv_legendre(leg_f, leg_g)

    gamma_left = h * gamma_left
    gamma_right = h * gamma_right

    cheb_left = leg2cheb(gamma_left)
    cheb_right = leg2cheb(gamma_right)

    mid = (a + b + c + d) / 2.0
    left_interval = Interval(a + c, mid)
    right_interval = Interval(mid, b + d)

    left_fun = Bndfun(Chebtech(cheb_left), left_interval)
    right_fun = Bndfun(Chebtech(cheb_right), right_interval)

    return f.__class__([left_fun, right_fun])


def _piecewise(f: Chebfun, g: Chebfun) -> Chebfun:
    """General piecewise convolution via Gauss-Legendre quadrature.

    The breakpoints of the result are the sorted, unique pairwise sums of the
    breakpoints of ``f`` and ``g``.  On each sub-interval the convolution
    integral is smooth, so we construct it adaptively.  When either input
    contains :class:`CompactFun` pieces, the corresponding ``±inf`` breakpoints
    are replaced with the numerical-support bounds for the purposes of
    integration; the outermost output pieces are then wrapped as
    :class:`CompactFun` so the result preserves the unbounded logical support.
    """
    f_logical_breaks = np.array(f.breakpoints, dtype=float)
    g_logical_breaks = np.array(g.breakpoints, dtype=float)
    left_inf = (not np.isfinite(f_logical_breaks[0])) or (not np.isfinite(g_logical_breaks[0]))
    right_inf = (not np.isfinite(f_logical_breaks[-1])) or (not np.isfinite(g_logical_breaks[-1]))

    f_breaks = _effective_breakpoints(f)
    g_breaks = _effective_breakpoints(g)

    # Output breakpoints: all pairwise sums, uniquified and coalesced.
    out_breaks = np.unique(np.add.outer(f_breaks, g_breaks).ravel())
    hscl = max(abs(out_breaks[0]), abs(out_breaks[-1]), 1.0)
    tol = 10.0 * np.finfo(float).eps * hscl
    mask = np.concatenate(([True], np.diff(out_breaks) > tol))
    out_breaks = out_breaks[mask]

    conv_eval = _make_evaluator(f, g, f_breaks, g_breaks)
    return _build_pieces(f, out_breaks, conv_eval, left_inf=left_inf, right_inf=right_inf)


def _effective_breakpoints(h: Chebfun) -> np.ndarray:
    """Return ``h``'s breakpoints with ±inf replaced by numerical-support bounds."""
    from .compactfun import CompactFun

    bps = np.array(h.breakpoints, dtype=float)
    if not np.isfinite(bps[0]) and isinstance(h.funs[0], CompactFun):
        bps[0] = float(h.funs[0].numerical_support[0])
    if not np.isfinite(bps[-1]) and isinstance(h.funs[-1], CompactFun):
        bps[-1] = float(h.funs[-1].numerical_support[1])
    return bps


def _make_evaluator(
    f: Chebfun, g: Chebfun, f_breaks: np.ndarray, g_breaks: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the Gauss-Legendre quadrature evaluator for ``(f ★ g)``.

    The integrand is broken at the breakpoints of ``f`` and the shifted
    breakpoints of ``g`` so it is polynomial on each sub-interval; the
    quadrature order is chosen to integrate that polynomial exactly.
    """
    f_a, f_b = float(f_breaks[0]), float(f_breaks[-1])
    g_c, g_d = float(g_breaks[0]), float(g_breaks[-1])

    max_deg = max(fun.size for fun in f.funs) + max(fun.size for fun in g.funs)
    n_quad = max(int(np.ceil((max_deg + 1) / 2)), 16)
    quad_nodes, quad_weights = np.polynomial.legendre.leggauss(n_quad)

    f_bps = [float(bp) for bp in f_breaks]
    g_bps = [float(bp) for bp in g_breaks]

    def conv_eval(x: np.ndarray) -> np.ndarray:
        """Evaluate (f ★ g)(x) via Gauss-Legendre quadrature."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        result = np.zeros(x.shape)
        for idx in range(x.size):
            xi = x[idx]
            t_lo = max(f_a, xi - g_d)
            t_hi = min(f_b, xi - g_c)
            if t_hi <= t_lo:
                continue
            inner = _subinterval_nodes(xi, t_lo, t_hi, f_bps, g_bps)
            total = 0.0
            for j in range(len(inner) - 1):
                a_int, b_int = inner[j], inner[j + 1]
                hw = (b_int - a_int) / 2.0
                mid = (a_int + b_int) / 2.0
                nodes = hw * quad_nodes + mid
                wts = hw * quad_weights
                total += np.dot(wts, f(nodes) * g(xi - nodes))
            result[idx] = total
        return result

    return conv_eval


def _subinterval_nodes(xi: float, t_lo: float, t_hi: float, f_bps: list[float], g_bps: list[float]) -> list[float]:
    """Return the sorted integration sub-interval boundaries in ``(t_lo, t_hi)``.

    Breaks at the breakpoints of ``f`` and the shifted breakpoints of ``g``
    that fall strictly inside ``(t_lo, t_hi)``, keeping the integrand polynomial
    on each resulting sub-interval.
    """
    inner = [t_lo, t_hi]
    inner.extend(bp for bp in f_bps if t_lo < bp < t_hi)
    inner.extend(xi - bp for bp in g_bps if t_lo < xi - bp < t_hi)
    return sorted(set(inner))


def _build_pieces(
    f: Chebfun,
    out_breaks: np.ndarray,
    conv_eval: Callable[[np.ndarray], np.ndarray],
    *,
    left_inf: bool,
    right_inf: bool,
) -> Chebfun:
    """Build one fun per output sub-interval, wrapping the unbounded ends.

    Interior pieces are finite :class:`Bndfun`; the outermost pieces are wrapped
    as :class:`CompactFun` when the corresponding logical edge is ``±inf`` so
    the result preserves the unbounded logical support.
    """
    from .compactfun import CompactFun

    n_pieces = len(out_breaks) - 1
    funs_list: list[Fun] = []
    for i in range(n_pieces):
        a_storage = float(out_breaks[i])
        b_storage = float(out_breaks[i + 1])
        interval = Interval(a_storage, b_storage)
        bnd = Bndfun.initfun_adaptive(conv_eval, interval)
        wrap_left = i == 0 and left_inf
        wrap_right = i == n_pieces - 1 and right_inf
        if wrap_left or wrap_right:
            a_logical = -np.inf if wrap_left else a_storage
            b_logical = np.inf if wrap_right else b_storage
            funs_list.append(CompactFun(bnd.onefun, interval, logical_interval=(a_logical, b_logical)))
        else:
            funs_list.append(bnd)

    return f.__class__(funs_list)
