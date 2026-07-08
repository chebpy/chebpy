"""Construction of piecewise Chebfuns with endpoint singularities.

Kept separate from :mod:`chebpy.chebfun` so the singularity-piece plumbing
does not bloat the main class; it is used by the ``sing``-aware Chebfun
constructors via :func:`generate_singular_funs`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .bndfun import Bndfun
from .settings import _preferences as prefs
from .utilities import Domain


def _piece_singularity(sing: str, is_first: bool, is_last: bool) -> str | None:
    """Return the singularity side for one piece, or ``None`` for a smooth piece.

    Maps the whole-domain ``sing`` request onto the side (if any) that a
    single piece carries, given whether it is the first and/or last piece.

    Args:
        sing: One of ``"left"``, ``"right"``, ``"both"``.
        is_first: Whether the piece is the leftmost in the domain.
        is_last: Whether the piece is the rightmost in the domain.

    Returns:
        ``"left"``, ``"right"``, ``"both"``, or ``None`` (a smooth Bndfun piece).
    """
    left = is_first and sing in ("left", "both")
    right = is_last and sing in ("right", "both")
    if left and right:
        return "both"
    if left:
        return "left"
    if right:
        return "right"
    return None


def generate_singular_funs(
    f: Callable[..., Any],
    domain: Any,
    *,
    sing: str,
    params: Any,
) -> list[Any]:
    """Build per-piece funs for a Chebfun with endpoint singularities.

    The leftmost / rightmost pieces (depending on ``sing``) are built as
    :class:`~chebpy.singfun.Singfun` instances using the Adcock-Richardson
    clustering map; all interior pieces are ordinary :class:`Bndfun`.

    Args:
        f: Callable evaluating the function in logical coordinates.
        domain: Breakpoint sequence; outermost endpoints must be finite.
        sing: One of ``"left"``, ``"right"``, ``"both"``.
        params: A :class:`~chebpy.maps.MapParams` instance carrying
            ``(L, alpha)`` for the slit-strip clustering map. ``None`` means
            use :class:`~chebpy.maps.MapParams` defaults.

    Returns:
        list: Per-piece funs ready to feed to :class:`Chebfun`.

    Raises:
        ValueError: If ``sing`` is not one of the recognised values.
    """
    # Local import: chebfun and singfun are siblings under classicfun and
    # would otherwise risk a cyclic top-level import.
    from .maps import MapParams
    from .singfun import Singfun

    if sing not in ("left", "right", "both"):
        msg = f"sing must be 'left', 'right', or 'both'; got {sing!r}"
        raise ValueError(msg)
    if params is None:
        params = MapParams()
    dom = Domain(domain if domain is not None else prefs.domain)
    intervals = list(dom.intervals)
    n_pieces = len(intervals)
    funs: list[Any] = []
    for i, interval in enumerate(intervals):
        piece_sing = _piece_singularity(sing, i == 0, i == n_pieces - 1)
        if piece_sing is None:
            funs.append(Bndfun.initfun_adaptive(f, interval))
        else:
            funs.append(Singfun.initfun_adaptive(f, interval, sing=piece_sing, params=params))
    return funs
