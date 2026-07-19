"""Pointwise and step-function operations on :class:`~chebpy.chebfun.Chebfun`.

Kept separate from :mod:`chebpy.chebfun` so the root-splitting machinery behind
``abs``, ``sign``, ``ceil``, ``floor`` and pointwise ``maximum``/``minimum``
does not bloat the main class.  Each public Chebfun method is a thin wrapper
around the corresponding function here.

Every function assumes non-empty operands; the wrapping Chebfun methods handle
the empty case (via ``@self_empty()``) before delegating.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from .bndfun import Bndfun
from .exceptions import SupportMismatch
from .settings import _preferences as prefs

if TYPE_CHECKING:
    from .chebfun import Chebfun


def absolute(f: Chebfun) -> Chebfun:
    """Return ``|f|`` by splitting the domain at the roots of ``f``."""
    newdom = f.domain.merge(f.roots())
    funs = [x.absolute() for x in f._break(newdom)]
    return f.__class__(funs)


def sign(f: Chebfun) -> Chebfun:
    """Return the piecewise sign of ``f``.

    Splits the domain at the roots of ``f`` and builds a constant piece with
    the appropriate sign on each sub-interval; breakpoints at roots are set to
    ``0``.
    """
    roots = f.roots()
    newdom = f.domain.merge(roots)
    funs = []
    for fun in f._break(newdom):
        mid = fun.support[0] + 0.5 * (fun.support[-1] - fun.support[0])
        s = float(np.sign(float(f(mid))))
        funs.append(Bndfun.initconst(s, fun.interval))
    result = f.__class__(funs)
    # Set breakdata: at roots sign is 0, elsewhere use sign of function.
    htol = max(1e2 * f.hscale * prefs.eps, prefs.eps)
    for bp in result.breakpoints:
        if roots.size > 0 and np.any(np.abs(bp - roots) <= htol):
            result.breakdata[bp] = 0.0
        else:
            result.breakdata[bp] = float(np.sign(float(f(bp))))
    return result


def ceil(f: Chebfun) -> Chebfun:
    """Return the piecewise ceiling of ``f``."""
    return _step_at_integer_crossings(f, np.ceil)


def floor(f: Chebfun) -> Chebfun:
    """Return the piecewise floor of ``f``."""
    return _step_at_integer_crossings(f, np.floor)


def _step_at_integer_crossings(f: Chebfun, rounder: Callable[[Any], Any]) -> Chebfun:
    """Build the piecewise-constant ``rounder(f)`` (used by :func:`ceil`/:func:`floor`).

    Splits the domain where ``f`` crosses integer values and assigns each piece
    the rounded value at its midpoint.
    """
    crossings = _integer_crossings(f)
    newdom = f.domain.merge(crossings)
    funs = []
    for fun in f._break(newdom):
        mid = fun.support[0] + 0.5 * (fun.support[-1] - fun.support[0])
        c = float(rounder(float(f(mid))))
        funs.append(Bndfun.initconst(c, fun.interval))
    result = f.__class__(funs)
    for bp in result.breakpoints:
        result.breakdata[bp] = float(rounder(float(f(bp))))
    return result


def _integer_crossings(f: Chebfun) -> np.ndarray:
    """Return the x-values where ``f`` crosses an integer value.

    Finds the roots of ``f - n`` for each integer ``n`` in the range of ``f``.
    """
    all_values = np.concatenate([fun.values() for fun in f])
    lo = int(np.floor(np.min(all_values)))
    hi = int(np.ceil(np.max(all_values)))
    crossings = []
    for n in range(lo, hi + 1):
        shifted = f - n
        crossings.extend(shifted.roots().tolist())
    return np.array(crossings)


def maximum_minimum(f: Chebfun, other: Chebfun, comparator: Callable[..., bool]) -> Any:
    """Return the pointwise maximum or minimum of ``f`` and ``other``.

    ``comparator`` selects which operand wins on each sub-interval:
    ``operator.ge`` gives the pointwise maximum and ``operator.lt`` the minimum.
    The switch points are the roots of ``f - other``.
    """
    if f.isempty or other.isempty:
        return f.__class__.initempty()

    # Find the intersection of domains.
    try:
        # Try to use union if supports match.
        newdom = f.domain.union(other.domain)
    except SupportMismatch:
        # If supports don't match, find the intersection.
        a_min, a_max = f.support
        b_min, b_max = other.support

        c_min = max(a_min, b_min)
        c_max = min(a_max, b_max)

        # If there's no intersection, return empty.
        if c_min >= c_max:
            return f.__class__.initempty()

        # Restrict both functions to the intersection and recurse.
        f_restricted = f.restrict([c_min, c_max])
        other_restricted = other.restrict([c_min, c_max])
        return maximum_minimum(f_restricted, other_restricted, comparator)

    roots = (f - other).roots()
    newdom = newdom.merge(roots)
    switch = newdom.support.merge(roots)

    if switch.size == 0:  # pragma: no cover
        return f.__class__.initempty()

    keys = 0.5 * ((-1) ** np.arange(switch.size - 1) + 1)
    if switch.size > 0 and comparator(other(switch[0]), f(switch[0])):
        keys = 1 - keys
    funs = np.array([])
    for interval, use_self in zip(switch.intervals, keys, strict=False):
        subdom = newdom.restrict(interval)
        subfun = f.restrict(subdom) if use_self else other.restrict(subdom)
        funs = np.append(funs, subfun.funs)
    return f.__class__(funs)
