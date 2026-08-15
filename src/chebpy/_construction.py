"""Construction of the piecewise fun list backing an ordinary Chebfun.

Sibling of :mod:`chebpy._singular_construction`, which does the analogous job
for domains with endpoint singularities. Both live above the ``Bndfun`` /
``CompactFun`` layer because they dispatch between those representations, which
is why neither belongs in :mod:`chebpy.utilities`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np

from .compactfun import CompactFun
from .exceptions import InvalidDomain
from .settings import _preferences as prefs
from .utilities import Domain, Interval


def generate_funs(
    domain: Domain | list[float] | None, bndfun_constructor: Callable[..., Any], kwds: dict[str, Any] | None = None
) -> list[Any]:
    """Generate a collection of function objects over a domain.

    This method is used by several of the Chebfun classmethod constructors to
    generate a collection of function objects over the specified domain. For
    pieces with finite endpoints the supplied ``bndfun_constructor`` is used;
    for pieces with one or both endpoints at ``±inf`` the corresponding
    classmethod on :class:`CompactFun` is invoked instead, dispatched by
    method name.

    Args:
        domain (array-like or None): Domain breakpoints. If None, uses default domain from preferences.
            The outermost breakpoints may be ``±inf``; interior breakpoints must be finite.
        bndfun_constructor (callable): Constructor function for creating function objects on
            finite intervals (typically a :class:`Bndfun` classmethod).
        kwds (dict, optional): Additional keyword arguments to pass to the constructor. Defaults to {}.

    Returns:
        list: List of function objects covering the domain.

    Raises:
        InvalidDomain: If a piece has an infinite endpoint but
            ``bndfun_constructor`` has no matching :class:`CompactFun`
            classmethod to build it with.
    """
    if kwds is None:
        kwds = {}
    domain = Domain(domain if domain is not None else prefs.domain)

    method_name = getattr(bndfun_constructor, "__name__", None)
    compact_constructor: Callable[..., Any] | None = (
        getattr(CompactFun, method_name) if method_name is not None and hasattr(CompactFun, method_name) else None
    )

    funs = []
    for a, b in itertools.pairwise(domain):
        a_f, b_f = float(a), float(b)
        if np.isfinite(a_f) and np.isfinite(b_f):
            interval: Any = Interval(a_f, b_f)
            ctor = bndfun_constructor
        else:
            if compact_constructor is None:
                raise InvalidDomain
            interval = (a_f, b_f)  # CompactFun classmethods accept (a, b) tuples with ±inf
            ctor = compact_constructor
        funs.append(ctor(**{**kwds, "interval": interval}))
    return funs
