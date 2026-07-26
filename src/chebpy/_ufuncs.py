"""Registration of elementwise NumPy ufunc methods on :class:`~chebpy.chebfun.Chebfun`.

Kept separate from :mod:`chebpy.chebfun` so the ufunc boilerplate does not
bloat the main class.  :func:`register_ufuncs` is called once, after the class
is defined, to attach ``sin``, ``cos``, ``exp``, ... methods that apply the
corresponding NumPy ufunc piecewise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .decorators import self_empty

if TYPE_CHECKING:
    from .chebfun import Chebfun

_UFUNCS = (
    np.arccos,
    np.arccosh,
    np.arcsin,
    np.arcsinh,
    np.arctan,
    np.arctanh,
    np.cos,
    np.cosh,
    np.exp,
    np.exp2,
    np.expm1,
    np.log,
    np.log2,
    np.log10,
    np.log1p,
    np.sinh,
    np.sin,
    np.tan,
    np.tanh,
    np.sqrt,
)


def _add_ufunc(cls: type, op: Any) -> None:
    """Attach a single NumPy ufunc *op* as an elementwise method on *cls*."""

    @self_empty()
    def method(self: Chebfun) -> Chebfun:
        """Apply a NumPy universal function to each piece of this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing ``op(f(x))``.
        """
        return self.__class__([op(fun) for fun in self])

    method.__name__ = op.__name__
    setattr(cls, op.__name__, method)


def register_ufuncs(cls: type) -> None:
    """Attach an elementwise method for each supported NumPy ufunc to *cls*.

    Args:
        cls: The :class:`~chebpy.chebfun.Chebfun` class to augment in place.
    """
    for op in _UFUNCS:
        _add_ufunc(cls, op)
