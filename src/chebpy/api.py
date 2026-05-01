"""User-facing functions for creating and manipulating Chebfun objects.

This module provides the main interface for users to create Chebfun objects,
which are the core data structure in ChebPy for representing functions.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from .algorithms import barywts2, chebpts2
from .bndfun import Bndfun
from .chebfun import Chebfun
from .settings import _preferences as prefs
from .utilities import Domain, Interval


def chebfun(
    f: Callable[..., Any] | str | float | None = None,
    domain: np.ndarray | list[float] | None = None,
    n: int | None = None,
    *,
    sing: str | None = None,
    params: Any = None,
) -> "Chebfun":
    """Create a Chebfun object representing a function.

    A Chebfun object represents a function using Chebyshev polynomials. This constructor
    can create Chebfun objects from various inputs including callable functions,
    constants, and special strings.

    Args:
        f: The function to represent. Can be:
            - None: Creates an empty Chebfun
            - callable: A function handle like lambda x: x**2
            - str: A single alphabetic character (e.g., 'x') for the identity function
            - numeric: A constant value
        domain: The domain on which to define the function. Defaults to the domain
            specified in preferences.
        n: Optional number of points to use in the discretization. If None, adaptive
            construction is used.
        sing: Optional endpoint-singularity hint, one of ``"left"``, ``"right"``,
            or ``"both"``.  When set, the appropriate boundary pieces are built
            as :class:`~chebpy.singfun.Singfun` instances using the
            Adcock-Richardson exponential clustering map; interior pieces remain
            :class:`~chebpy.bndfun.Bndfun`.  Only supported with ``n=None``.
        params: Slit-strip map parameters (a :class:`~chebpy.maps.MapParams`
            carrying ``L`` and ``alpha``).  Default ``None`` uses
            :class:`~chebpy.maps.MapParams` defaults.

    Returns:
        Chebfun: A Chebfun object representing the function.

    Raises:
        ValueError: If unable to construct a constant function from the input.

    Examples:
        >>> # Empty Chebfun
        >>> f = chebfun()
        >>>
        >>> # Function from a lambda
        >>> import numpy as np
        >>> f = chebfun(lambda x: np.sin(x), domain=[-np.pi, np.pi])
        >>>
        >>> # Identity function
        >>> x = chebfun('x')
        >>>
        >>> # Constant function
        >>> c = chebfun(3.14)
        >>>
        >>> # Function with an endpoint singularity
        >>> g = chebfun(np.sqrt, domain=[0.0, 1.0], sing="left")
    """
    # Empty via chebfun()
    if f is None:
        return Chebfun.initempty()

    domain = domain if domain is not None else prefs.domain

    # Callable fct in chebfun(lambda x: f(x), ... )
    if callable(f):
        return Chebfun.initfun(f, domain, n, sing=sing, params=params)

    # Identity via chebfun('x', ... )
    if isinstance(f, str) and len(f) == 1 and f.isalpha():
        if n:
            return Chebfun.initfun(lambda x: x, domain, n)
        else:
            return Chebfun.initidentity(domain)

    try:
        # Constant fct via chebfun(3.14, ... ), chebfun('3.14', ... )
        return Chebfun.initconst(float(f), domain)
    except (OverflowError, ValueError) as err:
        raise ValueError(f) from err


def pwc(domain: list[float] | None = None, values: list[float] | None = None) -> "Chebfun":
    """Initialize a piecewise-constant Chebfun.

    Creates a piecewise-constant function represented as a Chebfun object.
    The function takes constant values on each interval defined by the domain.

    Args:
        domain (list): A list of breakpoints defining the intervals. Must have
            length equal to len(values) + 1. Default is [-1, 0, 1].
        values (list): A list of constant values for each interval. Default is [0, 1].

    Returns:
        Chebfun: A piecewise-constant Chebfun object.

    Examples:
        >>> # Create a step function with value 0 on [-1,0] and 1 on [0,1]
        >>> f = pwc()
        >>>
        >>> # Create a custom piecewise-constant function
        >>> f = pwc(domain=[-2, -1, 0, 1, 2], values=[-1, 0, 1, 2])
    """
    if values is None:
        values = [0, 1]
    if domain is None:
        domain = [-1, 0, 1]
    funs: list[Any] = []
    intervals = list(Domain(domain).intervals)
    for interval, value in zip(intervals, values, strict=False):
        funs.append(Bndfun.initconst(value, interval))
    return Chebfun(funs)


def chebpts(
    n: int,
    domain: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return *n* Chebyshev points and barycentric weights on *domain*.

    This provides the same functionality as MATLAB's ``chebpts(n, [a, b])``.
    The points are Chebyshev points of the second kind (i.e. the extrema of
    the Chebyshev polynomial T_{n-1} plus the endpoints).

    Args:
        n: Number of Chebyshev points.
        domain: Two-element list ``[a, b]`` specifying the interval.
            Defaults to ``[-1, 1]``.

    Returns:
        A ``(points, weights)`` tuple where *points* is an array of *n*
        Chebyshev points on the given domain and *weights* is the
        corresponding array of barycentric interpolation weights.

    Examples:
        >>> pts, wts = chebpts(4)
        >>> pts, wts = chebpts(4, [0, 3])
    """
    if domain is None:
        domain = [-1, 1]
    pts = chebpts2(n)
    wts = barywts2(n)
    interval = Interval(*domain)
    pts = interval(pts)
    return pts, wts


def trigfun(
    f: Callable[..., Any] | str | float | None = None,
    domain: np.ndarray | list[float] | None = None,
    n: int | None = None,
) -> "Chebfun":
    """Create a Chebfun backed by Fourier (trigonometric) technology.

    This is the explicit entry point for constructing periodic functions.
    Unlike ``chebfun``, which always uses Chebyshev polynomial technology,
    ``trigfun`` always uses :class:`~chebpy.trigtech.Trigtech` as the
    underlying approximation technology.  The user is responsible for
    ensuring that *f* is smooth and periodic on *domain*.

    The API mirrors :func:`chebfun` exactly:

    * ``trigfun()`` → empty Chebfun
    * ``trigfun(lambda x: np.sin(np.pi*x), [-1, 1])`` → from callable
    * ``trigfun('x')`` → identity (not truly periodic; provided for
      interface compatibility)
    * ``trigfun(3.14)`` → constant function

    Args:
        f: The function to represent.  Same semantics as :func:`chebfun`.
        domain: Domain ``[a, b]``.  Defaults to ``prefs.domain``.
        n: Fixed number of Fourier modes.  If None, adaptive construction
            is used.

    Returns:
        Chebfun: A Chebfun object whose pieces are backed by Trigtech.

    Examples:
        >>> import numpy as np
        >>> from chebpy import trigfun
        >>> f = trigfun(lambda x: np.cos(np.pi * x), [-1, 1])
        >>> float(f(0.0))
        1.0
        >>> g = trigfun(lambda x: np.sin(2 * np.pi * x))
        >>> bool(abs(g.sum()) < 1e-12)
        True
    """
    with prefs:
        prefs.tech = "Trigtech"
        return chebfun(f, domain, n)
