"""User-facing functions for creating and manipulating Chebfun objects.

This module provides the main interface for users to create Chebfun objects,
which are the core data structure in ChebPy for representing functions.
"""

from .bndfun import Bndfun
from .chebfun import Chebfun
from .settings import _preferences as prefs
from .utilities import Domain


def chebfun(f=None, domain=None, n=None):
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

    Returns:
        Chebfun: A Chebfun object representing the function.

    Raises:
        ValueError: If unable to construct a constant function from the input.

    Examples:
        >>> # Empty Chebfun
        >>> f = chebfun()
        >>>
        >>> # Function from a lambda
        >>> f = chebfun(lambda x: np.sin(x), domain=[-np.pi, np.pi])
        >>>
        >>> # Identity function
        >>> x = chebfun('x')
        >>>
        >>> # Constant function
        >>> c = chebfun(3.14)
    """
    # Empty via chebfun()
    if f is None:
        return Chebfun.initempty()

    domain = domain if domain is not None else prefs.domain

    # Callable fct in chebfun(lambda x: f(x), ... )
    if hasattr(f, "__call__"):
        return Chebfun.initfun(f, domain, n)

    # Identity via chebfun('x', ... )
    if isinstance(f, str) and len(f) == 1 and f.isalpha():
        if n:
            return Chebfun.initfun(lambda x: x, domain, n)
        else:
            return Chebfun.initidentity(domain)

    try:
        # Constant fct via chebfun(3.14, ... ), chebfun('3.14', ... )
        return Chebfun.initconst(float(f), domain)
    except (OverflowError, ValueError):
        raise ValueError(f"Unable to construct const function from {{{f}}}")


def pwc(domain=[-1, 0, 1], values=[0, 1]):
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
    funs = []
    intervals = [x for x in Domain(domain).intervals]
    for interval, value in zip(intervals, values):
        funs.append(Bndfun.initconst(value, interval))
    return Chebfun(funs)
