"""User-facing functions for creating and manipulating Chebfun objects.

This module provides the main interface for users to create Chebfun objects,
which are the core data structure in ChebPy for representing functions.
"""

from .bndfun import Bndfun
from .chebfun import Chebfun
from .chebop import Chebop
from .settings import _preferences as prefs
from .utilities import Domain


def chebfun(f=None, domain=None, n=None, splitting=None):
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
        splitting: Enable automatic domain splitting for functions with discontinuities
            or singularities. If None, uses the value from preferences. If True, enables
            splitting. If False, disables splitting.

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
        >>> # With automatic splitting for discontinuous functions
        >>> f = chebfun(lambda x: np.abs(x), domain=[-1, 1], splitting=True)
    """
    # Empty via chebfun()
    if f is None:
        return Chebfun.initempty()

    domain = domain if domain is not None else prefs.domain

    # Callable fct in chebfun(lambda x: f(x), ... )
    if hasattr(f, "__call__"):
        return Chebfun.initfun(f, domain, n, splitting=splitting)

    # Identity via chebfun('x', ... )
    if isinstance(f, str) and len(f) == 1 and f.isalpha():
        if n:
            return Chebfun.initfun(lambda x: x, domain, n, splitting=splitting)
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


def chebop(*args, **kwargs):
    """Create a Chebop (differential operator) object.

    A Chebop represents a linear or nonlinear differential operator with
    boundary conditions, used for solving boundary value problems (BVPs)
    and eigenvalue problems.

    Args:
        *args: Positional arguments for domain specification. Can be:
            - Single argument: domain as [a, b] or Domain object
            - Two arguments: either (a, b) for domain endpoints,
              or (op, domain) / (domain, op) for operator and domain
        **kwargs: Keyword arguments including:
            domain: Domain specification (can also be passed as keyword arg)
            op: Operator function taking (x, u, u', u'', ...)
            lbc: Left boundary condition function
            rbc: Right boundary condition function
            bc: List of boundary conditions
            rhs: Right-hand side function f(x) for Chebop @ u = rhs
            init: Initial guess for nonlinear problems
            point_constraints: Point constraints for specific locations

    Returns:
        Chebop: A differential operator object.

    Examples:
        >>> # Create operator on [-1, 1] with no BCs
        >>> N = chebop([-1, 1])
        >>> isinstance(N, Chebop)
        True
    """
    return Chebop(*args, **kwargs)
