"""User-facing functions"""

from .core.bndfun import Bndfun
from .core.chebfun import Chebfun
from .core.utilities import Domain
from .core.settings import _preferences as prefs


def chebfun(f=None, domain=None, n=None):
    """Chebfun constructor"""
    # chebfun()
    if f is None:
        return Chebfun.initempty()

    domain = domain if domain is not None else prefs.domain

    # chebfun(lambda x: f(x), ... )
    if hasattr(f, "__call__"):
        return Chebfun.initfun(f, domain, n)

    # chebfun('x', ... )
    if isinstance(f, str) and len(f) == 1 and f.isalpha():
        if n:
            return Chebfun.initfun(lambda x: x, domain, n)
        else:
            return Chebfun.initidentity(domain)

    try:
        # chebfun(3.14, ... ), chebfun('3.14', ... )
        return Chebfun.initconst(float(f), domain)
    except (OverflowError, ValueError):
        raise ValueError(f"Unable to construct const function from {{{f}}}")


def pwc(domain=[-1, 0, 1], values=[0, 1]):
    """Initialise a piecewise-constant Chebfun"""
    funs = []
    intervals = [x for x in Domain(domain).intervals]
    for interval, value in zip(intervals, values):
        funs.append(Bndfun.initconst(value, interval))
    return Chebfun(funs)
