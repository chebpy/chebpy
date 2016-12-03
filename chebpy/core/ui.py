# -*- coding: utf-8 -*-

"""User-facing functions"""

from chebpy.core.bndfun import Bndfun
from chebpy.core.chebfun import Chebfun
from chebpy.core.utilities import Domain
from chebpy.core.settings import DefaultPrefs


def chebfun(f=None, domain=None, n=None):
    """Chebfun constructor
    """
    # chebfun()
    if f is None:
        return Chebfun.initempty()

    domain = DefaultPrefs.domain if domain is None else domain

    # chebfun(lambda x: f(x), ... )
    if hasattr(f, "__call__"):
        return Chebfun.initfun(f, domain, n)

    # chebfun('x', ... )
    if isinstance(f, str) and len(f) is 1 and f.isalpha():
        if n:
            return Chebfun.initfun(lambda x: x, domain, n)
        else:
            return Chebfun.initidentity(domain)

    try:
        # chebfun(3.14, ... ), chebfun('3.14', ... )
        return Chebfun.initconst(float(f), domain)
    except:
        raise ValueError(f)

def piecewise_constant(domain=[-1,0,1], values=[0,1]):
    """Initialise a piecewise constant Chebfun"""
    funs = []
    intervals = [x for x in Domain(domain).intervals]
    for interval, value in zip(intervals, values):
        funs.append(Bndfun.initconst(value, interval))
    return Chebfun(funs)

def heaviside(domain=[-1,0,1]):
    """Heaviside step-function"""
    return piecewise_constant(domain)
