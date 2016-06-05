# -*- coding: utf-8 -*-

"""User-interface functions"""

from chebpy.core.chebfun import Chebfun
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
        return _initfun(f, domain, n)

     # chebfun('x', ... )
    if isinstance(f, str) and len(f) is 1 and f.isalpha():
        return _initfun(lambda x: x, domain, n)

    try:
        # chebfun(3.14, ... ), chebfun('3.14', ... )
        return Chebfun.initconst(float(f), domain)
    except:
        raise ValueError(f)


def _initfun(f, domain, n):
    if n is None:
        return Chebfun.initfun_adaptive(f, domain)
    else:
        return Chebfun.initfun_fixedlen(f, domain, n)
