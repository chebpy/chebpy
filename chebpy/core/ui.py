# -*- coding: utf-8 -*-

"""User-interface functions"""

from chebpy.core.chebfun import Chebfun
from chebpy.core.settings import DefaultPrefs

# Matlab-Chebfun style constructor
def chebfun(f, domain=DefaultPrefs.domain, n=None):
    if hasattr(f, "__call__"):
        f = f
    elif isinstance(f, str):
        if len(f) is 1 and f.isalpha():
            f = lambda x: x
        else:
            raise ValueError(f)
    else:
        raise ValueError(f)
    if n is None:
        return Chebfun.initfun_adaptive(f, domain)
    else:
        return Chebfun.initfun_fixedlen(f, domain, n)
