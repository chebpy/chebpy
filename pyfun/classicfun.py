# -*- coding: utf-8 -*-
"""
Placeholder class
"""

from __future__ import division

from abc import ABCMeta

from numpy import linspace

from matplotlib.pyplot import gca

from pyfun.fun import Fun

from pyfun.chebtech import ChebTech2
from pyfun.settings import DefaultPrefs

from pyfun.utilities import checkempty
from pyfun.utilities import Domain

Techs = {
    "ChebTech2": ChebTech2,
}

Tech = Techs[DefaultPrefs.tech]

class InconsistentDomains(Exception):
    """Raised when two ClassicFun domains to not match"""

class ClassicFun(Fun):

    __metaclass__ = ABCMeta

    def __init__(self, onefun, domain):
        """Initialise a ClassicFun from its two defining properties: a
        Domain object and a Onefun object"""
        self.domain = domain
        self.onefun = onefun

    @classmethod
    def initempty(cls):
        """Adaptive initialisation of a ClassicFun from a callable
        function f and a Domain object. The domain's domain has no
        relevance to the emptiness status of a ClassicFun so we
        arbitrarily set this to be DefaultPrefs.domain"""
        domain = Domain()
        onefun = Tech.initempty()
        return cls(onefun, domain)

    @classmethod
    def initconst(cls, c, domain):
        """ """
        onefun = Tech.initconst(c)
        return cls(onefun, domain)

    @classmethod
    def initfun_adaptive(cls, f, domain):
        """Adaptive initialisation of a BndFun from a callable function f
        and a Domain object"""
        uifunc = lambda y: f(domain(y))
        onefun = Tech.initfun(uifunc)
        return cls(onefun, domain)

    @classmethod
    def initfun_fixedlen(cls, f, domain, n):
        """Fixed length initialisation of a BndFun from a callable
        function f and a Domain object"""
        uifunc = lambda y: f(domain(y))
        onefun = Tech.initfun(uifunc, n)
        return cls(onefun, domain)

    def __call__(self, x, how="clenshaw"):
        y = self.domain.invmap(x)
        return self.onefun(y, how)

    def plot(self, ax=None, *args, **kwargs):
        a, b = self.endpoints()
        ax = ax if ax else gca()
        xx = linspace(a, b, 2001)
        yy = self(xx)
        ax.plot(xx, yy, *args, **kwargs)
        return ax

    def __str__(self):
        out = "{0}([{2}, {3}], {1})".format(
            self.__class__.__name__, self.size(), *self.endpoints())
        return out

    def __repr__(self):
        return self.__str__()

    def domain(self):
        """Return the Domain object associated with the Classicfun"""
        return self.domain

    def endpoints(self):
        """Return a 2-array of endpointvalues taken from the domain"""
        return self.domain.values

    def sum(self):
        a, b = self.endpoints()
        return .5*(b-a) * self.onefun.sum()

    def cumsum(self):
        a, b = self.endpoints()
        onefun = .5*(b-a) * self.onefun.cumsum()
        return self.__class__(onefun, self.domain)

    def diff(self):
        a, b = self.endpoints()
        onefun = 2./(b-a) * self.onefun.diff()
        return self.__class__(onefun, self.domain)



# ----------------------------------------------------------------
#  methods that execute the corresponding onefun method as is
# ----------------------------------------------------------------

methods_onefun_other = (
    "size",
    "values",
    "coeffs",
    "isempty",
    "isconst",
    "vscale",
    "plotcoeffs",
)

def addUtility(methodname):
    def method(self, *args, **kwargs):
        return getattr(self.onefun, methodname)(*args, **kwargs)
    method.__name__ = methodname
    method.__doc__ = "TODO: CHANGE THIS TO SOMETHING MEANINGFUL"
    setattr(ClassicFun, methodname, method)

for methodname in methods_onefun_other:
    addUtility(methodname)


# -----------------------------------------------------------------------
#  unary operators and zero-argument utlity methods which return a onefun
# -----------------------------------------------------------------------

methods_onefun_zeroargs = (
    "__pos__",
    "__neg__",
    "copy",
    "simplify",
)

def addZeroArgOp(methodname):
    def method(self, *args, **kwargs):
        onefun = getattr(self.onefun, methodname)(*args, **kwargs)
        return self.__class__(onefun, self.domain)
    method.__name__ = methodname
    method.__doc__ = "TODO: CHANGE THIS TO SOMETHING MEANINGFUL"
    setattr(ClassicFun, methodname, method)

for methodname in methods_onefun_zeroargs:
    addZeroArgOp(methodname)


# -----------------------------------------
# binary operators which return a onefun
# -----------------------------------------

methods_onefun_binary= (
    "__add__",
    "__sub__",
    "__mul__",
    "__radd__",
    "__rsub__",
    "__rmul__",
)

def addBinaryOp(methodname):
    @checkempty()
    def method(self, f, *args, **kwargs):
        cls = self.__class__
        if isinstance(f, cls):
            # TODO: as in ChebTech, is a decorator apporach here better?
            if f.isempty():
                return f.copy()
            g = f.onefun
            # raise Exception if domains are not consistent
            if self.domain != f.domain:
                raise InconsistentDomains(self.domain(), f.domain())
        else:
            # let the lower level classes raise any other exceptions
            g = f
        onefun = getattr(self.onefun, methodname)(g, *args, **kwargs)
        return cls(onefun, self.domain)
    method.__name__ = methodname
    method.__doc__ = "TODO: CHANGE THIS TO SOMETHING MEANINGFUL"
    setattr(ClassicFun, methodname, method)

for methodname in methods_onefun_binary:
    addBinaryOp(methodname)
