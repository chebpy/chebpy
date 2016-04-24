# -*- coding: utf-8 -*-
"""
Placeholder class
"""

from __future__ import division

from abc import ABCMeta

from numpy import linspace

from matplotlib.pyplot import gca

from pyfun.fun import Fun
from pyfun.chebtech import Chebtech2
from pyfun.utilities import Subdomain

from pyfun.settings import DefaultPrefs
from pyfun.decorators import emptycase

Techs = {
    "Chebtech2": Chebtech2,
}

Tech = Techs[DefaultPrefs.tech]

class InconsistentSubdomains(Exception):
    """Raised when two Classicfun subdomains to not match"""

class Classicfun(Fun):

    __metaclass__ = ABCMeta

    def __init__(self, onefun, subdomain):
        """Initialise a Classicfun from its two defining properties: a
        Subdomain object and a Onefun object"""
        self.onefun = onefun
        self.subdomain = subdomain

    @classmethod
    def initempty(cls):
        """Adaptive initialisation of a Classicfun from a callable
        function f and a Subdomain object. The subdomain's subdomain has no
        relevance to the emptiness status of a Classicfun so we
        arbitrarily set this to be DefaultPrefs.subdomain"""
        subdomain = Subdomain()
        onefun = Tech.initempty()
        return cls(onefun, subdomain)

    @classmethod
    def initconst(cls, c, subdomain):
        """ """
        onefun = Tech.initconst(c)
        return cls(onefun, subdomain)

    @classmethod
    def initfun_adaptive(cls, f, subdomain):
        """Adaptive initialisation of a BndFun from a callable function f
        and a Subdomain object"""
        uifunc = lambda y: f(subdomain(y))
        onefun = Tech.initfun(uifunc)
        return cls(onefun, subdomain)

    @classmethod
    def initfun_fixedlen(cls, f, subdomain, n):
        """Fixed length initialisation of a BndFun from a callable
        function f and a Subdomain object"""
        uifunc = lambda y: f(subdomain(y))
        onefun = Tech.initfun(uifunc, n)
        return cls(onefun, subdomain)

    def __call__(self, x, how="clenshaw"):
        y = self.subdomain.invmap(x)
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

    def subdomain(self):
        """Return the Subdomain object associated with the Classicfun"""
        return self.subdomain

    def endpoints(self):
        """Return a 2-array of endpointvalues taken from the subdomain"""
        return self.subdomain.values

    def endvalues(self):
        """Return a 2-array of endpointvalues taken from the subdomain"""
        return self(self.endpoints())

    def sum(self):
        a, b = self.endpoints()
        return .5*(b-a) * self.onefun.sum()

    def cumsum(self):
        a, b = self.endpoints()
        onefun = .5*(b-a) * self.onefun.cumsum()
        return self.__class__(onefun, self.subdomain)

    def diff(self):
        a, b = self.endpoints()
        onefun = 2./(b-a) * self.onefun.diff()
        return self.__class__(onefun, self.subdomain)

    def roots(self):
        uroots = self.onefun.roots()
        return self.subdomain(uroots)

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
    setattr(Classicfun, methodname, method)

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
        return self.__class__(onefun, self.subdomain)
    method.__name__ = methodname
    method.__doc__ = "TODO: CHANGE THIS TO SOMETHING MEANINGFUL"
    setattr(Classicfun, methodname, method)

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
    @emptycase()
    def method(self, f, *args, **kwargs):
        cls = self.__class__
        if isinstance(f, cls):
            # TODO: as in ChebTech, is a decorator apporach here better?
            if f.isempty():
                return f.copy()
            g = f.onefun
            # raise Exception if subdomains are not consistent
            if self.subdomain != f.subdomain:
                raise InconsistentSubdomains(self.subdomain(), f.subdomain())
        else:
            # let the lower level classes raise any other exceptions
            g = f
        onefun = getattr(self.onefun, methodname)(g, *args, **kwargs)
        return cls(onefun, self.subdomain)
    method.__name__ = methodname
    method.__doc__ = "TODO: CHANGE THIS TO SOMETHING MEANINGFUL"
    setattr(Classicfun, methodname, method)

for methodname in methods_onefun_binary:
    addBinaryOp(methodname)
