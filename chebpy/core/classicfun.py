# -*- coding: utf-8 -*-
"""
Placeholder class
"""

from __future__ import division

from abc import ABCMeta

from numpy import arccos
from numpy import arccosh
from numpy import arcsin
from numpy import arcsinh
from numpy import arctan
from numpy import arctanh
from numpy import cos
from numpy import cosh
from numpy import exp
from numpy import exp2
from numpy import expm1
from numpy import sin
from numpy import sinh
from numpy import tan
from numpy import tanh
from numpy import log
from numpy import log2
from numpy import log10
from numpy import log1p
from numpy import sqrt
from numpy import linspace

from matplotlib.pyplot import gca

from chebpy.core.fun import Fun
from chebpy.core.chebtech import Chebtech2
from chebpy.core.utilities import Interval

from chebpy.core.settings import DefaultPrefs
from chebpy.core.decorators import self_empty
from chebpy.core.exceptions import IntervalMismatch
from chebpy.core.exceptions import NotSubinterval


Techs = {
    "Chebtech2": Chebtech2,
}

Tech = Techs[DefaultPrefs.tech]


class Classicfun(Fun):

    __metaclass__ = ABCMeta

    # --------------------------
    #  alternative constructors
    # --------------------------
    @classmethod
    def initempty(cls):
        """Adaptive initialisation of a Classicfun from a callable
        function f and a Interval object. The interval's interval has no
        relevance to the emptiness status of a Classicfun so we
        arbitrarily set this to be DefaultPrefs.interval"""
        interval = Interval()
        onefun = Tech.initempty()
        return cls(onefun, interval)

    @classmethod
    def initconst(cls, c, interval):
        """Classicfun representation of a constant on the supplied interval"""
        onefun = Tech.initconst(c)
        return cls(onefun, interval)

    @classmethod
    def initidentity(cls, interval):
        """Classicfun representation of f(x) = x on the supplied interval"""
        onefun = Tech.initvalues(interval.values)
        return cls(onefun, interval)

    @classmethod
    def initfun_adaptive(cls, f, interval):
        """Adaptive initialisation of a BndFun from a callable function f
        and a Interval object"""
        uifunc = lambda y: f(interval(y))
        onefun = Tech.initfun(uifunc)
        return cls(onefun, interval)

    @classmethod
    def initfun_fixedlen(cls, f, interval, n):
        """Fixed length initialisation of a BndFun from a callable
        function f and a Interval object"""
        uifunc = lambda y: f(interval(y))
        onefun = Tech.initfun(uifunc, n)
        return cls(onefun, interval)

    # -------------------
    #  "private" methods
    # -------------------
    def __call__(self, x, how="clenshaw"):
        y = self.interval.invmap(x)
        return self.onefun(y, how)

    def __init__(self, onefun, interval):
        """Initialise a Classicfun from its two defining properties: a
        Interval object and a Onefun object"""
        self.onefun = onefun
        self._interval = interval

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        out = "{0}([{2}, {3}], {1})".format(
            self.__class__.__name__, self.size, *self.endpoints)
        return out

    # ------------
    #  properties
    # ------------
    @property
    def coeffs(self):
        """Return the coeffs property of the underlying onefun"""
        return self.onefun.coeffs

    @property
    def endpoints(self):
        """Return a 2-array of endpoints taken from the interval"""
        return self.interval.values

    @property
    def endvalues(self):
        """Return a 2-array of endpointvalues taken from the interval"""
        return self.__call__(self.endpoints)

    @property
    def interval(self):
        """Return the Interval object associated with the Classicfun"""
        return self._interval

    @property
    def isconst(self):
        """Return the isconst property of the underlying onefun"""
        return self.onefun.isconst

    @property
    def isempty(self):
        """Return the isempty property of the underlying onefun"""
        return self.onefun.isempty

    @property
    def size(self):
        """Return the size property of the underlying onefun"""
        return self.onefun.size

    @property
    def vscale(self):
        """Return the vscale property of the underlying onefun"""
        return self.onefun.vscale

    # -----------
    #  utilities
    # -----------
    def restrict(self, subinterval):
        """Return a Classicfun that matches self on a subinterval of its
        interval of definition. The output is formed using a fixed length
        construction using same number of degrees of freedom as self."""
        if subinterval not in self.interval:
            raise NotSubinterval(self.interval, subinterval)
        if self.interval == subinterval:
            return self
        else:
            return type(self).initfun_fixedlen(self, subinterval, self.size)

    # -------------
    #  rootfinding
    # -------------
    def roots(self):
        uroots = self.onefun.roots()
        return self.interval(uroots)

    # ----------
    #  calculus
    # ----------
    def cumsum(self):
        a, b = self.endpoints
        onefun = .5*(b-a) * self.onefun.cumsum()
        return self.__class__(onefun, self.interval)

    def diff(self):
        a, b = self.endpoints
        onefun = 2./(b-a) * self.onefun.diff()
        return self.__class__(onefun, self.interval)

    def sum(self):
        a, b = self.endpoints
        return .5*(b-a) * self.onefun.sum()

    # ----------
    #  plotting
    # ----------
    def plot(self, ax=None, *args, **kwargs):
        a, b = self.endpoints
        ax = ax if ax else gca()
        xx = linspace(a, b, 2001)
        yy = self(xx)
        ax.plot(xx, yy, *args, **kwargs)
        return ax


# ----------------------------------------------------------------
#  methods that execute the corresponding onefun method as is
# ----------------------------------------------------------------

methods_onefun_other = (
    "values",
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
        return self.__class__(onefun, self.interval)
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
    "__div__",
    "__mul__",
    "__radd__",
    "__rdiv__",
    "__rmul__",
    "__rsub__",
    "__rtruediv__",
    "__sub__",
    "__truediv__",
)

def addBinaryOp(methodname):
    @self_empty()
    def method(self, f, *args, **kwargs):
        cls = self.__class__
        if isinstance(f, cls):
            # TODO: as in ChebTech, is a decorator apporach here better?
            if f.isempty:
                return f.copy()
            g = f.onefun
            # raise Exception if intervals are not consistent
            if self.interval != f.interval:
                raise IntervalMismatch(self.interval, f.interval)
        else:
            # let the lower level classes raise any other exceptions
            g = f
        onefun = getattr(self.onefun, methodname)(g, *args, **kwargs)
        return cls(onefun, self.interval)
    method.__name__ = methodname
    method.__doc__ = "TODO: CHANGE THIS TO SOMETHING MEANINGFUL"
    setattr(Classicfun, methodname, method)

for methodname in methods_onefun_binary:
    addBinaryOp(methodname)


# -----------------------
#  numpy unary functions
# -----------------------

def addUfunc(op):
    @self_empty()
    def method(self):
        cls = self.__class__
        fun = lambda x: op(self(x))
        return cls.initfun_adaptive(fun, self.interval)
    name = op.__name__
    method.__name__ = name
    method.__doc__ = "TODO: CHANGE THIS TO SOMETHING MEANINGFUL"
    setattr(Classicfun, name, method)

ufuncs = (
    arccos, arccosh, arcsin, arcsinh, arctan, arctanh, cos, cosh, exp, exp2,
    expm1, log, log2, log10, log1p, sinh, sin, tan, tanh, sqrt,
)

for op in ufuncs:
    addUfunc(op)
