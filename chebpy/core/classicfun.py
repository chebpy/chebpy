from abc import ABC

import numpy as np

from .fun import Fun
from .chebtech import Chebtech2
from .utilities import Interval
from .settings import _preferences as prefs
from .decorators import self_empty
from .exceptions import IntervalMismatch, NotSubinterval
from .plotting import import_plt, plotfun


techdict = {
    "Chebtech2": Chebtech2,
}


class Classicfun(Fun, ABC):

    # --------------------------
    #  alternative constructors
    # --------------------------
    @classmethod
    def initempty(cls):
        """Adaptive initialisation of a Classicfun from a callable
        function f and a Interval object. The interval's interval has no
        relevance to the emptiness status of a Classicfun so we
        arbitrarily set this to be DefaultPreferences.interval"""
        interval = Interval()
        onefun = techdict[prefs.tech].initempty(interval=interval)
        return cls(onefun, interval)

    @classmethod
    def initconst(cls, c, interval):
        """Classicfun representation of a constant on the supplied interval"""
        onefun = techdict[prefs.tech].initconst(c, interval=interval)
        return cls(onefun, interval)

    @classmethod
    def initidentity(cls, interval):
        """Classicfun representation of f(x) = x on the supplied interval"""
        onefun = techdict[prefs.tech].initvalues(
            np.asarray(interval), interval=interval
        )
        return cls(onefun, interval)

    @classmethod
    def initfun_adaptive(cls, f, interval):
        """Adaptive initialisation of a BndFun from a callable function f
        and a Interval object"""
        onefun = techdict[prefs.tech].initfun(
            lambda y: f(interval(y)), interval=interval
        )
        return cls(onefun, interval)

    @classmethod
    def initfun_fixedlen(cls, f, interval, n):
        """Fixed length initialisation of a BndFun from a callable
        function f and a Interval object"""
        onefun = techdict[prefs.tech].initfun(
            lambda y: f(interval(y)), n, interval=interval
        )
        return cls(onefun, interval)

    # -------------------
    #  'private' methods
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
        out = "{0}([{2}, {3}], {1})".format(
            self.__class__.__name__, self.size, *self.support
        )
        return out

    # ------------
    #  properties
    # ------------
    @property
    def coeffs(self):
        """Return the coeffs property of the underlying onefun"""
        return self.onefun.coeffs

    @property
    def endvalues(self):
        """Return a 2-array of endpointvalues taken from the interval"""
        return self.__call__(self.support)

    @property
    def interval(self):
        """Return the Interval object associated with the Classicfun"""
        return self._interval

    @property
    def isconst(self):
        """Return the isconst property of the underlying onefun"""
        return self.onefun.isconst

    @property
    def iscomplex(self):
        """Determine whether the underlying onefun is complex or real valued"""
        return self.onefun.iscomplex

    @property
    def isempty(self):
        """Return the isempty property of the underlying onefun"""
        return self.onefun.isempty

    @property
    def size(self):
        """Return the size property of the underlying onefun"""
        return self.onefun.size

    @property
    def support(self):
        """Return a 2-array of endpoints taken from the interval"""
        return np.asarray(self.interval)

    @property
    def vscale(self):
        """Return the vscale property of the underlying onefun"""
        return self.onefun.vscale

    # -----------
    #  utilities
    # -----------

    def imag(self):
        if self.iscomplex:
            return self.__class__(self.onefun.imag(), self.interval)
        else:
            return self.initconst(0, interval=self.interval)

    def real(self):
        if self.iscomplex:
            return self.__class__(self.onefun.real(), self.interval)
        else:
            return self

    def restrict(self, subinterval):
        """Return a Classicfun that matches self on a subinterval of its
        interval of definition. The output is formed using a fixed length
        construction using same number of degrees of freedom as self."""
        if subinterval not in self.interval:
            raise NotSubinterval(self.interval, subinterval)
        if self.interval == subinterval:
            return self
        else:
            return self.__class__.initfun_fixedlen(self, subinterval, self.size)

    def translate(self, c):
        """Translate a fun by c, i.e., return f(x-c)"""
        return self.__class__(self.onefun, self.interval + c)

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
        a, b = self.support
        onefun = 0.5 * (b - a) * self.onefun.cumsum()
        return self.__class__(onefun, self.interval)

    def diff(self):
        a, b = self.support
        onefun = 2.0 / (b - a) * self.onefun.diff()
        return self.__class__(onefun, self.interval)

    def sum(self):
        a, b = self.support
        return 0.5 * (b - a) * self.onefun.sum()


# ----------
#  plotting
# ----------

plt = import_plt()
if plt:

    def plot(self, ax=None, **kwds):
        return plotfun(self, self.support, ax=ax, **kwds)

    setattr(Classicfun, "plot", plot)

# ----------------------------------------------------------------
#  methods that execute the corresponding onefun method as is
# ----------------------------------------------------------------

methods_onefun_other = ("values", "plotcoeffs")


def addUtility(methodname):
    def method(self, *args, **kwds):
        return getattr(self.onefun, methodname)(*args, **kwds)

    method.__name__ = methodname
    method.__doc__ = "TODO: CHANGE THIS TO SOMETHING MEANINGFUL"
    setattr(Classicfun, methodname, method)


for methodname in methods_onefun_other:
    if methodname[:4] == "plot" and plt is None:
        continue
    addUtility(methodname)


# -----------------------------------------------------------------------
#  unary operators and zero-argument utlity methods returning a onefun
# -----------------------------------------------------------------------

methods_onefun_zeroargs = ("__pos__", "__neg__", "copy", "simplify")


def addZeroArgOp(methodname):
    def method(self, *args, **kwds):
        onefun = getattr(self.onefun, methodname)(*args, **kwds)
        return self.__class__(onefun, self.interval)

    method.__name__ = methodname
    method.__doc__ = "TODO: CHANGE THIS TO SOMETHING MEANINGFUL"
    setattr(Classicfun, methodname, method)


for methodname in methods_onefun_zeroargs:
    addZeroArgOp(methodname)

# -----------------------------------------
# binary operators returning a onefun
# -----------------------------------------

# ToDo: change these to operator module methods
methods_onefun_binary = (
    "__add__",
    "__div__",
    "__mul__",
    "__pow__",
    "__radd__",
    "__rdiv__",
    "__rmul__",
    "__rpow__",
    "__rsub__",
    "__rtruediv__",
    "__sub__",
    "__truediv__",
)


def addBinaryOp(methodname):
    @self_empty()
    def method(self, f, *args, **kwds):
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
        onefun = getattr(self.onefun, methodname)(g, *args, **kwds)
        return cls(onefun, self.interval)

    method.__name__ = methodname
    method.__doc__ = "TODO: CHANGE THIS TO SOMETHING MEANINGFUL"
    setattr(Classicfun, methodname, method)


for methodname in methods_onefun_binary:
    addBinaryOp(methodname)

# ---------------------------
#  numpy universal functions
# ---------------------------


def addUfunc(op):
    @self_empty()
    def method(self):
        return self.__class__.initfun_adaptive(lambda x: op(self(x)), self.interval)

    name = op.__name__
    method.__name__ = name
    method.__doc__ = "TODO: CHANGE THIS TO SOMETHING MEANINGFUL"
    setattr(Classicfun, name, method)


ufuncs = (
    np.absolute,
    np.arccos,
    np.arccosh,
    np.arcsin,
    np.arcsinh,
    np.arctan,
    np.arctanh,
    np.cos,
    np.cosh,
    np.exp,
    np.exp2,
    np.expm1,
    np.log,
    np.log2,
    np.log10,
    np.log1p,
    np.sinh,
    np.sin,
    np.tan,
    np.tanh,
    np.sqrt,
)

for op in ufuncs:
    addUfunc(op)
