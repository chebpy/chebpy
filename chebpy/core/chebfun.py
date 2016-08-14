# -*- coding: utf-8 -*-

from __future__ import division

from itertools import izip

from operator import __add__
from operator import __div__
from operator import __mul__
from operator import __sub__

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

from numpy import array
from numpy import append
from numpy import concatenate
from numpy import full
from numpy import isscalar
from numpy import linspace
from numpy import mean
from numpy import max
from numpy import nan
from numpy import ones
from numpy import sum

from matplotlib.pyplot import gca

from chebpy.core.bndfun import Bndfun
from chebpy.core.settings import DefaultPrefs
from chebpy.core.utilities import Interval
from chebpy.core.utilities import Domain
from chebpy.core.utilities import check_funs
from chebpy.core.utilities import generate_funs
from chebpy.core.utilities import compute_breakdata
from chebpy.core.decorators import self_empty
from chebpy.core.decorators import float_argument
from chebpy.core.decorators import cast_arg_to_chebfun
from chebpy.core.exceptions import BadDomainArgument
from chebpy.core.exceptions import BadFunLengthArgument
from chebpy.core.exceptions import DomainBreakpoints


class Chebfun(object):

    # --------------------------
    #  alternative constructors
    # --------------------------
    @classmethod
    def initempty(cls):
        return cls(array([]))

    @classmethod
    def initconst(cls, c, domain=DefaultPrefs.domain):
        funs = generate_funs(domain, Bndfun.initconst, [c])
        return cls(funs)

    @classmethod
    def initidentity(cls, domain=DefaultPrefs.domain):
        funs = generate_funs(domain, Bndfun.initidentity, [])
        return cls(funs)

    @classmethod
    def initfun_adaptive(cls, f, domain=DefaultPrefs.domain):
        funs = generate_funs(domain, Bndfun.initfun_adaptive, [f])
        return cls(funs)

    @classmethod
    def initfun_fixedlen(cls, f, n, domain=DefaultPrefs.domain):
        domain = array(domain)
        nn = array(n)
        if nn.size == 1:
            nn = nn * ones(domain.size-1)
        elif nn.size > 1:
            if nn.size != domain.size - 1:
                raise BadFunLengthArgument
        if domain.size < 2:
            raise BadDomainArgument
        funs = array([])
        intervals = zip(domain[:-1], domain[1:])
        for interval, length in zip(intervals, nn):
            interval = Interval(*interval)
            fun = Bndfun.initfun_fixedlen(f, interval, length)
            funs = append(funs, fun)
        return cls(funs)

    # --------------------
    #  operator overloads
    # --------------------
    def __add__(self, f):
        return self._apply_binop(f, __add__)

    @self_empty(array([]))
    @float_argument
    def __call__(self, x):

        # initialise output
        out = full(x.size, nan)

        # evaluate a fun when x is an interior point
        for fun in self:
            idx = fun.interval.isinterior(x)
            out[idx] = fun(x[idx])

        # evaluate the breakpoint data for x at a breakpoint
        breakpoints = self.breakpoints
        for breakpoint in breakpoints:
            out[x==breakpoint] = self.breakdata[breakpoint]

        # first and last funs used to evaluate outside of the chebfun domain
        lpts, rpts = x < breakpoints[0], x > breakpoints[-1]
        out[lpts] = self.funs[0](x[lpts])
        out[rpts] = self.funs[-1](x[rpts])
        return out

    def __init__(self, funs):
        self.funs = check_funs(funs)
        self.breakdata = compute_breakdata(self.funs)
        self.transposed = False

    def __iter__(self):
        return self.funs.__iter__()

    def __div__(self, f):
        return self._apply_binop(f, __div__)

    def __mul__(self, f):
        return self._apply_binop(f, __mul__)

    def __neg__(self):
        return self.__class__([-fun for fun in self])

    def __pos__(self):
        return self.__class__([+fun for fun in self])

    __radd__ = __add__

    def __rdiv__(self, c):
        # Executed when __div__(f, self) fails, which is to say whenever c
        # is not a Chebfun. We proceeed on the assumption f is a scalar.
        constfun = lambda x: .0*x + c
        newfuns = []
        for fun in self:
            quotnt = lambda x: constfun(x) / fun(x)
            newfun = fun.initfun_adaptive(quotnt, fun.interval)
            newfuns.append(newfun)
        return self.__class__(newfuns)

    @self_empty("chebfun<empty>")
    def __repr__(self):
        rowcol = "row" if self.transposed else "column"
        numpcs = self.funs.size
        plural = "" if numpcs == 1 else "s"
        header = "chebfun {} ({} smooth piece{})\n"\
            .format(rowcol, numpcs, plural)
        toprow = "       interval       length     endpoint values\n"
        tmplat = "[{:8.2g},{:8.2g}]   {:6}  {:8.2g} {:8.2g}\n"
        rowdta = ""
        for fun in self:
            endpts = fun.endpoints
            xl, xr = endpts
            fl, fr = fun(endpts)
            row = tmplat.format(xl, xr, fun.size, fl, fr)
            rowdta += row
        btmrow = "vertical scale = {:3.2g}".format(self.vscale)
        btmxtr = "" if numpcs == 1 else \
            "    total length = {}".format(sum([f.size for f in self]))
        return header + toprow + rowdta + btmrow + btmxtr

    __rmul__ = __mul__
    __rtruediv__ = __rdiv__

    def __rsub__(self, f):
        return -(self-f)

    def __str__(self):
        rowcol = "row" if self.transposed else "col"
        out = "<chebfun-{},{},{}>\n".format(
            rowcol, self.funs.size, sum([f.size for f in self]))
        return out

    def __sub__(self, f):
        return self._apply_binop(f, __sub__)

    __truediv__ = __div__

    # -------------------
    #  "private" methods
    # -------------------
    @self_empty()
    def _apply_binop(self, f, op):
        """Funnel method used in the implementation of Chebfun binary
        operators. The high-level idea is to first break each chebfun into a
        series of pieces corresponding to the union of the domains of each
        before applying the supplied binary operator and simplifying. In the
        case of the second argument being a scalar we don't need to do the
        simplify step, since at the Tech-level these operations are are defined
        such that there is no change in the number of coefficients.
        """
        try:
            if f.isempty:
                return f
        except:
            pass
        if isscalar(f):
            chbfn1 = self
            chbfn2 = f * ones(self.funs.size)
            simplify = False
        else:
            newdom = self.domain.union(f.domain)
            chbfn1 = self._break(newdom)
            chbfn2 = f._break(newdom)
            simplify = True
        newfuns = []
        for fun1, fun2 in izip(chbfn1, chbfn2):
            newfun = op(fun1, fun2)
            if simplify:
                newfun = newfun.simplify()
            newfuns.append(newfun)
        return self.__class__(newfuns)


    def _break(self, targetdomain):
        """Resamples self to the supplied Domain object, targetdomain. All of
        the breakpoints of self are required to be breakpoints of targetdomain.
        This can be achieved using Domain.union(f) prior to call."""

        if not self.domain.breakpoints_in(targetdomain).all():
            raise DomainBreakpoints

        newfuns = []
        subitvls = targetdomain.intervals
        interval = subitvls.next()

        # loop over the funs in self, incrementing subitvls
        # so long as interval remains contained within fun.interval
        for fun in self:
            while interval in fun.interval:
                newfun = fun.restrict(interval)
                newfuns.append(newfun)
                try:
                    interval = subitvls.next()
                except StopIteration:
                    break
        return self.__class__(newfuns)

    # ------------
    #  properties
    # ------------
    @property
    def breakpoints(self):
        return array(self.breakdata.keys())

    @property
    @self_empty(array([]))
    def domain(self):
        """Construct and return a Domain object corresponding to self"""
        return Domain.from_chebfun(self)

    @property
    @self_empty(array([]))
    def support(self):
        """Return an array containing the first and last breakpoints"""
        return self.breakpoints[[0,-1]]

    @property
    @self_empty(0)
    def hscale(self):
        return abs(self.support).max()

    @property
    @self_empty(False)
    def isconst(self):
        # TODO: find an abstract way of referencing funs[0].coeffs[0]
        c = self.funs[0].coeffs[0]
        return all(fun.isconst and fun.coeffs[0]==c for fun in self)

    @property
    def isempty(self):
        return self.funs.size == 0

    @property
    @self_empty(0)
    def vscale(self):
        return max([fun.vscale for fun in self])

    @property
    @self_empty()
    def x(self):
        """Return a Chebfun representing the identity the support of self"""
        return self.__class__.initidentity(self.support)

    # -----------
    #  utilities
    # -----------
    def copy(self):
        return self.__class__([fun.copy() for fun in self])

    @self_empty(array([]))
    def roots(self):
        allrts = []
        prvrts = array([])
        htol = 1e2 * self.hscale * DefaultPrefs.eps
        for fun in self:
            rts = fun.roots()
            # ignore first root if equal to the last root of previous fun
            # TODO: there could be multiple roots at breakpoints
            if prvrts.size > 0 and rts.size > 0:
                if abs(prvrts[-1]-rts[0]) <= htol:
                    rts = rts[1:]
            allrts.append(rts)
            prvrts = rts
        return concatenate([x for x in allrts])

    # ----------
    #  calculus
    # ----------
    def cumsum(self):
        newfuns = []
        prevfun = None
        for fun in self:
            integral = fun.cumsum()
            if prevfun:
                # enforce continuity by adding the function value
                # at the right endpoint of the previous fun
                _, fb = prevfun.endvalues
                integral = integral + fb
            newfuns.append(integral)
            prevfun = integral
        return self.__class__(newfuns)

    def diff(self):
        dfuns = array([fun.diff() for fun in self])
        return self.__class__(dfuns)

    def sum(self):
        return sum([fun.sum() for fun in self])

    # ----------
    #  plotting
    # ----------
    def plot(self, ax=None, *args, **kwargs):
        ax = ax if ax else gca()
        a, b = self.support
        xx = linspace(a, b, 2001)
        ax.plot(xx, self(xx), *args, **kwargs)
        return ax

    def plotcoeffs(self, ax=None, *args, **kwargs):
        ax = ax if ax else gca()
        for fun in self:
            fun.plotcoeffs(ax=ax)
        return ax

    # ----------
    #  utilities
    # ----------
    @self_empty()
    @cast_arg_to_chebfun
    def maximum(self, other):
        diffnc = self - other
        joined = self.domain.union(other.domain)
        newdom = joined.merge(diffnc.roots())
        funsA = self._break(newdom).funs
        funsB = other._break(newdom).funs
        x0 = mean(newdom.breakpoints[:2])
        if self(x0) > other(x0):
            funsA[1::2] = funsB[1::2]
        else:
            funsA[0::2] = funsB[0::2]
        funs = [x.simplify() for x in funsA]
        return self.__class__(funs)

# -----------------------
#  numpy unary functions
# -----------------------
def addUfunc(op):
    @self_empty()
    def method(self):
        return self.__class__([op(fun) for fun in self])
    name = op.__name__
    method.__name__ = name
    method.__doc__ = "TODO: CHANGE THIS TO SOMETHING MEANINGFUL"
    setattr(Chebfun, name, method)

ufuncs = (
    arccos, arccosh, arcsin, arcsinh, arctan, arctanh, cos, cosh, exp, exp2,
    expm1, log, log2, log10, log1p, sinh, sin, tan, tanh, sqrt,
)

for op in ufuncs:
    addUfunc(op)
