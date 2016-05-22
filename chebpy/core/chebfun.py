# -*- coding: utf-8 -*-

from itertools import izip

from operator import __add__

from numpy import array
from numpy import append
from numpy import concatenate
from numpy import full
from numpy import isscalar
from numpy import linspace
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
from chebpy.core.utilities import compute_breakdata
from chebpy.core.decorators import self_empty
from chebpy.core.decorators import float_argument
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
    def initfun_adaptive(cls, f, domain):
        domain = array(domain)
        if domain.size < 2:
            raise BadDomainArgument
        funs = array([])
        for interval in zip(domain[:-1], domain[1:]):
            interval = Interval(*interval)
            fun = Bndfun.initfun_adaptive(f, interval)
            funs = append(funs, fun)
        return cls(funs)

    @classmethod
    def initfun_fixedlen(cls, f, domain, n):
        domain = array(domain)
        if domain.size < 2:
            raise BadDomainArgument
        nn = array(n)
        if nn.size == 1:
            nn = nn * ones(domain.size-1)
        elif nn.size > 1:
            if nn.size != domain.size - 1:
                raise BadFunLengthArgument
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
    def __add__(self, other):
        return self.__apply_binop(other, __add__)

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

    def __mul__(self, other):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __pos__(self):
        raise NotImplementedError

    def __radd__(self):
        raise NotImplementedError

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

    def __rmul__(self):
        raise NotImplementedError

    def __rsub__(self):
        raise NotImplementedError

    def __str__(self):
        rowcol = "row" if self.transposed else "col"
        out = "<chebfun-{},{},{}>\n".format(
            rowcol, self.funs.size, sum([f.size() for f in self]))
        return out

    def __sub__(self):
        raise NotImplementedError

    # -------------------
    #  "private" methods
    # -------------------
    @self_empty()
    def __apply_binop(self, other, binop):
        """Funnel method used in the implementation of Chebfun binary
        operators. The high-level idea is to first break each chebfun into a
        series of pieces corresponding to the union of the domains of each
        before applying the supplied binary operator and simplifying. In the
        case of the second argument being a scalar we don't need to do the
        simplify step, since at the Tech-level these operations are are defined
        such that there is no change in the number of coefficients.
        """
        if other.isempty:
            return other
        if isscalar(other):
            chbfn1 = self
            chbfn2 = other*ones(self.funs.size)
            simplify = False
        else:
            newdom = self.domain.union(other.domain)
            chbfn1 = self.__break(newdom)
            chbfn2 = other.__break(newdom)
            simplify = True
        newfuns = []
        for fun1, fun2 in izip(chbfn1, chbfn2):
            newfun = binop(fun1, fun2)
            if simplify:
                newfun = newfun.simplify()
            newfuns.append(newfun)
        return self.__class__(newfuns)


    def __break(self, targetdomain):
        """Resamples self to the supplied Domain object, targetdomain. All of
        the breakpoints of self are required to be breakpoints of targetdomain.
        This is best achieved by using Domain.union(other) method prior to
        calling."""

        if not self.domain.breakpoints_in(targetdomain).all():
            raise DomainBreakpoints

        newfuns = []
        intvl_gen = targetdomain.__iter__()
        intvl = Interval(*intvl_gen.next())

        # loop over the funs in self, incrementing the intvl_gen generator so
        # long as intvl remains contained within fun.interval
        for fun in self:
            while intvl in fun.interval:
                newfuns.append(fun.restrict(intvl))
                try:
                    intvl = Interval(*intvl_gen.next())
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
        """The support of a Chebfun is an array containing its first and last
        breakpoint"""
        return self.breakpoints[[0,-1]]

    @property
    @self_empty(0)
    def hscale(self):
        return abs(self.support).max()

    @property
    def isempty(self):
        return self.funs.size == 0

    @property
    @self_empty(0)
    def vscale(self):
        return max([fun.vscale for fun in self])

    # -----------
    #  utilities
    # -----------
    def copy(self):
        return self.__class__([fun.copy() for fun in self])

    # -------------
    #  rootfinding
    # -------------
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
