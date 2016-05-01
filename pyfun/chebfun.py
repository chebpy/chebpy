# -*- coding: utf-8 -*-

from numpy import array
from numpy import append
from numpy import sum
from numpy import max
from numpy import nan
from numpy import full
from numpy import sort
from numpy import ones
from numpy import linspace
from numpy import concatenate

from matplotlib.pyplot import gca

from pyfun.bndfun import Bndfun
from pyfun.settings import DefaultPrefs
from pyfun.utilities import Interval
from pyfun.utilities import Domain
from pyfun.utilities import sortandverify
from pyfun.utilities import breakdata
from pyfun.decorators import emptycase
from pyfun.decorators import singletoncase
from pyfun.exceptions import BadDomainArgument
from pyfun.exceptions import BadFunLengthArgument


class Chebfun(object):

    def __init__(self, funs):
        # TODO: there is a minor inefficiency here - sortandverify()
        # and Domain.from_funs() do the same integrity check
        self.funs = sortandverify(funs)
        self.domain = Domain.init_from_funs(funs)
        self.breaks = breakdata(self.funs)
        self.transposed = False

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

    def __str__(self):
        rowcol = "row" if self.transposed else "col"
        out = "<chebfun-{},{},{}>\n".format(rowcol, self.funs.size, self.size())
        return out

    @emptycase("chebfun<empty>")
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
            endpts = fun.endpoints()
            xl, xr = endpts
            fl, fr = fun(endpts)
            row = tmplat.format(xl, xr, fun.size(), fl, fr)
            rowdta += row
        btmrow = "vertical scale = {:3.2g}".format(self.vscale())
        btmxtr = "" if numpcs == 1 else \
            "    total length = {}".format(sum([f.size() for f in self]))
        return header + toprow + rowdta + btmrow + btmxtr

    def isempty(self):
        return self.funs.size == 0

    def copy(self):
        return self.__class__([fun.copy() for fun in self])

    @emptycase(0)
    def vscale(self):
        return max([fun.vscale() for fun in self])

    @emptycase(0)
    def hscale(self):
        return abs(self.endpoints()).max()

    @emptycase(array([]))
    @singletoncase
    def __call__(self, x):

        # initialise output
        out = full(x.size, nan)

        # evaluate a fun when x is an interior point
        for fun in self:
            idx = fun.interval.isinterior(x)
            out[idx] = fun(x[idx])

        # evaluate the breakpoint data for x at a breakpoint
        breakpoints = self.breakpoints()
        for breakpoint in breakpoints:
            out[x==breakpoint] = self.breaks[breakpoint]

        # first and last funs used to evaluate outside of the chebfun domain
        lpts, rpts = x < breakpoints[0], x > breakpoints[-1]
        out[lpts] = self.funs[0](x[lpts])
        out[rpts] = self.funs[-1](x[rpts])
        return out

    def plot(self, ax=None, *args, **kwargs):
        ax = ax if ax else gca()
        a, b = self.endpoints()
        xx = linspace(a, b, 2001)
        ax.plot(xx, self(xx), *args, **kwargs)
        return ax

    def plotcoeffs(self, ax=None, *args, **kwargs):
        ax = ax if ax else gca()
        for fun in self:
            fun.plotcoeffs(ax=ax)
        return ax

    def breakpoints(self):
        return sort(self.breaks.keys())

    @emptycase(array([]))
    def endpoints(self):
        breakpoints = self.breakpoints()
        return array([breakpoints[0], breakpoints[-1]])

    def __iter__(self):
        return self.funs.__iter__()

    def sum(self):
        return sum([fun.sum() for fun in self])

    def cumsum(self):
        newfuns = []
        prevfun = None
        for fun in self:
            integral = fun.cumsum()
            if prevfun:
                # enforce continuity by adding the function value
                # at the right endpoint of the previous fun
                _, fb = prevfun.endvalues()
                integral = integral + fb
            newfuns.append(integral)
            prevfun = integral
        return self.__class__(newfuns)

    def diff(self):
        dfuns = array([fun.diff() for fun in self])
        return self.__class__(dfuns)

    @emptycase(array([]))
    def roots(self):
        allrts = []
        prvrts = array([])
        htol = 1e2 * self.hscale() * DefaultPrefs.eps
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
