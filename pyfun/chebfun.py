# -*- coding: utf-8 -*-

from collections import OrderedDict

from numpy import array
from numpy import append
from numpy import sum
from numpy import max
from numpy import nan
from numpy import full
from numpy import sort
from numpy import ones
#from numpy import inf
from numpy import linspace
from numpy import concatenate
#from numpy.linalg import norm

from matplotlib.pyplot import gca

from pyfun.bndfun import Bndfun
from pyfun.settings import DefaultPrefs
from pyfun.utilities import Subdomain
from pyfun.decorators import checkempty

from pyfun.exceptions import SubdomainGap
from pyfun.exceptions import SubdomainOverlap
from pyfun.exceptions import BadDomainArgument
from pyfun.exceptions import BadFunLengthArgument


class Chebfun(object):

    def __init__(self, funs):
        funs = array(funs)
        funs = verify(funs)
        breaks = breakdata(funs)
        self.funs = funs
        self.breaks = breaks
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
            subdomain = Subdomain(*interval)
            fun = Bndfun.initfun_adaptive(f, subdomain)
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
            subdomain = Subdomain(*interval)
            fun = Bndfun.initfun_fixedlen(f, subdomain, length)
            funs = append(funs, fun)
        return cls(funs)

    def __str__(self):
        rowcol = "row" if self.transposed else "col"
        out = "<chebfun-{},{},{}>\n".format(rowcol, self.funs.size, self.size())
        return out

    @checkempty("chebfun<empty>")
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
            "    total length = {}".format(self.size())
        return header + toprow + rowdta + btmrow + btmxtr

    def isempty(self):
        return self.funs.size == 0

    @checkempty(0)
    def size(self):
        return sum([fun.size() for fun in self])

    @checkempty(None)
    def vscale(self):
        return max([fun.vscale() for fun in self])

    def __call__(self, x):
        x = array(x)
        out = full(x.size, nan)

        # evaluate a fun when x is an interior point
        for fun in self:
            idx = fun.subdomain.isinterior(x)
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
        a, b = self.endpoints()
        ax = ax if ax else gca()
        xx = linspace(a, b, 2001)
        yy = self(xx)
        ax.plot(xx, yy, *args, **kwargs)
        return ax

    def plotcoeffs(self, ax=None, *args, **kwargs):
        ax = ax if ax else gca()
        for fun in self:
            fun.plotcoeffs(ax=ax)
        return ax

    def breakpoints(self):
        return sort(self.breaks.keys())

    def endpoints(self):
        breakpoints = self.breakpoints()
        return breakpoints[0], breakpoints[-1]

    def __iter__(self):
        return self.funs.__iter__()

    def sum(self):
        return sum([fun.sum() for fun in self])

    def cumsum(self):
        ifuns = array([fun.cumsum() for fun in self])
        return self.__class__(ifuns)

    def diff(self):
        dfuns = array([fun.diff() for fun in self])
        return self.__class__(dfuns)

    def roots(self):
        return concatenate([fun.roots() for fun in self])


def verify(funs):
    """funs is in principle arbitrary, thus it is necessary to first sort
    and then verify that the corresponding subdomains: (1) do not overlap,
    and (2) represent a complete partition of the broader approximation
    interval"""
    if funs.size == 0:
        return array([])
    else:
        subintervals = array([fun.endpoints() for fun in funs])
        leftbreakpts = array([s[0] for s in subintervals])
        idx = leftbreakpts.argsort()
        srt = subintervals[idx]
        x = srt.flatten()[1:-1]
        d = x[1::2] - x[::2]
        if (d<0).any():
            raise SubdomainOverlap
        if (d>0).any():
            raise SubdomainGap
        return array(funs[idx])

def breakdata(funs):
    """Define function values at the interior breakpoints by averaging the
    left and right limits. This method is called after verify() so we are
    guaranteed to have a fully partitioned and nonoverlapping domain."""
    if funs.size == 0:
        return OrderedDict()
    else:
        points = array([fun.endpoints() for fun in funs])
        values = array([fun.endvalues() for fun in funs])
        points = points.flatten()
        values = values.flatten()
        xl, xr = points[0], points[-1]
        yl, yr = values[0], values[-1]
        xx, yy = points[1:-1], values[1:-1]
        x = .5 * (xx[::2] + xx[1::2])
        y = .5 * (yy[::2] + yy[1::2])
        xout = append(append(xl, x), xr)
        yout = append(append(yl, y), yr)
        return OrderedDict(zip(xout, yout))
