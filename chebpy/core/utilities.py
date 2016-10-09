# -*- coding: utf-8 -*-

from __future__ import division

from collections import OrderedDict

from numpy import append
from numpy import array
from numpy import asarray
from numpy import diff
from numpy import empty
from numpy import logical_and
from numpy import maximum
from numpy import ndarray
from numpy import sort
from numpy import unique

from chebpy.core.settings import DefaultPrefs
from chebpy.core.decorators import cast_other
from chebpy.core.exceptions import IntervalGap
from chebpy.core.exceptions import IntervalOverlap
from chebpy.core.exceptions import IntervalValues
from chebpy.core.exceptions import InvalidDomain
from chebpy.core.exceptions import SupportMismatch
from chebpy.core.exceptions import NotSubdomain
from chebpy.core.exceptions import BadDomainArgument

HTOL = 5 * DefaultPrefs.eps


class Interval(object):
    """
    Utility class to implement Interval logic. The purpose of this class
    is to both enforce certain properties of domain components such as
    having exactly two monotonically increasing elements and also to
    implement the functionality of mapping to and from the unit interval.

        formap: y in [-1,1] -> x in [a,b]
        invmap: x in  [a,b] -> y in [-1,1]
        drvmap: y in [-1,1] -> x in [a,b]

    We also provide a convenience __eq__ method amd set the __call__
    method to evaluate self.formap since this is the most frequently used
    mapping operation.

    Currently only implemented for finite a and b.
    """
    def __init__(self, a=-1, b=1):
        if a >= b:
            raise IntervalValues
        self.values = array([a,b], dtype=float)
        self.formap = lambda y: .5*b*(y+1.) + .5*a*(1.-y)
        self.invmap = lambda x: (2.*x-a-b) / (b-a)
        self.drvmap = lambda y: 0.*y + .5*(b-a)
        
    def __eq__(self, other):
        """Test for equality (within a tolerance) of the numbers defining an
        Interval object"""
        dbpt = abs(self.values-other.values)
        htol = maximum(HTOL, HTOL*abs(self.values))
        return all(dbpt<=htol)

    def __ne__(self, other):
        return not self==other

    def __call__(self, y):
        return self.formap(y)

    def __contains__(self, other):
        """Test whether another Interval object is a 'subinterval' of self, to
        within a tolerance"""
        a,b = self.values
        x,y = other.values
        bounds = array([1-HTOL, 1+HTOL])
        lbnd, rbnd = min(a*bounds), max(b*bounds)
        return (lbnd<=x) & (y<=rbnd)

    def __str__(self):
        cls = self.__class__
        out = "{}({:4.2g},{:4.2g})".format(cls.__name__, *self.values)
        return out

    def __repr__(self):
        return self.__str__()

    def isinterior(self, x):
        a, b = self.values
        return logical_and(a<x, x<b)


def _merge_duplicates(arr, tols):
    """Remove duplicate entries from an input array to within array tolerance
    tols, working from left to right."""
    # TODO: pathological cases may make this break since this method works by
    # using consecutive differences. Might be better to take an average
    # (median?), rather than the left-hand value.
    idx = append(abs(diff(arr))>tols[:-1], True)
    return arr[idx]


class Domain(ndarray):
    """Numpy ndarray, with additional Chebfun-specific domain logic"""

    def __new__(cls, breakpoints):
        bpts = asarray(breakpoints, dtype=float)
        if bpts.size < 2 or any(diff(bpts)<=0):
            raise InvalidDomain
        return bpts.view(cls)

    def __contains__(self, other):
        """Checks whether one domain object is a subodomain of another (to
        within a tolerance)"""
        a,b = self.support
        x,y = other.support
        bounds = array([1-HTOL, 1+HTOL])
        lbnd, rbnd = min(a*bounds), max(b*bounds)
        return (lbnd<=x) & (y<=rbnd)

    @classmethod
    def from_chebfun(cls, chebfun):
        """Initialise a Domain object from a Chebfun"""
        return cls(chebfun.breakpoints)

    @property
    def intervals(self):
        """Iterate across adajacent pairs of breakpoints, yielding an interval
        object."""
        for a,b in zip(self[:-1], self[1:]):
            yield Interval(a,b)

    @property
    def support(self):
        """First and last breakpoints"""
        return self[[0,-1]]

    @cast_other
    def union(self, other):
        """Union of two domain objects with an initial check that the support
        of each object matches"""
        dspt = abs(self.support-other.support)
        htol = maximum(HTOL, HTOL*abs(self.support))
        if any(dspt>htol):
            raise SupportMismatch
        return self.merge(other)

    def merge(self, other):
        """Merge two domain objects without checking first whether they have
        the same support"""
        all_bpts = append(self, other)
        new_bpts = unique(all_bpts)
        mergetol = maximum(HTOL, HTOL*abs(new_bpts))
        mgd_bpts = _merge_duplicates(new_bpts, mergetol)
        return self.__class__(mgd_bpts)

    @cast_other
    def restrict(self, other):
        """Truncate self to the support of other, retaining any interior
        breakpoints"""
        if other not in self:
            raise NotSubdomain
        dom = self.merge(other)
        a,b = other.support
        bounds = array([1-HTOL, 1+HTOL])
        lbnd, rbnd = min(a*bounds), max(b*bounds)
        new = dom[(lbnd<=dom)&(dom<=rbnd)]
        return self.__class__(new)

    def breakpoints_in(self, other):
        """Return a Boolean array of size equal to self where True indicates
        that the breakpoint is in other to within the specified tolerance"""
        out = empty(self.size, dtype=bool)
        window = array([1-HTOL, 1+HTOL])
        # TODO: is there way to vectorise this?
        for idx, bpt in enumerate(self):
            lbnd, rbnd = sort(bpt*window)
            lbnd = -HTOL if abs(lbnd) < HTOL else lbnd
            rbnd = +HTOL if abs(rbnd) < HTOL else rbnd            
            isin = (lbnd<=other) & (other<=rbnd)
            out[idx] = any(isin)
        return out

    # TODO: modify this to give behaviour consistent with numpy's ==
    def __eq__(self, other):
        """Test for pointwise equality (within a tolerance) of two Domain
        objects"""
        if self.size != other.size:
            return False
        else:
            dbpt = abs(self-other)
            htol = maximum(HTOL, HTOL*abs(self))
            return all(dbpt<=htol)

    def __ne__(self, other):
        return not self==other


def _sortindex(intervals):
    """Helper function to return an index determining the ordering of the
    supplied array of interval objects. We check that the intervals (1) do not
    overlap, and (2) represent a complete partition of the broader
    approximation domain"""

    # sort by the left endpoint Interval values
    subintervals = array([x.values for x in intervals])
    leftbreakpts = array([s[0] for s in subintervals])
    idx = leftbreakpts.argsort()

    # check domain consistency
    srt = subintervals[idx]
    x = srt.flatten()[1:-1]
    d = x[1::2] - x[::2]
    if (d<0).any():
        raise IntervalOverlap
    if (d>0).any():
        raise IntervalGap

    return idx


def check_funs(funs):
    """Return an array of sorted funs.  As the name suggests, this method
    checks that the funs provided do not overlap or have gaps (the actual
    checks are performed in _sortindex)"""
    funs = array(funs)
    if funs.size == 0:
        sortedfuns = array([])
    else:
        intervals = (fun.interval for fun in funs)
        idx = _sortindex(intervals)
        sortedfuns = funs[idx]
    return sortedfuns


def compute_breakdata(funs):
    """Define function values at the interior breakpoints by averaging the
    left and right limits. This method is called after check_funs() and
    thus at the point of calling we are guaranteed to have a fully partitioned
    and nonoverlapping domain."""
    if funs.size == 0:
        return OrderedDict()
    else:
        points = array([fun.support for fun in funs])
        values = array([fun.endvalues for fun in funs])
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


def generate_funs(domain, bndfun_constructor, arglist):
    """Method used by several of the Chebfun classmethod constructors to
    generate a collection of funs."""
    domain = array(domain)
    if domain.size < 2:
        raise BadDomainArgument
    funs = array([])
    for interval in zip(domain[:-1], domain[1:]):
        interval = Interval(*interval)
        args = arglist + [interval]
        fun = bndfun_constructor(*args)
        funs = append(funs, fun)
    return funs
