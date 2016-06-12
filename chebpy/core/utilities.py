# -*- coding: utf-8 -*-

from __future__ import division

from collections import OrderedDict

from numpy import append
from numpy import array
from numpy import diff
from numpy import in1d
from numpy import logical_and
from numpy import maximum
from numpy import unique

from chebpy.core.settings import DefaultPrefs
from chebpy.core.exceptions import IntervalGap
from chebpy.core.exceptions import IntervalOverlap
from chebpy.core.exceptions import IntervalValues
from chebpy.core.exceptions import InvalidDomain
from chebpy.core.exceptions import SupportMismatch
from chebpy.core.exceptions import NotSubdomain


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
        return (self.values==other.values).all()

    def __ne__(self, other):
        return not self==other

    def __call__(self, y):
        return self.formap(y)

    def __contains__(self, other):
        """Test whether another Interval object is a 'subdomain' of self"""
        a,b = self.values
        x,y = other.values
        return (a<=x) & (y<=b)

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

class Domain(object):
    """Convenience class to implement Chebfun domain logic. Instances are
    intended to be created on-the-fly rather than being precomputed and stored
    as hard attributes of Chebfun"""

    def __init__(self, breakpoints):
        try:
            breakpoints = array(breakpoints, dtype=float)
        except:
            raise InvalidDomain
        if breakpoints.size < 2:
            raise InvalidDomain
        if any(diff(breakpoints)<=0):
            raise InvalidDomain
        self.breakpoints = breakpoints

    def __iter__(self):
        """Iterate over breakpoints"""
        return self.breakpoints.__iter__()

    def __contains__(self, other):
        """Test whether another Domain object is a 'subdomain' of self"""
        a,b = self.support
        x,y = other.support
        return (a<=x) & (y<=b)

    @classmethod
    def from_chebfun(cls, chebfun):
        """Initialise a Domain object from a Chebfun"""
        return cls(chebfun.breakpoints)

    @property
    def intervals(self):
        """Generator to iterate across adajacent pairs of breakpoints,
        yielding an interval object."""
        for a,b in zip(self.breakpoints[:-1], self.breakpoints[1:]):
            yield Interval(a,b)

    @property
    def size(self):
        """The size of a Domain object is the number of breakpoints"""
        return self.breakpoints.size

    @property
    def support(self):
        """The support of a Domain object is an array containing its first and
        last breakpoints"""
        return self.breakpoints[[0,-1]]

    def union(self, other):
        """Return a Domain object representing the union of self and another
        Domain object. We first check that the support of each object
        matches."""
        if any(self.support!=other.support):
            raise SupportMismatch
        return self.merge(other)

    def merge(self, other):
        """Merge two domain objects (without checking first whether they have
        the same support)."""
        all_bpts = append(self.breakpoints, other.breakpoints)
        new_bpts = unique(all_bpts)
        mergetol = maximum(HTOL, HTOL*abs(new_bpts))
        mgd_bpts = _merge_duplicates(new_bpts, mergetol)
        return self.__class__(mgd_bpts)

    def restrict(self, other):
        """Truncate self to the support of other, retaining any interior
        breakpoints."""
        if other not in self:
            raise NotSubdomain
        dom = self.merge(other)
        a,b = other.support
        bounds = array([1-HTOL, 1+HTOL])
        lbnd, rbnd = min(a*bounds), max(b*bounds)
        pts = dom.breakpoints
        new = pts[(lbnd<=pts)&(pts<=rbnd)]
        return self.__class__(new)

    def breakpoints_in(self, other):
        """Return a Boolean array of size self.breakpoints where True indicates
        that the breakpoint is in other.breakpoints"""
        return in1d(self.breakpoints, other.breakpoints)

    def __eq__(self, other):
        """Test for pointwise equality (within a tolerance) of two Domain
        objects"""
        if self.size != other.size:
            return False
        else:
            dbpt = abs(self.breakpoints-other.breakpoints)
            htol = maximum(HTOL, HTOL*abs(self.breakpoints))
            return all(dbpt<=htol)

    def __ne__(self, other):
        return not self==other

    def __str__(self):
        return self.__class__.__name__.lower()

    def __repr__(self):
        out = self.breakpoints.__repr__()
        return out.replace("array", self.__class__.__name__.lower())



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
        points = array([fun.endpoints for fun in funs])
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
