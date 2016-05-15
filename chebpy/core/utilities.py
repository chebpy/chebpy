# -*- coding: utf-8 -*-

from __future__ import division

from collections import OrderedDict

from numpy import append
from numpy import array
from numpy import diff
from numpy import logical_and
from numpy import unique

from chebpy.core.exceptions import IntervalGap
from chebpy.core.exceptions import IntervalOverlap
from chebpy.core.exceptions import IntervalValues
from chebpy.core.exceptions import InvalidDomain
from chebpy.core.exceptions import InconsistentSupport

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
        self.values = array([a, b])
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
        """Check that another Interval object is a subinterval of self"""
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


class Domain(object):
    """Convenience class to implement Chebfun domain logic. Instances are
    intended to be created on-the-fly rather than being precomputed and stored
    as hard attributes of Chebfun"""

    def __init__(self, breakpoints):
        breakpoints = array(breakpoints)
        try:
            if breakpoints.size < 2:
                raise InvalidDomain
            if any(diff(breakpoints)<=0):
                raise InvalidDomain
        except:
            # raised if, for example, we don't have an array of numbers
            raise InvalidDomain
        self.breakpoints = breakpoints

    def __iter__(self):
        """Iterate across adajacent pairs of breakpoints"""
        # TODO: is there more pythonic way of doing this?
        return zip(self.breakpoints[:-1], self.breakpoints[1:]).__iter__()

    @classmethod
    def from_chebfun(cls, chebfun):
        """Initialise a Domain object from a Chebfun"""
        return cls(chebfun.breakpoints)

    @property
    def size(self):
        """The size of a Domain object is the number of subintervals,
        equivalent to the number of breakpoints minus one"""
        return self.breakpoints.size - 1

    @property
    def support(self):
        """The support of a Domain object is an computed as an array
        containing its first and last breakpoints"""
        return self.breakpoints[[0,-1]]

    def union(self, other):
        """Return an array denoting the union of self and other. We check
        that the support of each object is the same before proceeding."""
        # attempt to cast other to Domain object if it is not one already
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        if any(self.support!=other.support):
            raise InconsistentSupport
        all_breakpoints = append(self.breakpoints, other.breakpoints)
        new_breakpoints = unique(all_breakpoints)
        return self.__class__(new_breakpoints)

    def __eq__(self, other):
        if self.size != other.size:
            return False
        else:
            subintervals = zip(self.breakpoints, other.breakpoints)
            return all([x==y for x,y in subintervals])

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
