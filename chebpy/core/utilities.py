from collections import OrderedDict
from collections.abc import Iterable

import numpy as np

from .settings import _preferences as prefs
from .decorators import cast_other
from .exceptions import (
    IntervalGap,
    IntervalOverlap,
    IntervalValues,
    InvalidDomain,
    SupportMismatch,
    NotSubdomain,
)


def HTOL():
    return 5 * prefs.eps


class Interval(np.ndarray):
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

    def __new__(cls, a=-1.0, b=1.0):
        if a >= b:
            raise IntervalValues
        return np.asarray((a, b), dtype=float).view(cls)

    def formap(self, y):
        a, b = self
        return 0.5 * b * (y + 1.0) + 0.5 * a * (1.0 - y)

    def invmap(self, x):
        a, b = self
        return (2.0 * x - a - b) / (b - a)

    def drvmap(self, y):
        a, b = self
        return 0.0 * y + 0.5 * (b - a)

    def __eq__(self, other):
        (a, b), (x, y) = self, other
        return (a == x) & (y == b)

    def __ne__(self, other):
        return not self == other

    def __call__(self, y):
        return self.formap(y)

    def __contains__(self, other):
        (a, b), (x, y) = self, other
        return (a <= x) & (y <= b)

    def isinterior(self, x):
        a, b = self
        return np.logical_and(a < x, x < b)

    @property
    def hscale(self):
        a, b = self
        h = max(infnorm(self), 1)
        hF = b - a  # if interval == domain: scale hscale back to 1
        hscale = max(h / hF, 1)  # else: hscale < 1
        return hscale


def _merge_duplicates(arr, tols):
    """Remove duplicate entries from an input array to within array tolerance
    tols, working from left to right."""
    # TODO: pathological cases may make this break since this method works by
    # using consecutive differences. Might be better to take an average
    # (median?), rather than the left-hand value.
    idx = np.append(np.abs(np.diff(arr)) > tols[:-1], True)
    return arr[idx]


class Domain(np.ndarray):
    """Numpy ndarray, with additional Chebfun-specific domain logic"""

    def __new__(cls, breakpoints):
        bpts = np.asarray(breakpoints, dtype=float)
        if bpts.size == 0:
            return bpts.view(cls)
        elif bpts.size < 2 or np.any(np.diff(bpts) <= 0):
            raise InvalidDomain
        else:
            return bpts.view(cls)

    def __contains__(self, other):
        """Checks whether one domain object is a subodomain of another (to
        within a tolerance)"""
        a, b = self.support
        x, y = other.support
        bounds = np.array([1 - HTOL(), 1 + HTOL()])
        lbnd, rbnd = np.min(a * bounds), np.max(b * bounds)
        return (lbnd <= x) & (y <= rbnd)

    @classmethod
    def from_chebfun(cls, chebfun):
        """Initialise a Domain object from a Chebfun"""
        return cls(chebfun.breakpoints)

    @property
    def intervals(self):
        """Iterate across adajacent pairs of breakpoints, yielding an interval
        object."""
        for a, b in zip(self[:-1], self[1:]):
            yield Interval(a, b)

    @property
    def support(self):
        """First and last breakpoints"""
        return self[[0, -1]]

    @cast_other
    def union(self, other):
        """Union of two domain objects with an initial check that the support
        of each object matches"""
        dspt = np.abs(self.support - other.support)
        htol = np.maximum(HTOL(), HTOL() * np.abs(self.support))
        if np.any(dspt > htol):
            raise SupportMismatch
        return self.merge(other)

    def merge(self, other):
        """Merge two domain objects without checking first whether they have
        the same support"""
        all_bpts = np.append(self, other)
        new_bpts = np.unique(all_bpts)
        mergetol = np.maximum(HTOL(), HTOL() * np.abs(new_bpts))
        mgd_bpts = _merge_duplicates(new_bpts, mergetol)
        return self.__class__(mgd_bpts)

    @cast_other
    def restrict(self, other):
        """Truncate self to the support of other, retaining any interior
        breakpoints"""
        if other not in self:
            raise NotSubdomain
        dom = self.merge(other)
        a, b = other.support
        bounds = np.array([1 - HTOL(), 1 + HTOL()])
        lbnd, rbnd = np.min(a * bounds), np.max(b * bounds)
        new = dom[(lbnd <= dom) & (dom <= rbnd)]
        return self.__class__(new)

    def breakpoints_in(self, other):
        """Return a Boolean array of size equal to self where True indicates
        that the breakpoint is in other to within the specified tolerance"""
        out = np.empty(self.size, dtype=bool)
        window = np.array([1 - HTOL(), 1 + HTOL()])
        # TODO: is there way to vectorise this?
        for idx, bpt in enumerate(self):
            lbnd, rbnd = np.sort(bpt * window)
            lbnd = -HTOL() if np.abs(lbnd) < HTOL() else lbnd
            rbnd = +HTOL() if np.abs(rbnd) < HTOL() else rbnd
            isin = (lbnd <= other) & (other <= rbnd)
            out[idx] = np.any(isin)
        return out

    def __eq__(self, other):
        """Test for pointwise equality (within a tolerance) of two Domain
        objects"""
        if self.size != other.size:
            return False
        else:
            dbpt = np.abs(self - other)
            htol = np.maximum(HTOL(), HTOL() * np.abs(self))
            return bool(np.all(dbpt <= htol))  # cast back to bool

    def __ne__(self, other):
        return not self == other


def _sortindex(intervals):
    """Helper function to return an index determining the ordering of the
    supplied array of interval objects. We check that the intervals (1) do not
    overlap, and (2) represent a complete partition of the broader
    approximation domain"""

    # sort by the left endpoint Interval values
    subintervals = np.array([x for x in intervals])
    leftbreakpts = np.array([s[0] for s in subintervals])
    idx = leftbreakpts.argsort()

    # check domain consistency
    srt = subintervals[idx]
    x = srt.flatten()[1:-1]
    d = x[1::2] - x[::2]
    if (d < 0).any():
        raise IntervalOverlap
    if (d > 0).any():
        raise IntervalGap

    return idx


def check_funs(funs):
    """Return an array of sorted funs.  As the name suggests, this method
    checks that the funs provided do not overlap or have gaps (the actual
    checks are performed in _sortindex)"""
    funs = np.array(funs)
    if funs.size == 0:
        sortedfuns = np.array([])
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
        points = np.array([fun.support for fun in funs])
        values = np.array([fun.endvalues for fun in funs])
        points = points.flatten()
        values = values.flatten()
        xl, xr = points[0], points[-1]
        yl, yr = values[0], values[-1]
        xx, yy = points[1:-1], values[1:-1]
        x = 0.5 * (xx[::2] + xx[1::2])
        y = 0.5 * (yy[::2] + yy[1::2])
        xout = np.append(np.append(xl, x), xr)
        yout = np.append(np.append(yl, y), yr)
        return OrderedDict(zip(xout, yout))


def generate_funs(domain, bndfun_constructor, kwds={}):
    """Method used by several of the Chebfun classmethod constructors to
    generate a collection of funs."""
    domain = Domain(domain if domain is not None else prefs.domain)
    funs = []
    for interval in domain.intervals:
        kwds = {**kwds, **{"interval": interval}}
        funs.append(bndfun_constructor(**kwds))
    return funs


def infnorm(vals):
    return np.linalg.norm(vals, np.inf)


def coerce_list(x):
    if not isinstance(x, Iterable) or isinstance(x, str):
        x = [x]
    return x
