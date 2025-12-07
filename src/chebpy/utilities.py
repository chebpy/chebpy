"""Utility functions and classes for the ChebPy package.

This module provides various utility functions and classes used throughout the ChebPy
package, including interval operations, domain manipulations, and tolerance functions.
It defines the core data structures for representing and manipulating intervals and domains.
"""

from collections import OrderedDict
from collections.abc import Iterable

import numpy as np

from .decorators import cast_other
from .exceptions import IntervalGap, IntervalOverlap, IntervalValues, InvalidDomain, NotSubdomain, SupportMismatch
from .settings import _preferences as prefs


def htol() -> float:
    """Return the horizontal tolerance used for interval comparisons.

    Returns:
        float: 5 times the machine epsilon from preferences.
    """
    return 5 * prefs.eps


def is_scalar_type(x) -> bool:
    """Check if x is a scalar numeric type.

    Args:
        x: Value to check

    Returns:
        bool: True if x is int, float, or numpy scalar
    """
    return isinstance(x, (int, float, np.number))


def ensure_interval(x) -> "Interval":
    """Convert various types to Interval.

    Args:
        x: Interval, list, tuple, or array-like with 2 elements

    Returns:
        Interval: Validated Interval object

    Raises:
        ValueError: If x cannot be converted to Interval
    """
    if isinstance(x, Interval):
        return x
    if hasattr(x, "__iter__") and len(x) == 2:
        return Interval(x[0], x[1])
    raise ValueError(f"Cannot convert {type(x)} to Interval")


class Interval(np.ndarray):
    """Utility class to implement Interval logic.

    The purpose of this class is to both enforce certain properties of domain
    components such as having exactly two monotonically increasing elements and
    also to implement the functionality of mapping to and from the unit interval.

    Attributes:
        formap: Maps y in [-1,1] to x in [a,b]
        invmap: Maps x in [a,b] to y in [-1,1]
        drvmap: Derivative mapping from y in [-1,1] to x in [a,b]

    Note:
        Currently only implemented for finite a and b.
        The __call__ method evaluates self.formap since this is the most
        frequently used mapping operation.
    """

    def __new__(cls, a: float = -1.0, b: float = 1.0) -> "Interval":
        """Create a new Interval instance.

        Args:
            a (float, optional): Left endpoint of the interval. Defaults to -1.0.
            b (float, optional): Right endpoint of the interval. Defaults to 1.0.

        Raises:
            IntervalValues: If a >= b.

        Returns:
            Interval: A new Interval instance.
        """
        if a >= b:
            raise IntervalValues
        return np.asarray((a, b), dtype=float).view(cls)

    def formap(self, y: float | np.ndarray) -> float | np.ndarray:
        """Map from the reference interval [-1,1] to this interval [a,b].

        Args:
            y (float or numpy.ndarray): Points in the reference interval [-1,1].

        Returns:
            float or numpy.ndarray: Corresponding points in the interval [a,b].
        """
        a, b = self
        return 0.5 * b * (y + 1.0) + 0.5 * a * (1.0 - y)

    def invmap(self, x: float | np.ndarray) -> float | np.ndarray:
        """Map from this interval [a,b] to the reference interval [-1,1].

        Args:
            x (float or numpy.ndarray): Points in the interval [a,b].

        Returns:
            float or numpy.ndarray: Corresponding points in the reference interval [-1,1].
        """
        a, b = self
        return (2.0 * x - a - b) / (b - a)

    def drvmap(self, y: float | np.ndarray) -> float | np.ndarray:
        """Compute the derivative of the forward map.

        Args:
            y (float or numpy.ndarray): Points in the reference interval [-1,1].

        Returns:
            float or numpy.ndarray: Derivative values at the corresponding points.
        """
        a, b = self  # pragma: no cover
        return 0.0 * y + 0.5 * (b - a)  # pragma: no cover

    def __eq__(self, other: "Interval") -> bool:
        """Check if two intervals are equal.

        Args:
            other (Interval): Another interval to compare with.

        Returns:
            bool: True if the intervals have the same endpoints, False otherwise.
        """
        (a, b), (x, y) = self, other
        return (a == x) & (y == b)

    def __ne__(self, other: "Interval") -> bool:
        """Check if two intervals are not equal.

        Args:
            other (Interval): Another interval to compare with.

        Returns:
            bool: True if the intervals have different endpoints, False otherwise.
        """
        return not self == other

    def __call__(self, y: float | np.ndarray) -> float | np.ndarray:
        """Map points from [-1,1] to this interval (shorthand for formap).

        Args:
            y (float or numpy.ndarray): Points in the reference interval [-1,1].

        Returns:
            float or numpy.ndarray: Corresponding points in the interval [a,b].
        """
        return self.formap(y)

    def __contains__(self, other: "Interval") -> bool:
        """Check if another interval is contained within this interval.

        Args:
            other (Interval): Another interval to check.

        Returns:
            bool: True if other is contained within this interval, False otherwise.
        """
        (a, b), (x, y) = self, other
        return (a <= x) & (y <= b)

    def isinterior(self, x: float | np.ndarray) -> bool | np.ndarray:
        """Check if points are strictly in the interior of the interval.

        Args:
            x (float or numpy.ndarray): Points to check.

        Returns:
            bool or numpy.ndarray: Boolean array indicating which points are in the interior.
        """
        a, b = self
        return np.logical_and(a < x, x < b)

    @property
    def hscale(self) -> float:
        """Calculate the horizontal scale factor of the interval.

        Returns:
            float: The horizontal scale factor.
        """
        a, b = self
        h = max(infnorm(self), 1)
        h_factor = b - a  # if interval == domain: scale hscale back to 1
        hscale = max(h / h_factor, 1)  # else: hscale < 1
        return hscale


def _merge_duplicates(arr: np.ndarray, tols: np.ndarray) -> np.ndarray:
    """Remove duplicate entries from an input array within specified tolerances.

    This function works from left to right, keeping the first occurrence of
    values that are within tolerance of each other.

    Args:
        arr (numpy.ndarray): Input array to remove duplicates from.
        tols (numpy.ndarray): Array of tolerance values for each pair of adjacent elements.
            Should have length one less than arr.

    Returns:
        numpy.ndarray: Array with duplicates removed.

    Note:
        Pathological cases may cause issues since this method works by using
        consecutive differences. It might be better to take an average (median?),
        rather than the left-hand value.
    """
    idx = np.append(np.abs(np.diff(arr)) > tols[:-1], True)
    return arr[idx]


class Domain(np.ndarray):
    """Numpy ndarray with additional Chebfun-specific domain logic.

    A Domain represents a collection of breakpoints that define a piecewise domain.
    It provides methods for manipulating and comparing domains, as well as
    generating intervals between adjacent breakpoints.

    Attributes:
        intervals: Generator yielding Interval objects between adjacent breakpoints.
        support: First and last breakpoints of the domain.
    """

    def __new__(cls, breakpoints):
        """Create a new Domain instance.

        Args:
            breakpoints (array-like): Collection of monotonically increasing breakpoints.
                Must have at least 2 elements.

        Raises:
            InvalidDomain: If breakpoints has fewer than 2 elements or is not monotonically increasing.

        Returns:
            Domain: A new Domain instance.
        """
        bpts = np.asarray(breakpoints, dtype=float)
        if bpts.size == 0:
            return bpts.view(cls)
        elif bpts.size < 2 or np.any(np.diff(bpts) <= 0):
            raise InvalidDomain
        else:
            return bpts.view(cls)

    def __contains__(self, other: "Domain") -> bool:
        """Check whether one domain object is a subdomain of another (within tolerance).

        Args:
            other (Domain): Another domain to check.

        Returns:
            bool: True if other is contained within this domain (within tolerance), False otherwise.
        """
        a, b = self.support
        x, y = other.support
        bounds = np.array([1 - htol(), 1 + htol()])
        lbnd, rbnd = np.min(a * bounds), np.max(b * bounds)
        return (lbnd <= x) & (y <= rbnd)

    @classmethod
    def from_chebfun(cls, chebfun):
        """Initialize a Domain object from a Chebfun.

        Args:
            chebfun: A Chebfun object with breakpoints.

        Returns:
            Domain: A new Domain instance with the same breakpoints as the Chebfun.
        """
        return cls(chebfun.breakpoints)

    @property
    def intervals(self) -> Iterable[Interval]:
        """Generate Interval objects between adjacent breakpoints.

        Yields:
            Interval: Interval objects for each pair of adjacent breakpoints.
        """
        for a, b in zip(self[:-1], self[1:]):
            yield Interval(a, b)

    @property
    def support(self) -> Interval:
        """Get the first and last breakpoints of the domain.

        Returns:
            numpy.ndarray: Array containing the first and last breakpoints.
        """
        return self[[0, -1]]

    @cast_other
    def union(self, other: "Domain") -> "Domain":
        """Create a union of two domain objects with matching support.

        Args:
            other (Domain): Another domain to union with.

        Raises:
            SupportMismatch: If the supports of the two domains don't match within tolerance.

        Returns:
            Domain: A new Domain containing all breakpoints from both domains.
        """
        dspt = np.abs(self.support - other.support)
        tolerance = np.maximum(htol(), htol() * np.abs(self.support))
        if np.any(dspt > tolerance):
            raise SupportMismatch
        return self.merge(other)

    def merge(self, other: "Domain") -> "Domain":
        """Merge two domain objects without checking if they have the same support.

        Args:
            other (Domain): Another domain to merge with.

        Returns:
            Domain: A new Domain containing all breakpoints from both domains.
        """
        all_bpts = np.append(self, other)
        new_bpts = np.unique(all_bpts)
        mergetol = np.maximum(htol(), htol() * np.abs(new_bpts))
        mgd_bpts = _merge_duplicates(new_bpts, mergetol)
        return self.__class__(mgd_bpts)

    @cast_other
    def restrict(self, other: "Domain") -> "Domain":
        """Truncate self to the support of other, retaining any interior breakpoints.

        Args:
            other (Domain): Domain to restrict to.

        Raises:
            NotSubdomain: If other is not a subdomain of self.

        Returns:
            Domain: A new Domain with breakpoints from self restricted to other's support.
        """
        if other not in self:
            raise NotSubdomain
        dom = self.merge(other)
        a, b = other.support
        bounds = np.array([1 - htol(), 1 + htol()])
        lbnd, rbnd = np.min(a * bounds), np.max(b * bounds)
        new = dom[(lbnd <= dom) & (dom <= rbnd)]
        return self.__class__(new)

    def breakpoints_in(self, other: "Domain") -> np.ndarray:
        """Check which breakpoints are in another domain within tolerance.

        Args:
            other (Domain): Domain to check against.

        Returns:
            numpy.ndarray: Boolean array of size equal to self where True indicates
                that the breakpoint is in other within the specified tolerance.
        """
        out = np.empty(self.size, dtype=bool)
        window = np.array([1 - htol(), 1 + htol()])
        # TODO: is there way to vectorise this?
        for idx, bpt in enumerate(self):
            lbnd, rbnd = np.sort(bpt * window)
            lbnd = -htol() if np.abs(lbnd) < htol() else lbnd
            rbnd = +htol() if np.abs(rbnd) < htol() else rbnd
            isin = (lbnd <= other) & (other <= rbnd)
            out[idx] = np.any(isin)
        return out

    def __eq__(self, other: "Domain") -> bool:
        """Test for pointwise equality (within a tolerance) of two Domain objects.

        Args:
            other (Domain): Another domain to compare with.

        Returns:
            bool: True if domains have the same size and all breakpoints match within tolerance.
        """
        if self.size != other.size:
            return False
        else:
            dbpt = np.abs(self - other)
            tolerance = np.maximum(htol(), htol() * np.abs(self))
            return bool(np.all(dbpt <= tolerance))  # cast back to bool

    def __ne__(self, other: "Domain") -> bool:
        """Test for inequality of two Domain objects.

        Args:
            other (Domain): Another domain to compare with.

        Returns:
            bool: True if domains differ in size or any breakpoints don't match within tolerance.
        """
        return not self == other


def _sortindex(intervals: list[Interval]) -> np.ndarray:
    """Return an index determining the ordering of interval objects.

    This helper function checks that the intervals:
    1. Do not overlap
    2. Represent a complete partition of the broader approximation domain

    Args:
        intervals (array-like): Array of Interval objects to sort.

    Returns:
        numpy.ndarray: Index array for sorting the intervals.

    Raises:
        IntervalOverlap: If any intervals overlap.
        IntervalGap: If there are gaps between intervals.
    """
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


def check_funs(funs: list) -> np.ndarray:
    """Return an array of sorted funs with validation checks.

    This function checks that the provided funs do not overlap or have gaps
    between their intervals. The actual checks are performed in _sortindex.

    Args:
        funs (array-like): Array of function objects with interval attributes.

    Returns:
        numpy.ndarray: Sorted array of funs.

    Raises:
        IntervalOverlap: If any function intervals overlap.
        IntervalGap: If there are gaps between function intervals.
    """
    funs = np.array(funs)
    if funs.size == 0:
        sortedfuns = np.array([])
    else:
        intervals = (fun.interval for fun in funs)
        idx = _sortindex(intervals)
        sortedfuns = funs[idx]
    return sortedfuns


def compute_breakdata(funs: np.ndarray) -> OrderedDict:
    """Define function values at breakpoints by averaging left and right limits.

    This function computes values at breakpoints by averaging the left and right
    limits of adjacent functions. It is typically called after check_funs(),
    which ensures that the domain is fully partitioned and non-overlapping.

    Args:
        funs (numpy.ndarray): Array of function objects with support and endvalues attributes.

    Returns:
        OrderedDict: Dictionary mapping breakpoints to function values.
    """
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


def generate_funs(domain: Domain | list | None, bndfun_constructor: callable, kwds: dict = {}) -> list:
    """Generate a collection of function objects over a domain.

    This method is used by several of the Chebfun classmethod constructors to
    generate a collection of function objects over the specified domain.

    Args:
        domain (array-like or None): Domain breakpoints. If None, uses default domain from preferences.
        bndfun_constructor (callable): Constructor function for creating function objects.
        kwds (dict, optional): Additional keyword arguments to pass to the constructor. Defaults to {}.

    Returns:
        list: List of function objects covering the domain.
    """
    domain = Domain(domain if domain is not None else prefs.domain)
    funs = []
    for interval in domain.intervals:
        kwds = {**kwds, **{"interval": interval}}
        funs.append(bndfun_constructor(**kwds))
    return funs


def infnorm(vals: np.ndarray) -> float:
    """Calculate the infinity norm of an array.

    Args:
        vals (array-like): Input array.

    Returns:
        float: The infinity norm (maximum absolute value) of the input.
    """
    return np.linalg.norm(vals, np.inf)


def coerce_list(x: object) -> list | Iterable:
    """Convert a non-iterable object to a list containing that object.

    If the input is already an iterable (except strings), it is returned unchanged.
    Strings are treated as non-iterables and wrapped in a list.

    Args:
        x: Input object to coerce to a list if necessary.

    Returns:
        list or iterable: The input wrapped in a list if it was not an iterable,
            or the original input if it was already an iterable (except strings).
    """
    if not isinstance(x, Iterable) or isinstance(x, str):  # pragma: no cover
        x = [x]
    return x
