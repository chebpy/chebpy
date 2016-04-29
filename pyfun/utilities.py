# -*- coding: utf-8 -*-

from __future__ import division

from collections import OrderedDict

from numpy import append
from numpy import array
from numpy import logical_and

from pyfun.exceptions import SubdomainGap
from pyfun.exceptions import SubdomainOverlap
from pyfun.exceptions import SubdomainValues

class Subdomain(object):
    """
    Utility class to implement subdomain logic. The purpose of this class
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
            raise SubdomainValues
        self.values = array([a, b])
        self.formap = lambda y: .5*b*(y+1.) + .5*a*(1.-y)
        self.invmap = lambda x: (2.*x-a-b) / (b-a)
        self.drvmap = lambda y: 0.*y + .5*(b-a)
        
    def __eq__(self, other):
        return (self.values == other.values).all() 

    def __call__(self, y):
        return self.formap(y)

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
    """Convenience class to express key relationships between collections
    of Subdomain objects"""

    def __init__(self, subdomains):
        subdomains = array(subdomains)
        if subdomains.size == 0:
            sortedsubdomains = array([])
        else:
            idx = sortindex(subdomains)
            sortedsubdomains = subdomains[idx]
        self.subdomains = sortedsubdomains

    @classmethod
    def init_from_funs(cls, funs):
        subdomains = [fun.subdomain for fun in funs]
        return cls(subdomains)

    def __str__(self):
        out = "Domain("
        for s in self.subdomains:
            out += "\n    " + str(s)
        out += "\n)"
        return out

    def __repr__(self):
        return self.__str__()


def sortindex(subdomains):
    """Return an index determining the ordering of the subdomains.
    The methods ensures that the subdomains: (1) do not overlap, and
    (2) represent a complete partition of the broader domain"""

    # sort by the left endpoint subdomain values
    subintervals = array([x.values for x in subdomains])
    leftbreakpts = array([s[0] for s in subintervals])
    idx = leftbreakpts.argsort()

    # check domain consistency
    srt = subintervals[idx]
    x = srt.flatten()[1:-1]
    d = x[1::2] - x[::2]
    if (d<0).any():
        raise SubdomainOverlap
    if (d>0).any():
        raise SubdomainGap

    return idx

# TODO: move elsewhere (currently being tested in test_chebfun)
def sortandverify(funs):
    """Return an array of sorted funs and a corresponding Domain
    object"""
    funs = array(funs)
    if funs.size == 0:
        sortedfuns = array([])
        return sortedfuns
    else:
        subdomains = array([fun.subdomain for fun in funs])
        idx = sortindex(subdomains)
        sortedfuns = array(funs[idx])
        return sortedfuns

# TODO: move elsewhere (currently being tested in test_chebfun)
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
