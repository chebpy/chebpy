# -*- coding: utf-8 -*-

from __future__ import division

from numpy import array
from numpy import logical_and

from pyfun.exceptions import SubdomainValues

class Subdomain(object):
    """
    Utility class to implement subdomain logic. The purpose of this class is to
    both enforce certain properties of domain components such as having exactly
    two monotonically increasing elements and also to implement the
    functionality of mapping to and from the unit interval.

        formap: y in [-1,1] -> x in [a,b]
        invmap: x in  [a,b] -> y in [-1,1]
        drvmap: y in [-1,1] -> x in [a,b]

    We also provide a convenience __eq__ method amd set the __call__ method to
    evaluate self.formap since this will be used the most frequently used
    operation.

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
        out = "{0}([{1}, {2}])".format(cls.__name__, *self.values)
        return out

    def __repr__(self):
        return self.__str__()

    def isinterior(self, x):
        a, b = self.values
        return logical_and(a<x, x<b)
