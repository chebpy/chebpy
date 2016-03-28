# -*- coding: utf-8 -*-
"""
Placeholder class
"""

from __future__ import division

from pyfun.classicfun import ClassicFun
from pyfun.settings import DefaultPrefs

from pyfun.chebtech import ChebTech2

techs = {
    "ChebTech2": ChebTech2,
}

class BndFun(ClassicFun):
    """Class to aproximate functions on bounded intervals [a,b]"""

    def __init__(self, _callable_, domain=None):
        domain = domain if domain is not None else DefaultPrefs.domain
        mapper = UnitIntervalMapping(*domain)
        uifunc = lambda y: _callable_(mapper.formap(y))
        self.onefun = techs[DefaultPrefs.tech].initfun(uifunc)
        self.domain = domain
        self.mapper = mapper

    def sum(self):
        a, b = self.domain
        return .5 * (b-a) * self.onefun.sum()

class UnitIntervalMapping(object):
    """
    formap: y in [-1,1] -> x in [a,b]
    invmap: x in  [a,b] -> y in [-1,1]
    drvmap: y in [-1,1] -> x in [a,b]
    
    Currently only implemented for finite a and b.
    """
    def __init__(self, a, b):
        self.formap = lambda y: .5 * b * (y + 1.) + .5 * a * (1. - y)
        self.invmap = lambda x: (2.*x - a - b) / (b - a)
        self.drvmap = lambda y: 0*y + .5*(b-a)
