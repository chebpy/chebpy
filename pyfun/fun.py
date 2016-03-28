# -*- coding: utf-8 -*-
"""
Placeholder class
"""

from __future__ import division

from abc import ABCMeta
from abc import abstractmethod

class Fun(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

#    @abstractmethod
#    def initconst(cls):
#        pass
#
#    @abstractmethod
#    def initempty(cls):
#        pass
#
#    @abstractmethod
#    def initfun(cls):
#        pass
#
#    @abstractmethod
#    def initfun_fixedlen(cls):
#        pass
#
#    @abstractmethod
#    def initfun_adaptive(cls):
#        pass
#
#    @abstractmethod
#    def __call__(self):
#        pass
#
#    @abstractmethod
#    def __str__(self):
#        pass
#
#    @abstractmethod
#    def __repr__(self):
#        pass
#
#    # ---------------------------------
#    #      utilities
#    # ---------------------------------
#    @abstractmethod
#    def prolong(self):
#        pass
#
#    @abstractmethod
#    def copy(self):
#        pass
#
#    @abstractmethod
#    def coeffs(self):
#        pass
#
#    @abstractmethod
#    def values(self):
#        pass
#
#    @abstractmethod
#    def size(self):
#        pass
#
#    @abstractmethod
#    def isempty(self):
#        pass
#
#    @abstractmethod
#    def isconst(self):
#        pass
#
#    @abstractmethod
#    def simplify(self):
#        pass
#
#    @abstractmethod
#    def vscale(self):
#        pass
#
#    # ---------------------------------
#    #        algebra
#    # ---------------------------------
#    @abstractmethod
#    def __add__(self):
#        pass
#
#    @abstractmethod
#    def __sub__(self):
#        pass
#
#    @abstractmethod
#    def __pos__(self):
#        pass
#
#    @abstractmethod
#    def __neg__(self):
#        pass
#
#    @abstractmethod
#    def __radd__(self):
#        pass
#
#    @abstractmethod
#    def __rsub__(self):
#        pass
#
#    @abstractmethod
#    def __mul__(self):
#        pass
#
#    @abstractmethod
#    def __rmul__(self):
#        pass
#
#    # ---------------------------------
#    #            calculus
#    # ---------------------------------
    @abstractmethod
    def sum(self):
        pass
#
#    @abstractmethod
#    def cumsum(self):
#        pass
#
#    @abstractmethod
#    def diff(self):
#        pass
#
#    # ---------------------------------
#    #            plotting
#    # ---------------------------------
#    @abstractmethod
#    def plot(self):
#        pass
#
#    @abstractmethod
#    def plotcoeffs(self):
#        pass
