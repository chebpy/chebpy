# -*- coding: utf-8 -*-
"""
Fun Class
"""

from __future__ import division

from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty

from chebpy.core.decorators import abstractclassmethod

class Fun(object):

    __metaclass__ = ABCMeta

    # --------------------------
    #  alternative constructors
    # --------------------------
    @abstractclassmethod
    def initconst(cls):
        pass

    @abstractclassmethod
    def initempty(cls):
        pass

    @abstractclassmethod
    def initfun_adaptive(cls):
        pass

    @abstractclassmethod
    def initfun_fixedlen(cls):
        pass

    # -------------------
    #  "private" methods
    # -------------------
    @abstractmethod
    def __add__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __mul__(self):
        pass

    @abstractmethod
    def __neg__(self):
        pass

    @abstractmethod
    def __pos__(self):
        pass

    @abstractmethod
    def __radd__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __rmul__(self):
        pass

    @abstractmethod
    def __rsub__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __sub__(self):
        pass

    # ------------
    #  properties
    # ------------
    @abstractproperty
    def coeffs(self):
        pass

    @abstractproperty
    def interval(self):
        pass

    @abstractproperty
    def isconst(self):
        pass

    @abstractproperty
    def isempty(self):
        pass

    @abstractproperty
    def size(self):
        pass

    @abstractproperty
    def support(self):
        pass

    @abstractproperty
    def vscale(self):
        pass

    # -----------
    #  utilities
    # -----------
    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def restrict(self):
        pass

    @abstractmethod
    def simplify(self):
        pass

    @abstractmethod
    def values(self):
        pass

    # -------------
    #  rootfinding
    # -------------
    @abstractmethod
    def roots(self):
        pass

    # ----------
    #  calculus
    # ----------
    @abstractmethod
    def cumsum(self):
        pass

    @abstractmethod
    def diff(self):
        pass

    @abstractmethod
    def sum(self):
        pass

    # ----------
    #  plotting
    # ----------
    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def plotcoeffs(self):
        pass
