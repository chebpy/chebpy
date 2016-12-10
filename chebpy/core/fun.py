# -*- coding: utf-8 -*-

from __future__ import division

import abc

from chebpy.core.decorators import abstractclassmethod

class Fun(object):

    __metaclass__ = abc.ABCMeta

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
    @abc.abstractmethod
    def __add__(self):
        pass

    @abc.abstractmethod
    def __call__(self):
        pass

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def __mul__(self):
        pass

    @abc.abstractmethod
    def __neg__(self):
        pass

    @abc.abstractmethod
    def __pos__(self):
        pass

    @abc.abstractmethod
    def __pow__(self):
        pass

    @abc.abstractmethod
    def __radd__(self):
        pass

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def __rmul__(self):
        pass

    @abc.abstractmethod
    def __rsub__(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def __sub__(self):
        pass

    # ------------
    #  properties
    # ------------
    @abc.abstractproperty
    def coeffs(self):
        pass

    @abc.abstractproperty
    def interval(self):
        pass

    @abc.abstractproperty
    def isconst(self):
        pass

    @abc.abstractproperty
    def isempty(self):
        pass

    @abc.abstractproperty
    def size(self):
        pass

    @abc.abstractproperty
    def support(self):
        pass

    @abc.abstractproperty
    def vscale(self):
        pass

    # -----------
    #  utilities
    # -----------
    @abc.abstractmethod
    def copy(self):
        pass

    @abc.abstractmethod
    def restrict(self):
        pass

    @abc.abstractmethod
    def simplify(self):
        pass

    @abc.abstractmethod
    def values(self):
        pass

    # -------------
    #  rootfinding
    # -------------
    @abc.abstractmethod
    def roots(self):
        pass

    # ----------
    #  calculus
    # ----------
    @abc.abstractmethod
    def cumsum(self):
        pass

    @abc.abstractmethod
    def diff(self):
        pass

    @abc.abstractmethod
    def sum(self):
        pass

    # ----------
    #  plotting
    # ----------
    @abc.abstractmethod
    def plot(self):
        pass

    @abc.abstractmethod
    def plotcoeffs(self):
        pass
