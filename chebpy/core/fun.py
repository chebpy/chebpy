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
        raise NotImplementedError

    @abstractclassmethod
    def initempty(cls):
        raise NotImplementedError

    @abstractclassmethod
    def initfun_adaptive(cls):
        raise NotImplementedError

    @abstractclassmethod
    def initfun_fixedlen(cls):
        raise NotImplementedError

    # -------------------
    #  "private" methods
    # -------------------
    @abc.abstractmethod
    def __add__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __mul__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __neg__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __pos__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __pow__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __radd__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __rmul__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __rsub__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __sub__(self):
        raise NotImplementedError

    # ------------
    #  properties
    # ------------
    @abc.abstractproperty
    def coeffs(self):
        raise NotImplementedError

    @abc.abstractproperty
    def interval(self):
        raise NotImplementedError

    @abc.abstractproperty
    def isconst(self):
        raise NotImplementedError

    @abc.abstractproperty
    def isempty(self):
        raise NotImplementedError

    @abc.abstractproperty
    def size(self):
        raise NotImplementedError

    @abc.abstractproperty
    def support(self):
        raise NotImplementedError

    @abc.abstractproperty
    def vscale(self):
        raise NotImplementedError

    # -----------
    #  utilities
    # -----------
    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

    @abc.abstractmethod
    def restrict(self):
        raise NotImplementedError

    @abc.abstractmethod
    def simplify(self):
        raise NotImplementedError

    @abc.abstractmethod
    def values(self):
        raise NotImplementedError

    # -------------
    #  rootfinding
    # -------------
    @abc.abstractmethod
    def roots(self):
        raise NotImplementedError

    # ----------
    #  calculus
    # ----------
    @abc.abstractmethod
    def cumsum(self):
        raise NotImplementedError

    @abc.abstractmethod
    def diff(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sum(self):
        raise NotImplementedError

    # ----------
    #  plotting
    # ----------
    @abc.abstractmethod
    def plot(self):
        raise NotImplementedError

    @abc.abstractmethod
    def plotcoeffs(self):
        raise NotImplementedError
