# -*- coding: utf-8 -*-

from __future__ import division

from functools import wraps

from numpy import ones
from numpy import array
from numpy import asarray
from numpy import isscalar
from numpy import any
from numpy import isnan
from numpy import NaN

# define an abstract class method decorator:
# http://stackoverflow.com/questions/11217878/python-2-7-combine-abc-abstractmethod-and-classmethod
class abstractclassmethod(classmethod):
    __isabstractmethod__ = True
    def __init__(self, callable):
        callable.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable)

# Factory method to produce a decorator that checks whether the object
# whose classmethod is being wrapped is empty, returning the object if
# so, but returning the supplied resultif if not. (Used in chebtech.py)
# TODO: add unit test for this
def self_empty(resultif=None):
    def decorator(f):
        @wraps(f)
        def wrapper(self, *args, **kwargs):
            if self.isempty():
                if resultif is not None:
                    return resultif
                else:
                    return self.copy()
            else:
                return f(self, *args, **kwargs)
        return wrapper
    return decorator


# pre- and post-processing tasks common to bary and clenshaw
def preandpostprocess(f):

    @wraps(f)
    def thewrapper(*args, **kwargs):
        xx, akfk = args[:2]

        # are any of the first two arguments empty arrays?
        if ( asarray(xx).size==0) | (asarray(akfk).size==0 ):
            return array([])

        # is the function constant?
        elif akfk.size == 1:
            if isscalar(xx):
                return akfk[0]
            else:
                return akfk * ones(xx.size)

        # are there any NaNs in the second argument?
        elif any(isnan(akfk)):
            return NaN * ones(xx.size)

        # convert first argument to an array if it is a scalar and then
        # return the first (and only) element of the result if so
        else:
            args = list(args)
            args[0] = array([xx]) if isscalar(xx) else args[0]
            out = f(*args, **kwargs)
            return out[0] if isscalar(xx) else out

    return thewrapper

# Chebfun classmethod wrapper for __call__: ensure that we provide
# float output for float input and array output otherwise
def float_argument(f):

    @wraps(f)
    def thewrapper(self, *args, **kwargs):
        x = args[0]
        xx = array([x]) if isscalar(x) else array(x)
        # discern between the array(0.1) and array([0.1]) cases
        if xx.size == 1:
            try:
                xx[0]
            except:
                xx = array([xx])
        args = list(args)
        args[0] = xx
        out = f(self, *args, **kwargs)
        return out[0] if isscalar(x) else out
    return thewrapper
