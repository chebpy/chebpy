# -*- coding: utf-8 -*-

from __future__ import division

import functools
import numpy as np

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
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            if self.isempty:
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
    @functools.wraps(f)
    def thewrapper(*args, **kwargs):
        xx, akfk = args[:2]
        # are any of the first two arguments empty arrays?
        if (np.asarray(xx).size==0) | (np.asarray(akfk).size==0):
            return np.array([])
        # is the function constant?
        elif akfk.size == 1:
            if np.isscalar(xx):
                return akfk[0]
            else:
                return akfk*np.ones(xx.size)
        # are there any NaNs in the second argument?
        elif np.any(np.isnan(akfk)):
            return np.nan*np.ones(xx.size)
        # convert first argument to an array if it is a scalar and then
        # return the first (and only) element of the result if so
        else:
            args = list(args)
            args[0] = np.array([xx]) if np.isscalar(xx) else args[0]
            out = f(*args, **kwargs)
            return out[0] if np.isscalar(xx) else out

    return thewrapper

# Chebfun classmethod wrapper for __call__: ensure that we provide
# float output for float input and array output otherwise
def float_argument(f):
    @functools.wraps(f)
    def thewrapper(self, *args, **kwargs):
        x = args[0]
        xx = np.array([x]) if np.isscalar(x) else np.array(x)
        # discern between the array(0.1) and array([0.1]) cases
        if xx.size == 1:
            try:
                xx[0]
            except:
                xx = np.array([xx])
        args = list(args)
        args[0] = xx
        out = f(self, *args, **kwargs)
        return out[0] if np.isscalar(x) else out
    return thewrapper

# attempt to cast the first argument to chebfun if is not so already. The only
# castable type at this point is a numeric type
def cast_arg_to_chebfun(f):
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        other = args[0]
        if not isinstance(other, self.__class__):
            fun = self.initconst(args[0], self.support)
            args = list(args)
            args[0] = fun
        return f(self, *args, **kwargs)
    return wrapper

def cast_other(f):
    """Generic wrapper to be applied to binary operator type class methods and
    whose purpose is to cast the second positional argument to the type self"""
    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        cls   = self.__class__
        other = args[0]
        if not isinstance(other, cls):
            args = list(args)
            args[0] = cls(other)
        return f(self, *args, **kwargs)
    return wrapper