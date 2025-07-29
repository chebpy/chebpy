"""Decorator functions for the ChebPy package.

This module provides various decorators used throughout the ChebPy package to
implement common functionality such as caching, handling empty objects,
pre- and post-processing of function inputs/outputs, and type conversion.
These decorators help reduce code duplication and ensure consistent behavior
across the package.
"""

from functools import wraps

import numpy as np


def cache(f: callable) -> callable:
    """Object method output caching mechanism.

    This decorator caches the output of zero-argument methods to speed up repeated
    execution of relatively expensive operations such as .roots(). Cached computations
    are stored in a dictionary called _cache which is bound to self using keys
    corresponding to the method name.

    Args:
        f (callable): The method to be cached. Must be a zero-argument method.

    Returns:
        callable: A wrapped version of the method that implements caching.

    Note:
        Can be used in principle on arbitrary objects.
    """

    # TODO: look into replacing this with one of the functools cache decorators
    @wraps(f)
    def wrapper(self):
        try:
            # f has been executed previously
            out = self._cache[f.__name__]
        except AttributeError:
            # f has not been executed previously and self._cache does not exist
            self._cache = {}
            out = self._cache[f.__name__] = f(self)
        except KeyError:  # pragma: no cover
            # f has not been executed previously, but self._cache exists
            out = self._cache[f.__name__] = f(self)
        return out

    return wrapper


def self_empty(resultif=None) -> callable:
    """Factory method to produce a decorator for handling empty objects.

    This factory creates a decorator that checks whether the object whose method
    is being wrapped is empty. If the object is empty, it returns either the supplied
    resultif value or a copy of the object. Otherwise, it executes the wrapped method.

    Args:
        resultif: Value to return when the object is empty. If None, returns a copy
            of the object instead.

    Returns:
        callable: A decorator function that implements the empty-checking logic.

    Note:
        This decorator is primarily used in chebtech.py.
    """

    # TODO: add unit test for this
    def decorator(f):
        @wraps(f)
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


def preandpostprocess(f: callable) -> callable:
    """Decorator for pre- and post-processing tasks common to bary and clenshaw.

    This decorator handles several edge cases for functions like bary and clenshaw:
    - Empty arrays in input arguments
    - Constant functions
    - NaN values in coefficients
    - Scalar vs. array inputs

    Args:
        f (callable): The function to be wrapped.

    Returns:
        callable: A wrapped version of the function with pre- and post-processing.
    """

    @wraps(f)
    def thewrapper(*args, **kwargs):
        xx, akfk = args[:2]
        # are any of the first two arguments empty arrays?
        if (np.asarray(xx).size == 0) | (np.asarray(akfk).size == 0):
            return np.array([])
        # is the function constant?
        elif akfk.size == 1:
            if np.isscalar(xx):
                return akfk[0]
            else:
                return akfk * np.ones(xx.size)
        # are there any NaNs in the second argument?
        elif np.any(np.isnan(akfk)):
            return np.nan * np.ones(xx.size)
        # convert first argument to an array if it is a scalar and then
        # return the first (and only) element of the result if so
        else:
            args = list(args)
            args[0] = np.array([xx]) if np.isscalar(xx) else args[0]
            out = f(*args, **kwargs)
            return out[0] if np.isscalar(xx) else out

    return thewrapper


def float_argument(f: callable) -> callable:
    """Decorator to ensure consistent input/output types for Chebfun __call__ method.

    This decorator ensures that when a Chebfun object is called with a float input,
    it returns a float output, and when called with an array input, it returns an
    array output. It handles various input formats including scalars, numpy arrays,
    and nested arrays.

    Args:
        f (callable): The __call__ method to be wrapped.

    Returns:
        callable: A wrapped version of the method that ensures type consistency.
    """

    @wraps(f)
    def thewrapper(self, *args, **kwargs):
        x = args[0]
        xx = np.array([x]) if np.isscalar(x) else np.array(x)
        # discern between the array(0.1) and array([0.1]) cases
        if xx.size == 1:
            if xx.ndim == 0:
                xx = np.array([xx])
        args = list(args)
        args[0] = xx
        out = f(self, *args, **kwargs)
        return out[0] if np.isscalar(x) else out

    return thewrapper


def cast_arg_to_chebfun(f: callable) -> callable:
    """Decorator to cast the first argument to a chebfun object if needed.

    This decorator attempts to convert the first argument to a chebfun object
    if it is not already one. Currently, only numeric types can be cast to chebfun.

    Args:
        f (callable): The method to be wrapped.

    Returns:
        callable: A wrapped version of the method that ensures the first argument
            is a chebfun object.
    """

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        other = args[0]
        if not isinstance(other, self.__class__):
            fun = self.initconst(args[0], self.support)
            args = list(args)
            args[0] = fun
        return f(self, *args, **kwargs)

    return wrapper


def cast_other(f: callable) -> callable:
    """Decorator to cast the first argument to the same type as self.

    This generic decorator is applied to binary operator methods to ensure that
    the first positional argument (typically 'other') is cast to the same type
    as the object on which the method is called.

    Args:
        f (callable): The binary operator method to be wrapped.

    Returns:
        callable: A wrapped version of the method that ensures type consistency
            between self and the first argument.
    """

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        cls = self.__class__
        other = args[0]
        if not isinstance(other, cls):
            args = list(args)
            args[0] = cls(other)
        return f(self, *args, **kwargs)

    return wrapper
