"""Implementation of the Classicfun class for functions on arbitrary intervals.

This module provides the Classicfun class, which represents functions on arbitrary intervals
by mapping them to a standard domain [-1, 1] and using a Onefun representation.
"""

from abc import ABC

import matplotlib.pyplot as plt
import numpy as np

from .chebtech import Chebtech
from .decorators import self_empty
from .exceptions import IntervalMismatch, NotSubinterval
from .fun import Fun
from .plotting import plotfun
from .settings import _preferences as prefs
from .trigtech import Trigtech
from .utilities import Interval

techdict = {
    "Chebtech": Chebtech,
    "Trigtech": Trigtech,
}


class Classicfun(Fun, ABC):
    """Abstract base class for functions defined on arbitrary intervals using a mapped representation.

    This class implements the Fun interface for functions defined on arbitrary intervals
    by mapping them to a standard domain [-1, 1] and using a Onefun representation
    (such as Chebtech) on that standard domain.

    The Classicfun class serves as a base class for specific implementations like Bndfun.
    It handles the mapping between the arbitrary interval and the standard domain,
    delegating the actual function representation to the underlying Onefun object.
    """

    # --------------------------
    #  alternative constructors
    # --------------------------
    @classmethod
    def initempty(cls):
        """Initialize an empty function.

        This constructor creates an empty function representation, which is
        useful as a placeholder or for special cases. The interval has no
        relevance to the emptiness status of a Classicfun, so we arbitrarily
        set it to be the default interval [-1, 1].

        Returns:
            Classicfun: A new empty instance.
        """
        interval = Interval()
        onefun = techdict[prefs.tech].initempty(interval=interval)
        return cls(onefun, interval)

    @classmethod
    def initconst(cls, c, interval):
        """Initialize a constant function.

        This constructor creates a function that represents a constant value
        on the specified interval.

        Args:
            c: The constant value.
            interval: The interval on which to define the function.

        Returns:
            Classicfun: A new instance representing the constant function f(x) = c.
        """
        onefun = techdict[prefs.tech].initconst(c, interval=interval)
        return cls(onefun, interval)

    @classmethod
    def initidentity(cls, interval):
        """Initialize the identity function f(x) = x.

        This constructor creates a function that represents f(x) = x
        on the specified interval.

        Args:
            interval: The interval on which to define the identity function.

        Returns:
            Classicfun: A new instance representing the identity function.
        """
        onefun = techdict[prefs.tech].initvalues(np.asarray(interval), interval=interval)
        return cls(onefun, interval)

    @classmethod
    def initfun_adaptive(cls, f, interval, maxpow2=None):
        """Initialize from a callable function using adaptive sampling.

        This constructor determines the appropriate number of points needed to
        represent the function to the specified tolerance using an adaptive algorithm.

        Args:
            f (callable): The function to be approximated.
            interval: The interval on which to define the function.
            maxpow2 (int, optional): Maximum power of 2 for adaptive refinement.
                Used during splitting to limit the size of each piece.

        Returns:
            Classicfun: A new instance representing the function f.
        """
        onefun = techdict[prefs.tech].initfun_adaptive(lambda y: f(interval(y)), interval=interval, maxpow2=maxpow2)
        return cls(onefun, interval)

    @classmethod
    def initfun_fixedlen(cls, f, interval, n):
        """Initialize from a callable function using a fixed number of points.

        This constructor uses a specified number of points to represent the function,
        rather than determining the number adaptively.

        Args:
            f (callable): The function to be approximated.
            interval: The interval on which to define the function.
            n (int): The number of points to use.

        Returns:
            Classicfun: A new instance representing the function f.
        """
        onefun = techdict[prefs.tech].initfun(lambda y: f(interval(y)), n, interval=interval)
        return cls(onefun, interval)

    # -------------------
    #  'private' methods
    # -------------------
    def __call__(self, x, how=None):
        """Evaluate the function at points x.

        This method evaluates the function at the specified points by mapping them
        to the standard domain [-1, 1] and evaluating the underlying onefun.

        Args:
            x (float or array-like): Points at which to evaluate the function.
            how (str, optional): Method to use for evaluation. If None, uses the default
                method for the onefun type ("clenshaw" for Chebtech, "fft" for Trigtech).

        Returns:
            float or array-like: The value(s) of the function at the specified point(s).
                Returns a scalar if x is a scalar, otherwise an array of the same size as x.
        """
        # Determine default evaluation method based on onefun type
        if how is None:
            if isinstance(self.onefun, Trigtech):
                how = "fft"
            else:
                how = "clenshaw"

        # For Trigtech, do not remap coordinates as Trigtech is defined directly on its interval
        # For Chebtech, remap from [a,b] to [-1,1]
        if isinstance(self.onefun, Trigtech):
            y = x
        else:
            y = self.interval.invmap(x)

        return self.onefun(y, how)

    def __init__(self, onefun, interval):
        """Initialize a new Classicfun instance.

        This method initializes a new function representation on the specified interval
        using the provided onefun object for the standard domain representation.

        Args:
            onefun: The Onefun object representing the function on [-1, 1].
            interval: The Interval object defining the domain of the function.
        """
        self.onefun = onefun
        self._interval = interval

    def __repr__(self):  # pragma: no cover
        """Return a string representation of the function.

        This method returns a string representation of the function that includes
        the class name, support interval, and size.

        Returns:
            str: A string representation of the function.
        """
        out = "{0}([{2}, {3}], {1})".format(self.__class__.__name__, self.size, *self.support)
        return out

    # ------------
    #  properties
    # ------------
    @property
    def coeffs(self):
        """Get the coefficients of the function representation.

        This property returns the coefficients used in the function representation,
        delegating to the underlying onefun object.

        Returns:
            array-like: The coefficients of the function representation.
        """
        return self.onefun.coeffs

    @property
    def endvalues(self):
        """Get the values of the function at the endpoints of its interval.

        This property evaluates the function at the endpoints of its interval
        of definition.

        Returns:
            numpy.ndarray: Array containing the function values at the endpoints
                of the interval [a, b].
        """
        return self.__call__(self.support)

    @property
    def interval(self):
        """Get the interval on which this function is defined.

        This property returns the interval object representing the domain
        of definition for this function.

        Returns:
            Interval: The interval on which this function is defined.
        """
        return self._interval

    @property
    def isconst(self):
        """Check if this function represents a constant.

        This property determines whether the function is constant (i.e., f(x) = c
        for some constant c) over its interval of definition, delegating to the
        underlying onefun object.

        Returns:
            bool: True if the function is constant, False otherwise.
        """
        return self.onefun.isconst

    @property
    def iscomplex(self):
        """Check if this function has complex values.

        This property determines whether the function has complex values or is
        purely real-valued, delegating to the underlying onefun object.

        Returns:
            bool: True if the function has complex values, False otherwise.
        """
        return self.onefun.iscomplex

    @property
    def isempty(self):
        """Check if this function is empty.

        This property determines whether the function is empty, which is a special
        state used as a placeholder or for special cases, delegating to the
        underlying onefun object.

        Returns:
            bool: True if the function is empty, False otherwise.
        """
        return self.onefun.isempty

    @property
    def size(self):
        """Get the size of the function representation.

        This property returns the number of coefficients or other measure of the
        complexity of the function representation, delegating to the underlying
        onefun object.

        Returns:
            int: The size of the function representation.
        """
        return self.onefun.size

    @property
    def support(self):
        """Get the support interval of this function.

        This property returns the interval on which this function is defined,
        represented as a numpy array with two elements [a, b].

        Returns:
            numpy.ndarray: Array containing the endpoints of the interval.
        """
        return np.asarray(self.interval)

    @property
    def vscale(self):
        """Get the vertical scale of the function.

        This property returns a measure of the range of function values, typically
        the maximum absolute value of the function on its interval of definition,
        delegating to the underlying onefun object.

        Returns:
            float: The vertical scale of the function.
        """
        return self.onefun.vscale

    # -----------
    #  utilities
    # -----------

    def imag(self):
        """Get the imaginary part of this function.

        This method returns a new function representing the imaginary part of this function.
        If this function is real-valued, returns a zero function.

        Returns:
            Classicfun: A new function representing the imaginary part of this function.
        """
        if self.iscomplex:
            return self.__class__(self.onefun.imag(), self.interval)
        else:
            return self.initconst(0, interval=self.interval)

    def real(self):
        """Get the real part of this function.

        This method returns a new function representing the real part of this function.
        If this function is already real-valued, returns this function.

        Returns:
            Classicfun: A new function representing the real part of this function.
        """
        if self.iscomplex:
            return self.__class__(self.onefun.real(), self.interval)
        else:
            return self

    def restrict(self, subinterval):
        """Restrict this function to a subinterval.

        This method creates a new function that is the restriction of this function
        to the specified subinterval. The output is formed using a fixed length
        construction with the same number of degrees of freedom as the original function.

        Args:
            subinterval (array-like): The subinterval to which this function should be restricted.
                Must be contained within the original interval of definition.

        Returns:
            Classicfun: A new function representing the restriction of this function to the subinterval.

        Raises:
            NotSubinterval: If the subinterval is not contained within the original interval.
        """
        if subinterval not in self.interval:  # pragma: no cover
            raise NotSubinterval(self.interval, subinterval)
        if self.interval == subinterval:
            return self
        else:
            return self.__class__.initfun_fixedlen(self, subinterval, self.size)

    def translate(self, c):
        """Translate this function by a constant c.

        This method creates a new function g(x) = f(x-c), which is the original
        function translated horizontally by c.

        Args:
            c (float): The amount by which to translate the function.

        Returns:
            Classicfun: A new function representing g(x) = f(x-c).
        """
        return self.__class__(self.onefun, self.interval + c)

    # -------------
    #  rootfinding
    # -------------
    def roots(self):
        """Find the roots (zeros) of the function on its interval of definition.

        This method computes the points where the function equals zero
        within its interval of definition by finding the roots of the
        underlying onefun and mapping them to the function's interval.

        Returns:
            numpy.ndarray: An array of the roots of the function in its interval of definition,
                sorted in ascending order.
        """
        uroots = self.onefun.roots()
        return self.interval(uroots)

    # ----------
    #  calculus
    # ----------
    def cumsum(self):
        """Compute the indefinite integral of the function.

        This method calculates the indefinite integral (antiderivative) of the function,
        with the constant of integration chosen so that the indefinite integral
        evaluates to 0 at the left endpoint of the interval.

        Returns:
            Classicfun: A new function representing the indefinite integral of this function.
        """
        a, b = self.support
        onefun = 0.5 * (b - a) * self.onefun.cumsum()
        return self.__class__(onefun, self.interval)

    def diff(self):
        """Compute the derivative of the function.

        This method calculates the derivative of the function with respect to x,
        applying the chain rule to account for the mapping between the standard
        domain [-1, 1] and the function's interval.

        Returns:
            Classicfun: A new function representing the derivative of this function.
        """
        # For Trigtech, no scaling needed as it's defined directly on the interval
        # For Chebtech, apply chain rule scaling: d/dx = (d/dy)(dy/dx) = (d/dy) * 2/(b-a)
        if isinstance(self.onefun, Trigtech):
            onefun = self.onefun.diff()
        else:
            a, b = self.support
            onefun = 2.0 / (b - a) * self.onefun.diff()
        return self.__class__(onefun, self.interval)

    def sum(self):
        """Compute the definite integral of the function over its interval of definition.

        This method calculates the definite integral of the function
        over its interval of definition [a, b], applying the appropriate
        scaling factor to account for the mapping from [-1, 1].

        Returns:
            float or complex: The definite integral of the function over its interval of definition.
        """
        a, b = self.support
        return 0.5 * (b - a) * self.onefun.sum()

    # ----------
    #  plotting
    # ----------
    def plot(self, ax=None, **kwds):
        """Plot the function over its interval of definition.

        This method plots the function over its interval of definition using matplotlib.
        For complex-valued functions, it plots the real part against the imaginary part.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None,
                a new axes will be created. Defaults to None.
            **kwds: Additional keyword arguments to pass to matplotlib's plot function.

        Returns:
            matplotlib.axes.Axes: The axes on which the plot was created.
        """
        return plotfun(self, self.support, ax=ax, **kwds)


# ----------------------------------------------------------------
#  methods that execute the corresponding onefun method as is
# ----------------------------------------------------------------

methods_onefun_other = ("values", "plotcoeffs")


def add_utility(methodname):
    """Add a utility method to the Classicfun class.

    This function creates a method that delegates to the corresponding method
    of the underlying onefun object and adds it to the Classicfun class.

    Args:
        methodname (str): The name of the method to add.

    Note:
        The created method will have the same name and signature as the
        corresponding method in the onefun object.
    """

    def method(self, *args, **kwds):
        """Delegate to the corresponding method of the underlying onefun object.

        This method calls the same-named method on the underlying onefun object
        and returns its result.

        Args:
            self (Classicfun): The Classicfun object.
            *args: Variable length argument list to pass to the onefun method.
            **kwds: Arbitrary keyword arguments to pass to the onefun method.

        Returns:
            The return value from the corresponding onefun method.
        """
        return getattr(self.onefun, methodname)(*args, **kwds)

    method.__name__ = methodname
    method.__doc__ = method.__doc__
    setattr(Classicfun, methodname, method)


for methodname in methods_onefun_other:
    if methodname[:4] == "plot" and plt is None:  # pragma: no cover
        continue
    add_utility(methodname)


# -----------------------------------------------------------------------
#  unary operators and zero-argument utlity methods returning a onefun
# -----------------------------------------------------------------------

methods_onefun_zeroargs = ("__pos__", "__neg__", "copy", "simplify")


def add_zero_arg_op(methodname):
    """Add a zero-argument operation method to the Classicfun class.

    This function creates a method that delegates to the corresponding method
    of the underlying onefun object and wraps the result in a new Classicfun
    instance with the same interval.

    Args:
        methodname (str): The name of the method to add.

    Note:
        The created method will have the same name and signature as the
        corresponding method in the onefun object, but will return a Classicfun
        instance instead of an onefun instance.
    """

    def method(self, *args, **kwds):
        """Apply a zero-argument operation and return a new Classicfun.

        This method calls the same-named method on the underlying onefun object
        and wraps the result in a new Classicfun instance with the same interval.

        Args:
            self (Classicfun): The Classicfun object.
            *args: Variable length argument list to pass to the onefun method.
            **kwds: Arbitrary keyword arguments to pass to the onefun method.

        Returns:
            Classicfun: A new Classicfun instance with the result of the operation.
        """
        onefun = getattr(self.onefun, methodname)(*args, **kwds)
        return self.__class__(onefun, self.interval)

    method.__name__ = methodname
    method.__doc__ = method.__doc__
    setattr(Classicfun, methodname, method)


for methodname in methods_onefun_zeroargs:
    add_zero_arg_op(methodname)

# -----------------------------------------
# binary operators returning a onefun
# -----------------------------------------

# ToDo: change these to operator module methods
methods_onefun_binary = (
    "__add__",
    "__div__",
    "__mul__",
    "__pow__",
    "__radd__",
    "__rdiv__",
    "__rmul__",
    "__rpow__",
    "__rsub__",
    "__rtruediv__",
    "__sub__",
    "__truediv__",
)


def add_binary_op(methodname):
    """Add a binary operation method to the Classicfun class.

    This function creates a method that implements a binary operation between
    two Classicfun objects or between a Classicfun and a scalar. It delegates
    to the corresponding method of the underlying onefun object and wraps the
    result in a new Classicfun instance with the same interval.

    Args:
        methodname (str): The name of the binary operation method to add.

    Note:
        The created method will check that both Classicfun objects have the
        same interval before performing the operation. If one operand is not
        a Classicfun, it will be passed directly to the onefun method.
    """

    @self_empty()
    def method(self, f, *args, **kwds):
        """Apply a binary operation and return a new Classicfun.

        This method implements a binary operation between this Classicfun and
        another object (either another Classicfun or a scalar). It delegates
        to the corresponding method of the underlying onefun object and wraps
        the result in a new Classicfun instance with the same interval.

        Args:
            self (Classicfun): The Classicfun object.
            f (Classicfun or scalar): The second operand of the binary operation.
            *args: Variable length argument list to pass to the onefun method.
            **kwds: Arbitrary keyword arguments to pass to the onefun method.

        Returns:
            Classicfun: A new Classicfun instance with the result of the operation.

        Raises:
            IntervalMismatch: If f is a Classicfun with a different interval.
        """
        cls = self.__class__
        if isinstance(f, cls):
            # TODO: as in ChebTech, is a decorator apporach here better?
            if f.isempty:
                return f.copy()
            g = f.onefun
            # raise Exception if intervals are not consistent
            if self.interval != f.interval:  # pragma: no cover
                raise IntervalMismatch(self.interval, f.interval)
        else:
            # let the lower level classes raise any other exceptions
            g = f
        onefun = getattr(self.onefun, methodname)(g, *args, **kwds)
        return cls(onefun, self.interval)

    method.__name__ = methodname
    method.__doc__ = method.__doc__
    setattr(Classicfun, methodname, method)


for methodname in methods_onefun_binary:
    add_binary_op(methodname)

# ---------------------------
#  numpy universal functions
# ---------------------------


def add_ufunc(op):
    """Add a NumPy universal function method to the Classicfun class.

    This function creates a method that applies a NumPy universal function (ufunc)
    to the values of a Classicfun and returns a new Classicfun representing the result.

    Args:
        op (callable): The NumPy universal function to apply.

    Note:
        The created method will have the same name as the NumPy function
        and will take no arguments other than self.
    """

    @self_empty()
    def method(self):
        """Apply a NumPy universal function to this function.

        This method applies a NumPy universal function (ufunc) to the values
        of this function and returns a new function representing the result.

        Returns:
            Classicfun: A new function representing op(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: op(self(x)), self.interval)

    name = op.__name__
    method.__name__ = name
    method.__doc__ = method.__doc__
    setattr(Classicfun, name, method)


ufuncs = (
    np.absolute,
    np.arccos,
    np.arccosh,
    np.arcsin,
    np.arcsinh,
    np.arctan,
    np.arctanh,
    np.cos,
    np.cosh,
    np.exp,
    np.exp2,
    np.expm1,
    np.log,
    np.log2,
    np.log10,
    np.log1p,
    np.sinh,
    np.sin,
    np.tan,
    np.tanh,
    np.sqrt,
)

for op in ufuncs:
    add_ufunc(op)
