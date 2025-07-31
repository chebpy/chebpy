"""Implementation of functions on bounded intervals.

This module provides the Bndfun class, which implements BaseFun to represent
functions on bounded intervals [a,b]. It is a concrete implementation of the
abstract BaseFun class, specifically designed for bounded domains.
"""

import numpy as np

from .basefun import BaseFun
from .decorators import self_empty
from .exceptions import IntervalMismatch, NotSubinterval
from .plotting import plotfun, plotfuncoeffs
from .settings import _preferences as prefs
from .utilities import Interval


def get_tech_class(tech_name):
    """Dynamically import and return the specified tech class.

    Args:
        tech_name (str): The name of the tech class to import.

    Returns:
        class: The imported tech class.
    """
    if tech_name == "Chebtech":
        from .chebtech import Chebtech
        return Chebtech
    else:
        raise ValueError(f"Unknown tech: {tech_name}")


techdict = {
    "Chebtech": "Chebtech",
}


class Bndfun(BaseFun):
    """Class to approximate functions on bounded intervals [a,b]."""

    @classmethod
    def initfun(cls, f, interval=None, n=None):
        """Initialize from a callable function.

        This is a general constructor that delegates to either initfun_adaptive
        or initfun_fixedlen based on the provided parameters.

        Args:
            f (callable): The function to be approximated.
            interval: The interval on which to define the function.
                If None, the standard interval [-1, 1] is used.
            n (int, optional): If provided, specifies the number of points to use.
                If None, determines the number adaptively.

        Returns:
            BaseFun: A new instance representing the function f.
        """
        if n is None:
            return cls.initfun_adaptive(f, interval)
        else:
            return cls.initfun_fixedlen(f, interval, n)

    @classmethod
    def initvalues(cls, values, interval=None):
        """Initialize from function values at Chebyshev points.

        This constructor creates a function representation from values
        at Chebyshev points.

        Args:
            values: Function values at Chebyshev points.
            interval: The interval on which to define the function.
                If None, the standard interval [-1, 1] is used.

        Returns:
            BaseFun: A new instance representing the function with the given values.
        """
        tech_class = get_tech_class(techdict[prefs.tech])
        onefun = tech_class.initvalues(values, interval=interval)
        return cls(onefun, interval or Interval())

    def prolong(self, n):
        """Extend the function representation to a larger size.

        This method extends the function representation to use more coefficients
        or a higher degree, which can be useful for certain operations.

        Args:
            n (int): The new size for the function representation.

        Returns:
            BaseFun: A new function with an extended representation.
        """
        return self.__class__(self.onefun.prolong(n), self.interval)

    # --------------------------
    #  alternative constructors
    # --------------------------
    @classmethod
    def initempty(cls):
        """Initialize an empty function.

        This constructor creates an empty function representation, which is
        useful as a placeholder or for special cases. The interval has no
        relevance to the emptiness status of a Bndfun, so we arbitrarily
        set it to be the default interval [-1, 1].

        Returns:
            Bndfun: A new empty instance.
        """
        interval = Interval()
        tech_class = get_tech_class(techdict[prefs.tech])
        onefun = tech_class.initempty(interval=interval)
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
            Bndfun: A new instance representing the constant function f(x) = c.
        """
        tech_class = get_tech_class(techdict[prefs.tech])
        onefun = tech_class.initconst(c, interval=interval)
        return cls(onefun, interval)

    @classmethod
    def initidentity(cls, interval):
        """Initialize the identity function f(x) = x.

        This constructor creates a function that represents f(x) = x
        on the specified interval.

        Args:
            interval: The interval on which to define the identity function.

        Returns:
            Bndfun: A new instance representing the identity function.
        """
        tech_class = get_tech_class(techdict[prefs.tech])
        onefun = tech_class.initvalues(np.asarray(interval), interval=interval)
        return cls(onefun, interval)

    @classmethod
    def initfun_adaptive(cls, f, interval):
        """Initialize from a callable function using adaptive sampling.

        This constructor determines the appropriate number of points needed to
        represent the function to the specified tolerance using an adaptive algorithm.

        Args:
            f (callable): The function to be approximated.
            interval: The interval on which to define the function.

        Returns:
            Bndfun: A new instance representing the function f.
        """
        tech_class = get_tech_class(techdict[prefs.tech])
        onefun = tech_class.initfun(lambda y: f(interval(y)), interval=interval)
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
            Bndfun: A new instance representing the function f.
        """
        tech_class = get_tech_class(techdict[prefs.tech])
        onefun = tech_class.initfun(lambda y: f(interval(y)), n, interval=interval)
        return cls(onefun, interval)

    # -------------------
    #  'private' methods
    # -------------------
    def __call__(self, x, how="clenshaw"):
        """Evaluate the function at points x.

        This method evaluates the function at the specified points by mapping them
        to the standard domain [-1, 1] and evaluating the underlying onefun.

        Args:
            x (float or array-like): Points at which to evaluate the function.
            how (str, optional): Method to use for evaluation. Defaults to "clenshaw".

        Returns:
            float or array-like: The value(s) of the function at the specified point(s).
                Returns a scalar if x is a scalar, otherwise an array of the same size as x.
        """
        y = self.interval.invmap(x)
        return self.onefun(y, how)

    def __init__(self, onefun, interval):
        """Initialize a new Bndfun instance.

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
            Bndfun: A new function representing the imaginary part of this function.
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
            Bndfun: A new function representing the real part of this function.
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
            Bndfun: A new function representing the restriction of this function to the subinterval.

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
            Bndfun: A new function representing g(x) = f(x-c).
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
            Bndfun: A new function representing the indefinite integral of this function.
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
            Bndfun: A new function representing the derivative of this function.
        """
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

    def plotcoeffs(self, ax=None, **kwds):
        """Plot the absolute values of the function's coefficients on a semilogy scale.

        This method creates a semilogy plot of the absolute values of the function's
        coefficients, which is useful for visualizing the decay of coefficients
        in a Chebyshev series.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None,
                a new axes will be created. Defaults to None.
            **kwds: Additional keyword arguments to pass to matplotlib's semilogy function.

        Returns:
            matplotlib.axes.Axes: The axes on which the plot was created.
        """
        return plotfuncoeffs(np.abs(self.coeffs), ax=ax, **kwds)

    # -----------
    #  Additional methods required by BaseFun
    # -----------
    def copy(self):
        """Create a deep copy of this function.

        This method creates a new function that is a deep copy of this function,
        ensuring that modifications to the copy do not affect the original.

        Returns:
            Bndfun: A new function that is a deep copy of this function.
        """
        return self.__class__(self.onefun.copy(), self.interval)

    def simplify(self):
        """Simplify the function representation.

        This method simplifies the function representation by removing unnecessary
        coefficients or reducing the degree, while maintaining the specified accuracy.

        Returns:
            Bndfun: A new function with a simplified representation.
        """
        return self.__class__(self.onefun.simplify(), self.interval)

    def values(self):
        """Get the values of the function at the points used for its representation.

        This method returns the values of the function at the points used for its
        representation, such as Chebyshev points.

        Returns:
            array-like: The values of the function at the representation points.
        """
        return self.onefun.values()

    # -----------
    #  Binary operators
    # -----------
    @self_empty()
    def __add__(self, f):
        """Add this function with another function or a scalar.

        This method implements the addition operation between this function
        and another function or a scalar.

        Args:
            f (Bndfun or scalar): The function or scalar to add to this function.

        Returns:
            Bndfun: A new function representing the sum.

        Raises:
            IntervalMismatch: If f is a Bndfun with a different interval.
        """
        cls = self.__class__
        if isinstance(f, cls):
            if f.isempty:
                return f.copy()
            g = f.onefun
            # raise Exception if intervals are not consistent
            if self.interval != f.interval:  # pragma: no cover
                raise IntervalMismatch(self.interval, f.interval)
        else:
            # let the lower level classes raise any other exceptions
            g = f
        onefun = self.onefun.__add__(g)
        return cls(onefun, self.interval)

    @self_empty()
    def __mul__(self, f):
        """Multiply this function with another function or a scalar.

        This method implements the multiplication operation between this function
        and another function or a scalar.

        Args:
            f (Bndfun or scalar): The function or scalar to multiply with this function.

        Returns:
            Bndfun: A new function representing the product.

        Raises:
            IntervalMismatch: If f is a Bndfun with a different interval.
        """
        cls = self.__class__
        if isinstance(f, cls):
            if f.isempty:
                return f.copy()
            g = f.onefun
            # raise Exception if intervals are not consistent
            if self.interval != f.interval:  # pragma: no cover
                raise IntervalMismatch(self.interval, f.interval)
        else:
            # let the lower level classes raise any other exceptions
            g = f
        onefun = self.onefun.__mul__(g)
        return cls(onefun, self.interval)

    @self_empty()
    def __pow__(self, f):
        """Raise this function to a power.

        This method implements the power operation for this function.

        Args:
            f (int, float, or Bndfun): The exponent to which this function is raised.

        Returns:
            Bndfun: A new function representing f(x)^power.

        Raises:
            IntervalMismatch: If f is a Bndfun with a different interval.
        """
        cls = self.__class__
        if isinstance(f, cls):
            if f.isempty:
                return f.copy()
            g = f.onefun
            # raise Exception if intervals are not consistent
            if self.interval != f.interval:  # pragma: no cover
                raise IntervalMismatch(self.interval, f.interval)
        else:
            # let the lower level classes raise any other exceptions
            g = f
        onefun = self.onefun.__pow__(g)
        return cls(onefun, self.interval)

    @self_empty()
    def __rpow__(self, f):
        """Raise a scalar or another function to the power of this function.

        This method is called when a scalar or another function is raised to the power of this function,
        i.e., other ** self.

        Args:
            f (scalar or Bndfun): The base to be raised to the power of this function.

        Returns:
            Bndfun: A new function representing f^(self(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: f ** self(x), self.interval)

    @self_empty()
    def __radd__(self, f):
        """Add a scalar or another function to this function (from the right).

        This method is called when a scalar or another function is added to this function,
        i.e., other + self.

        Args:
            f (scalar or Bndfun): The scalar or function to add to this function.

        Returns:
            Bndfun: A new function representing the sum.
        """
        return self.__add__(f)

    @self_empty()
    def __rmul__(self, f):
        """Multiply a scalar or another function with this function (from the right).

        This method is called when a scalar or another function is multiplied with this function,
        i.e., other * self.

        Args:
            f (scalar or Bndfun): The scalar or function to multiply with this function.

        Returns:
            Bndfun: A new function representing the product.
        """
        return self.__mul__(f)

    @self_empty()
    def __rsub__(self, f):
        """Subtract this function from a scalar or another function.

        This method is called when this function is subtracted from a scalar or another function,
        i.e., other - self.

        Args:
            f (scalar or Bndfun): The scalar or function from which to subtract this function.

        Returns:
            Bndfun: A new function representing the difference.
        """
        return -self + f

    @self_empty()
    def __sub__(self, f):
        """Subtract another function or a scalar from this function.

        This method implements the subtraction operation between this function
        and another function or a scalar.

        Args:
            f (Bndfun or scalar): The function or scalar to subtract from this function.

        Returns:
            Bndfun: A new function representing the difference.
        """
        return self + (-f)

    @self_empty()
    def __truediv__(self, f):
        """Divide this function by another function or a scalar.

        This method implements the division operation between this function
        and another function or a scalar.

        Args:
            f (Bndfun or scalar): The function or scalar by which to divide this function.

        Returns:
            Bndfun: A new function representing the quotient.

        Raises:
            IntervalMismatch: If f is a Bndfun with a different interval.
        """
        cls = self.__class__
        if isinstance(f, cls):
            if f.isempty:
                return f.copy()
            g = f.onefun
            # raise Exception if intervals are not consistent
            if self.interval != f.interval:  # pragma: no cover
                raise IntervalMismatch(self.interval, f.interval)
        else:
            # let the lower level classes raise any other exceptions
            g = f
        onefun = self.onefun.__truediv__(g)
        return cls(onefun, self.interval)

    @self_empty()
    def __rtruediv__(self, f):
        """Divide a scalar or another function by this function.

        This method is called when this function is the divisor in a division operation,
        i.e., other / self.

        Args:
            f (scalar or Bndfun): The scalar or function to be divided by this function.

        Returns:
            Bndfun: A new function representing the quotient.
        """
        cls = self.__class__
        if isinstance(f, cls):
            if f.isempty:
                return f.copy()
            g = f.onefun
            # raise Exception if intervals are not consistent
            if self.interval != f.interval:  # pragma: no cover
                raise IntervalMismatch(self.interval, f.interval)
        else:
            # let the lower level classes raise any other exceptions
            g = f
        onefun = self.onefun.__rtruediv__(g)
        return cls(onefun, self.interval)

    def __neg__(self):
        """Return the negative of this function.

        This method implements the unary negation operation for this function.

        Returns:
            Bndfun: A new function representing -f(x).
        """
        return self.__class__(self.onefun.__neg__(), self.interval)

    def __pos__(self):
        """Return the positive of this function (which is the function itself).

        This method implements the unary plus operation for this function.

        Returns:
            Bndfun: This function object (unchanged).
        """
        return self.__class__(self.onefun.__pos__(), self.interval)

    # ---------------------------
    #  numpy universal functions
    # ---------------------------

    def absolute(self):
        """Apply the NumPy absolute function to this function.

        This method applies the NumPy absolute function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing abs(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.absolute(self(x)), self.interval)

    def arccos(self):
        """Apply the NumPy arccos function to this function.

        This method applies the NumPy arccos function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing arccos(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.arccos(self(x)), self.interval)

    def arccosh(self):
        """Apply the NumPy arccosh function to this function.

        This method applies the NumPy arccosh function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing arccosh(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.arccosh(self(x)), self.interval)

    def arcsin(self):
        """Apply the NumPy arcsin function to this function.

        This method applies the NumPy arcsin function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing arcsin(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.arcsin(self(x)), self.interval)

    def arcsinh(self):
        """Apply the NumPy arcsinh function to this function.

        This method applies the NumPy arcsinh function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing arcsinh(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.arcsinh(self(x)), self.interval)

    def arctan(self):
        """Apply the NumPy arctan function to this function.

        This method applies the NumPy arctan function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing arctan(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.arctan(self(x)), self.interval)

    def arctanh(self):
        """Apply the NumPy arctanh function to this function.

        This method applies the NumPy arctanh function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing arctanh(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.arctanh(self(x)), self.interval)

    def cos(self):
        """Apply the NumPy cos function to this function.

        This method applies the NumPy cos function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing cos(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.cos(self(x)), self.interval)

    def cosh(self):
        """Apply the NumPy cosh function to this function.

        This method applies the NumPy cosh function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing cosh(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.cosh(self(x)), self.interval)

    def exp(self):
        """Apply the NumPy exp function to this function.

        This method applies the NumPy exp function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing exp(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.exp(self(x)), self.interval)

    def exp2(self):
        """Apply the NumPy exp2 function to this function.

        This method applies the NumPy exp2 function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing 2^(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.exp2(self(x)), self.interval)

    def expm1(self):
        """Apply the NumPy expm1 function to this function.

        This method applies the NumPy expm1 function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing exp(f(x))-1.
        """
        return self.__class__.initfun_adaptive(lambda x: np.expm1(self(x)), self.interval)

    def log(self):
        """Apply the NumPy log function to this function.

        This method applies the NumPy log function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing log(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.log(self(x)), self.interval)

    def log2(self):
        """Apply the NumPy log2 function to this function.

        This method applies the NumPy log2 function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing log2(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.log2(self(x)), self.interval)

    def log10(self):
        """Apply the NumPy log10 function to this function.

        This method applies the NumPy log10 function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing log10(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.log10(self(x)), self.interval)

    def log1p(self):
        """Apply the NumPy log1p function to this function.

        This method applies the NumPy log1p function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing log(1+f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.log1p(self(x)), self.interval)

    def sinh(self):
        """Apply the NumPy sinh function to this function.

        This method applies the NumPy sinh function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing sinh(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.sinh(self(x)), self.interval)

    def sin(self):
        """Apply the NumPy sin function to this function.

        This method applies the NumPy sin function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing sin(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.sin(self(x)), self.interval)

    def tan(self):
        """Apply the NumPy tan function to this function.

        This method applies the NumPy tan function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing tan(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.tan(self(x)), self.interval)

    def tanh(self):
        """Apply the NumPy tanh function to this function.

        This method applies the NumPy tanh function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing tanh(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.tanh(self(x)), self.interval)

    def sqrt(self):
        """Apply the NumPy sqrt function to this function.

        This method applies the NumPy sqrt function to the values
        of this function and returns a new function representing the result.

        Returns:
            Bndfun: A new function representing sqrt(f(x)).
        """
        return self.__class__.initfun_adaptive(lambda x: np.sqrt(self(x)), self.interval)
