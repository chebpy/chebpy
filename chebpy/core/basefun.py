"""Abstract base class for functions in ChebPy.

This module provides the BaseFun abstract base class, which defines the interface for
all function representations in ChebPy. It specifies the methods and properties that
concrete function classes must implement.

The BaseFun class serves as the foundation for the function class hierarchy in ChebPy,
with concrete implementations inheriting from it. It defines a comprehensive interface
for working with function representations, including algebraic operations, calculus
operations, and utility functions.

This class replaces the previous separate Fun and Onefun classes, merging their
functionality into a single base class.
"""

from abc import ABC, abstractmethod


class BaseFun(ABC):
    """Abstract base class for functions in ChebPy.

    This class defines the interface for all function representations in ChebPy.
    It serves as the base class for specific implementations like Chebtech, Bndfun,
    and Classicfun.

    Concrete subclasses must implement all the abstract methods defined here,
    which include constructors, algebraic operations, calculus operations,
    and utility functions.
    """

    # --------------------------
    #  alternative constructors
    # --------------------------
    @classmethod
    @abstractmethod
    def initconst(cls, c, interval=None):  # pragma: no cover
        """Initialize a constant function.

        This constructor creates a function that represents a constant value
        on the specified interval.

        Args:
            c: The constant value.
            interval: The interval on which to define the function.
                If None, the standard interval [-1, 1] is used.

        Returns:
            BaseFun: A new instance representing the constant function f(x) = c.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initempty(cls):  # pragma: no cover
        """Initialize an empty function.

        This constructor creates an empty function representation, which is
        useful as a placeholder or for special cases.

        Returns:
            BaseFun: A new empty instance.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initidentity(cls, interval=None):  # pragma: no cover
        """Initialize the identity function f(x) = x.

        This constructor creates a function that represents f(x) = x
        on the specified interval.

        Args:
            interval: The interval on which to define the function.
                If None, the standard interval [-1, 1] is used.

        Returns:
            BaseFun: A new instance representing the identity function.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initfun(cls, f, interval=None, n=None):  # pragma: no cover
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
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initfun_adaptive(cls, f, interval=None):  # pragma: no cover
        """Initialize from a callable function using adaptive sampling.

        This constructor determines the appropriate number of points needed to
        represent the function to a specified tolerance using an adaptive algorithm.

        Args:
            f (callable): The function to be approximated.
            interval: The interval on which to define the function.
                If None, the standard interval [-1, 1] is used.

        Returns:
            BaseFun: A new instance representing the function f.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initfun_fixedlen(cls, f, n, interval=None):  # pragma: no cover
        """Initialize from a callable function using a fixed number of points.

        This constructor uses a specified number of points to represent the function,
        rather than determining the number adaptively.

        Args:
            f (callable): The function to be approximated.
            n (int): The number of points to use.
            interval: The interval on which to define the function.
                If None, the standard interval [-1, 1] is used.

        Returns:
            BaseFun: A new instance representing the function f.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def initvalues(cls, values):  # pragma: no cover
        """Initialize from function values at Chebyshev points.

        This constructor creates a function representation from values
        at Chebyshev points.

        Args:
            values: Function values at Chebyshev points.

        Returns:
            BaseFun: A new instance representing the function with the given values.
        """
        raise NotImplementedError

    # -------------------
    #  "private" methods
    # -------------------
    @abstractmethod
    def __add__(self, other):  # pragma: no cover
        """Add this function with another function or a scalar.

        This method implements the addition operation between this function
        and another function or a scalar.

        Args:
            other (BaseFun or scalar): The function or scalar to add to this function.

        Returns:
            BaseFun: A new function representing the sum.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, x):  # pragma: no cover
        """Evaluate the function at points x.

        This method evaluates the function at the specified points.

        Args:
            x (float or array-like): Points at which to evaluate the function.

        Returns:
            float or array-like: The value(s) of the function at the specified point(s).
                Returns a scalar if x is a scalar, otherwise an array of the same size as x.
        """
        raise NotImplementedError

    @abstractmethod
    def __init__(self):  # pragma: no cover
        """Initialize a new BaseFun instance.

        This method initializes a new function representation.
        The specific initialization depends on the concrete subclass implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def __mul__(self, other):  # pragma: no cover
        """Multiply this function with another function or a scalar.

        This method implements the multiplication operation between this function
        and another function or a scalar.

        Args:
            other (BaseFun or scalar): The function or scalar to multiply with this function.

        Returns:
            BaseFun: A new function representing the product.
        """
        raise NotImplementedError

    @abstractmethod
    def __neg__(self):  # pragma: no cover
        """Return the negative of this function.

        This method implements the unary negation operation for this function.

        Returns:
            BaseFun: A new function representing -f(x).
        """
        raise NotImplementedError

    @abstractmethod
    def __pos__(self):  # pragma: no cover
        """Return the positive of this function (which is the function itself).

        This method implements the unary plus operation for this function.

        Returns:
            BaseFun: This function object (unchanged).
        """
        raise NotImplementedError

    @abstractmethod
    def __pow__(self, power):  # pragma: no cover
        """Raise this function to a power.

        This method implements the power operation for this function.

        Args:
            power (int, float, or BaseFun): The exponent to which this function is raised.

        Returns:
            BaseFun: A new function representing f(x)^power.
        """
        raise NotImplementedError

    @abstractmethod
    def __radd__(self, other):  # pragma: no cover
        """Add a scalar or another function to this function (from the right).

        This method is called when a scalar or another function is added to this function,
        i.e., other + self.

        Args:
            other (scalar or BaseFun): The scalar or function to add to this function.

        Returns:
            BaseFun: A new function representing the sum.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):  # pragma: no cover
        """Return a string representation of the function.

        This method returns a string representation of the function that includes
        relevant information about its representation and interval.

        Returns:
            str: A string representation of the function.
        """
        raise NotImplementedError

    @abstractmethod
    def __rmul__(self, other):  # pragma: no cover
        """Multiply a scalar or another function with this function (from the right).

        This method is called when a scalar or another function is multiplied with this function,
        i.e., other * self.

        Args:
            other (scalar or BaseFun): The scalar or function to multiply with this function.

        Returns:
            BaseFun: A new function representing the product.
        """
        raise NotImplementedError

    @abstractmethod
    def __rsub__(self, other):  # pragma: no cover
        """Subtract this function from a scalar or another function.

        This method is called when this function is subtracted from a scalar or another function,
        i.e., other - self.

        Args:
            other (scalar or BaseFun): The scalar or function from which to subtract this function.

        Returns:
            BaseFun: A new function representing the difference.
        """
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other):  # pragma: no cover
        """Subtract another function or a scalar from this function.

        This method implements the subtraction operation between this function
        and another function or a scalar.

        Args:
            other (BaseFun or scalar): The function or scalar to subtract from this function.

        Returns:
            BaseFun: A new function representing the difference.
        """
        raise NotImplementedError

    # ------------
    #  properties
    # ------------
    @property
    @abstractmethod
    def coeffs(self):  # pragma: no cover
        """Get the coefficients of the function representation.

        This property returns the coefficients used in the function representation,
        such as Chebyshev coefficients for a Chebyshev series.

        Returns:
            array-like: The coefficients of the function representation.
        """
        raise NotImplementedError

    @property
    def interval(self):  # pragma: no cover
        """Get the interval on which this function is defined.

        This property returns the interval object representing the domain
        of definition for this function. For functions defined on the standard
        interval [-1, 1], this returns a fixed interval.

        Returns:
            Interval: The interval on which this function is defined.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def isconst(self):  # pragma: no cover
        """Check if this function represents a constant.

        This property determines whether the function is constant (i.e., f(x) = c
        for some constant c) over its interval of definition.

        Returns:
            bool: True if the function is constant, False otherwise.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def isempty(self):  # pragma: no cover
        """Check if this function is empty.

        This property determines whether the function is empty, which is a special
        state used as a placeholder or for special cases.

        Returns:
            bool: True if the function is empty, False otherwise.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def iscomplex(self):  # pragma: no cover
        """Check if this function has complex values.

        This property determines whether the function has complex values or is
        purely real-valued.

        Returns:
            bool: True if the function has complex values, False otherwise.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self):  # pragma: no cover
        """Get the size of the function representation.

        This property returns the number of coefficients or other measure of the
        complexity of the function representation.

        Returns:
            int: The size of the function representation.
        """
        raise NotImplementedError

    @property
    def support(self):  # pragma: no cover
        """Get the support interval of this function.

        This property returns the interval on which this function is defined,
        represented as a numpy array with two elements [a, b].

        Returns:
            numpy.ndarray: Array containing the endpoints of the interval.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def vscale(self):  # pragma: no cover
        """Get the vertical scale of the function.

        This property returns a measure of the range of function values, typically
        the maximum absolute value of the function on its interval of definition.

        Returns:
            float: The vertical scale of the function.
        """
        raise NotImplementedError

    # -----------
    #  utilities
    # -----------
    @abstractmethod
    def copy(self):  # pragma: no cover
        """Create a deep copy of this function.

        This method creates a new function that is a deep copy of this function,
        ensuring that modifications to the copy do not affect the original.

        Returns:
            BaseFun: A new function that is a deep copy of this function.
        """
        raise NotImplementedError

    @abstractmethod
    def imag(self):  # pragma: no cover
        """Get the imaginary part of this function.

        This method returns a new function representing the imaginary part of this function.
        If this function is real-valued, returns a zero function.

        Returns:
            BaseFun: A new function representing the imaginary part of this function.
        """
        raise NotImplementedError

    @abstractmethod
    def prolong(self, n):  # pragma: no cover
        """Extend the function representation to a larger size.

        This method extends the function representation to use more coefficients
        or a higher degree, which can be useful for certain operations.

        Args:
            n (int): The new size for the function representation.

        Returns:
            BaseFun: A new function with an extended representation.
        """
        raise NotImplementedError

    @abstractmethod
    def real(self):  # pragma: no cover
        """Get the real part of this function.

        This method returns a new function representing the real part of this function.
        If this function is already real-valued, returns this function.

        Returns:
            BaseFun: A new function representing the real part of this function.
        """
        raise NotImplementedError

    @abstractmethod
    def restrict(self, subinterval):  # pragma: no cover
        """Restrict this function to a subinterval.

        This method creates a new function that is the restriction of this function
        to the specified subinterval.

        Args:
            subinterval (array-like): The subinterval to which this function should be restricted.
                Must be contained within the original interval of definition.

        Returns:
            BaseFun: A new function representing the restriction of this function to the subinterval.
        """
        raise NotImplementedError

    @abstractmethod
    def simplify(self):  # pragma: no cover
        """Simplify the function representation.

        This method simplifies the function representation by removing unnecessary
        coefficients or reducing the degree, while maintaining the specified accuracy.

        Returns:
            BaseFun: A new function with a simplified representation.
        """
        raise NotImplementedError

    @abstractmethod
    def values(self):  # pragma: no cover
        """Get the values of the function at the points used for its representation.

        This method returns the values of the function at the points used for its
        representation, such as Chebyshev points.

        Returns:
            array-like: The values of the function at the representation points.
        """
        raise NotImplementedError

    # -------------
    #  rootfinding
    # -------------
    @abstractmethod
    def roots(self):  # pragma: no cover
        """Find the roots (zeros) of the function on its interval of definition.

        This method computes the points where the function equals zero
        within its interval of definition.

        Returns:
            array-like: An array of the roots of the function in its interval of definition,
                sorted in ascending order.
        """
        raise NotImplementedError

    # ----------
    #  calculus
    # ----------
    @abstractmethod
    def cumsum(self):  # pragma: no cover
        """Compute the indefinite integral of the function.

        This method calculates the indefinite integral (antiderivative) of the function,
        with the constant of integration chosen so that the indefinite integral
        evaluates to 0 at the left endpoint of the interval.

        Returns:
            BaseFun: A new function representing the indefinite integral of this function.
        """
        raise NotImplementedError

    @abstractmethod
    def diff(self):  # pragma: no cover
        """Compute the derivative of the function.

        This method calculates the derivative of the function with respect to x.

        Returns:
            BaseFun: A new function representing the derivative of this function.
        """
        raise NotImplementedError

    @abstractmethod
    def sum(self):  # pragma: no cover
        """Compute the definite integral of the function over its interval of definition.

        This method calculates the definite integral of the function
        over its interval of definition.

        Returns:
            float or complex: The definite integral of the function over its interval of definition.
        """
        raise NotImplementedError
