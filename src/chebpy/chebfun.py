"""Implementation of the Chebfun class for piecewise function approximation.

This module provides the Chebfun class, which is the main user-facing class in the
ChebPy package. It represents functions using piecewise polynomial approximations
on arbitrary intervals, allowing for operations such as integration, differentiation,
root-finding, and more.

The Chebfun class is inspired by the MATLAB package of the same name and provides
similar functionality for working with functions rather than numbers.
"""

import operator
import warnings

import matplotlib.pyplot as plt
import numpy as np

from .bndfun import Bndfun
from .decorators import cache, cast_arg_to_chebfun, float_argument, self_empty
from .exceptions import BadFunLengthArgument, InvalidDomain, SupportMismatch
from .plotting import plotfun
from .settings import _preferences as prefs
from .trigtech import Trigtech
from .utilities import Domain, Interval, check_funs, compute_breakdata, generate_funs


class Chebfun:
    """Main class for representing and manipulating functions in ChebPy.

    The Chebfun class represents functions using piecewise polynomial approximations
    on arbitrary intervals. It provides a comprehensive set of operations for working
    with these function representations, including:

    - Function evaluation at arbitrary points
    - Algebraic operations (addition, multiplication, etc.)
    - Calculus operations (differentiation, integration, etc.)
    - Rootfinding
    - Plotting

    Chebfun objects can be created from callable functions, constant values, or
    directly from function pieces. The class supports both adaptive and fixed-length
    approximations, allowing for efficient representation of functions with varying
    complexity across different intervals.

    Attributes:
        funs (numpy.ndarray): Array of function pieces that make up the Chebfun.
        breakdata (OrderedDict): Mapping of breakpoints to function values.
        transposed (bool): Flag indicating if the Chebfun is transposed.
    """

    def __init__(self, funs):
        """Initialize a Chebfun object.

        Args:
            funs (list): List of function objects to be included in the Chebfun.
                These will be checked and sorted using check_funs.
        """
        self.funs = check_funs(funs)
        self.breakdata = compute_breakdata(self.funs)
        self.transposed = False

    @classmethod
    def initempty(cls):
        """Initialize an empty Chebfun.

        Returns:
            Chebfun: An empty Chebfun object with no functions.
        """
        return cls([])

    @classmethod
    def initidentity(cls, domain=None):
        """Initialize a Chebfun representing the identity function f(x) = x.

        Args:
            domain (array-like, optional): Domain on which to define the identity function.
                If None, uses the default domain from preferences.

        Returns:
            Chebfun: A Chebfun object representing the identity function on the specified domain.
        """
        return cls(generate_funs(domain, Bndfun.initidentity))

    @classmethod
    def initconst(cls, c, domain=None):
        """Initialize a Chebfun representing a constant function f(x) = c.

        Args:
            c (float or complex): The constant value.
            domain (array-like, optional): Domain on which to define the constant function.
                If None, uses the default domain from preferences.

        Returns:
            Chebfun: A Chebfun object representing the constant function on the specified domain.
        """
        return cls(generate_funs(domain, Bndfun.initconst, {"c": c}))

    @classmethod
    def initfun_adaptive(cls, f, domain=None, splitting=None):
        """Initialize a Chebfun by adaptively sampling a function.

        This method determines the appropriate number of points needed to represent
        the function to the specified tolerance using an adaptive algorithm.

        Args:
            f (callable): The function to be approximated.
            domain (array-like, optional): Domain on which to define the function.
                If None, uses the default domain from preferences.
            splitting (bool, optional): Whether to use automatic domain splitting
                for functions with discontinuities. If None, uses prefs.splitting.

        Returns:
            Chebfun: A Chebfun object representing the function on the specified domain.
        """
        # Determine splitting setting
        use_splitting = prefs.splitting if splitting is None else splitting

        # Handle empty and invalid domains early (before Domain validation)
        domain_array = np.asarray(domain if domain is not None else prefs.domain, dtype=float)
        if domain_array.size == 0:
            # Empty domain -> return empty Chebfun
            return cls([])
        if domain_array.size == 1:
            # Single-point domain is invalid
            raise InvalidDomain("Domain must have at least two points")
        if domain_array.size >= 2 and np.allclose(domain_array[0], domain_array[-1]):
            raise InvalidDomain("Domain endpoints cannot be equal")

        # Now create proper Domain object
        domain = Domain(domain_array)

        if use_splitting:
            return cls._initfun_splitting(f, domain)
        else:
            return cls(generate_funs(domain, Bndfun.initfun_adaptive, {"f": f}))

    @classmethod
    def initfun_fixedlen(cls, f, n, domain=None):
        """Initialize a Chebfun with a fixed number of points.

        This method uses a specified number of points to represent the function,
        rather than determining the number adaptively.

        Args:
            f (callable): The function to be approximated.
            n (int or array-like): Number of points to use. If a single value, uses the same
                number for each interval. If an array, must have one fewer elements than
                the size of the domain.
            domain (array-like, optional): Domain on which to define the function.
                If None, uses the default domain from preferences.

        Returns:
            Chebfun: A Chebfun object representing the function on the specified domain.

        Raises:
            BadFunLengthArgument: If n is an array and its size doesn't match domain.size - 1.
        """
        nn = np.array(n)
        if nn.size < 2:
            funs = generate_funs(domain, Bndfun.initfun_fixedlen, {"f": f, "n": n})
        else:
            domain = Domain(domain if domain is not None else prefs.domain)
            if not nn.size == domain.size - 1:
                raise BadFunLengthArgument
            funs = []
            for interval, length in zip(domain.intervals, nn):
                funs.append(Bndfun.initfun_fixedlen(f, interval, length))
        return cls(funs)

    @classmethod
    def initfun(cls, f, domain=None, n=None, splitting=None):
        """Initialize a Chebfun from a function.

        This is a general-purpose constructor that delegates to either initfun_adaptive
        or initfun_fixedlen based on whether n is provided.

        Args:
            f (callable): The function to be approximated.
            domain (array-like, optional): Domain on which to define the function.
                If None, uses the default domain from preferences.
            n (int or array-like, optional): Number of points to use. If None, determines
                the number adaptively. If provided, uses a fixed number of points.
            splitting (bool, optional): Whether to use automatic domain splitting
                for functions with discontinuities. Only used when n is None.

        Returns:
            Chebfun: A Chebfun object representing the function on the specified domain.
        """
        if n is None:
            return cls.initfun_adaptive(f, domain, splitting=splitting)
        else:
            return cls.initfun_fixedlen(f, n, domain)

    # --------------------
    #  operator overloads
    # --------------------
    def __add__(self, f):
        """Add a Chebfun with another Chebfun or a scalar.

        Args:
            f (Chebfun or scalar): The object to add to this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing the sum.
        """
        # Defer to OrderTracerAST if present (for order detection)
        if getattr(f, "_is_order_tracer", False):
            return NotImplemented
        return self._apply_binop(f, operator.add)

    @self_empty(np.array([]))
    @float_argument
    def __call__(self, x):
        """Evaluate the Chebfun at points x.

        This method evaluates the Chebfun at the specified points. It handles interior
        points, breakpoints, and points outside the domain appropriately.

        Args:
            x (float or array-like): Points at which to evaluate the Chebfun.

        Returns:
            float or numpy.ndarray: The value(s) of the Chebfun at the specified point(s).
                Returns a scalar if x is a scalar, otherwise an array of the same size as x.
        """
        # Check if any fun uses Trigtech - if so, always use complex dtype
        # since Trigtech operations may produce complex intermediate results

        uses_trigtech = any(isinstance(fun.onefun, Trigtech) for fun in self if hasattr(fun, "onefun"))

        # initialise output - use complex for Trigtech or complex functions
        dtype = complex if (self.iscomplex or uses_trigtech) else float
        out = np.full(x.size, np.nan, dtype=dtype)

        # evaluate a fun when x is an interior point
        for fun in self:
            idx = fun.interval.isinterior(x)
            out[idx] = fun(x[idx])

        # evaluate the breakpoint data for x at a breakpoint
        breakpoints = self.breakpoints
        for break_point in breakpoints:
            out[x == break_point] = self.breakdata[break_point]

        # first and last funs used to evaluate outside of the chebfun domain
        lpts, rpts = x < breakpoints[0], x > breakpoints[-1]
        out[lpts] = self.funs[0](x[lpts])
        out[rpts] = self.funs[-1](x[rpts])

        # For Trigtech-based results, convert to real if imaginary part is negligible
        # This handles the case where Fourier coefficients are complex but the
        # evaluated function values are essentially real
        if uses_trigtech and out.size > 0:
            max_imag = np.max(np.abs(out.imag))
            max_real = np.max(np.abs(out.real))
            if max_imag < 1e-13 * (max_real + 1e-14):
                out = out.real

        return out

    def __iter__(self):
        """Return an iterator over the functions in this Chebfun.

        Returns:
            iterator: An iterator over the functions (funs) in this Chebfun.
        """
        return self.funs.__iter__()

    def __len__(self):
        """Return the total number of coefficients across all funs.

        This is analogous to MATLAB Chebfun's length() function, which returns
        the total number of degrees of freedom in the representation.

        Returns:
            int: The sum of sizes of all constituent funs.
        """
        return sum(f.size for f in self.funs)

    def __eq__(self, other):
        """Test for equality between two Chebfun objects.

        Two Chebfun objects are considered equal if they have the same domain
        and their function values are equal (within tolerance) at a set of test points.

        Args:
            other (object): The object to compare with this Chebfun.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if not isinstance(other, self.__class__):
            return False

        # Check if both are empty
        if self.isempty and other.isempty:
            return True

        # Check if domains are equal
        if self.domain != other.domain:
            return False

        # Check function values at test points
        xx = np.linspace(self.support[0], self.support[1], 100)
        tol = 1e2 * max(self.vscale, other.vscale) * prefs.eps
        return np.all(np.abs(self(xx) - other(xx)) <= tol)

    def __mul__(self, f):
        """Multiply a Chebfun with another Chebfun or a scalar.

        Args:
            f (Chebfun or scalar): The object to multiply with this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing the product.
        """
        # Defer to OrderTracerAST if present (for order detection)
        if getattr(f, "_is_order_tracer", False):
            return NotImplemented
        return self._apply_binop(f, operator.mul)

    def __neg__(self):
        """Return the negative of this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing -f(x).
        """
        return self.__class__(-self.funs)

    def __pos__(self):
        """Return the positive of this Chebfun (which is the Chebfun itself).

        Returns:
            Chebfun: This Chebfun object (unchanged).
        """
        return self

    def __abs__(self):
        """Return the absolute value of this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing |f(x)|.
        """
        # Apply abs to each fun piece using the absolute() method
        abs_funs = []
        for fun in self.funs:
            abs_funs.append(fun.absolute())
        return self.__class__(abs_funs)

    def __pow__(self, f):
        """Raise this Chebfun to a power.

        Args:
            f (Chebfun or scalar): The exponent to which this Chebfun is raised.

        Returns:
            Chebfun: A new Chebfun representing self^f.
        """
        # Defer to OrderTracerAST if present (for order detection)
        if getattr(f, "_is_order_tracer", False):
            return NotImplemented
        return self._apply_binop(f, operator.pow)

    def __rtruediv__(self, c):
        """Divide a scalar by this Chebfun.

        This method is called when a scalar is divided by a Chebfun, i.e., c / self.

        Args:
            c (scalar): The scalar numerator.

        Returns:
            Chebfun: A new Chebfun representing c / self.

        Note:
            This is executed when truediv(f, self) fails, which is to say whenever c
            is not a Chebfun. We proceed on the assumption f is a scalar.
        """

        def constfun(cheb, const):
            return 0.0 * cheb + const

        newfuns = [fun.initfun_adaptive(lambda x: constfun(x, c) / fun(x), fun.interval) for fun in self]
        return self.__class__(newfuns)

    @self_empty("Chebfun<empty>")
    def __repr__(self):
        """Return a string representation of the Chebfun.

        This method returns a detailed string representation of the Chebfun,
        including information about its domain, intervals, and endpoint values.

        Returns:
            str: A string representation of the Chebfun.
        """
        rowcol = "row" if self.transposed else "column"
        numpcs = self.funs.size
        plural = "" if numpcs == 1 else "s"
        header = f"Chebfun {rowcol} ({numpcs} smooth piece{plural})\n"
        domain_info = f"domain: {self.support}\n"
        toprow = "       interval       length     endpoint values\n"
        tmplat = "[{:8.2g},{:8.2g}]   {:6}  {:8.2g} {:8.2g}\n"
        rowdta = ""
        for fun in self:
            endpts = fun.support
            xl, xr = endpts
            fl, fr = fun(endpts)
            row = tmplat.format(xl, xr, fun.size, fl, fr)
            rowdta += row
        btmrow = f"vertical scale = {self.vscale:3.2g}"
        btmxtr = "" if numpcs == 1 else f"    total length = {sum([f.size for f in self])}"
        return header + domain_info + toprow + rowdta + btmrow + btmxtr

    def __rsub__(self, f):
        """Subtract this Chebfun from another object.

        This method is called when another object is subtracted by this Chebfun,
        i.e., f - self.

        Args:
            f (Chebfun or scalar): The object from which to subtract this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing f - self.
        """
        return -(self - f)

    @cast_arg_to_chebfun
    def __rpow__(self, f):
        """Raise another object to the power of this Chebfun.

        This method is called when another object is raised to the power of this Chebfun,
        i.e., f ** self.

        Args:
            f (Chebfun or scalar): The base to be raised to the power of this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing f ** self.
        """
        return f**self

    def __truediv__(self, f):
        """Divide this Chebfun by another object.

        Args:
            f (Chebfun or scalar): The divisor.

        Returns:
            Chebfun: A new Chebfun representing self / f.
        """
        # Defer to OrderTracerAST if present (for order detection)
        if getattr(f, "_is_order_tracer", False):
            return NotImplemented
        return self._apply_binop(f, operator.truediv)

    __rmul__ = __mul__
    __div__ = __truediv__
    __rdiv__ = __rtruediv__
    __radd__ = __add__

    def __str__(self):
        """Return a concise string representation of the Chebfun.

        This method returns a brief string representation of the Chebfun,
        showing its orientation, number of pieces, total size, and domain.

        Returns:
            str: A concise string representation of the Chebfun.
        """
        rowcol = "row" if self.transposed else "col"
        domain_str = f"domain {self.support}" if not self.isempty else "empty"
        out = f"<Chebfun-{rowcol},{self.funs.size},{sum([f.size for f in self])}, {domain_str}>\n"
        return out

    def __sub__(self, f):
        """Subtract another object from this Chebfun.

        Args:
            f (Chebfun or scalar): The object to subtract from this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing self - f.
        """
        # Defer to OrderTracerAST if present (for order detection)
        if getattr(f, "_is_order_tracer", False):
            return NotImplemented
        return self._apply_binop(f, operator.sub)

    # ------------------
    #  internal helpers
    # ------------------
    @self_empty()
    def _apply_binop(self, f, op):
        """Apply a binary operation between this Chebfun and another object.

        This is a funnel method used in the implementation of Chebfun binary
        operators. The high-level idea is to first break each chebfun into a
        series of pieces corresponding to the union of the domains of each
        before applying the supplied binary operator and simplifying. In the
        case of the second argument being a scalar we don't need to do the
        simplify step, since at the Tech-level these operations are defined
        such that there is no change in the number of coefficients.

        Args:
            f (Chebfun or scalar): The second operand of the binary operation.
            op (callable): The binary operation to apply (e.g., operator.add).

        Returns:
            Chebfun: A new Chebfun resulting from applying the binary operation.
        """
        if hasattr(f, "isempty") and f.isempty:
            return f

        # Check if f is scalar-like (including objects with __float__ like PointEval)
        is_scalar = np.isscalar(f)
        if not is_scalar and hasattr(f, "__float__") and hasattr(f, "ndim") and f.ndim == 0:
            # Treat as scalar if it can be converted to float and has ndim=0
            f = float(f)
            is_scalar = True

        if is_scalar:
            chbfn1 = self
            chbfn2 = f * np.ones(self.funs.size)
            simplify = False
        else:
            newdom = self.domain.union(f.domain)
            chbfn1 = self._break(newdom)
            chbfn2 = f._break(newdom)
            simplify = True
        newfuns = []
        for fun1, fun2 in zip(chbfn1, chbfn2):
            newfun = op(fun1, fun2)
            if simplify:
                newfun = newfun.simplify()
            newfuns.append(newfun)
        return self.__class__(newfuns)

    def _break(self, targetdomain):
        """Resample this Chebfun to a new domain.

        This method resamples the Chebfun to the supplied Domain object. It is
        intended as a private method since one will typically need to have
        called either Domain.union(f) or Domain.merge(f) prior to calling this method.

        Args:
            targetdomain (Domain): The domain to which this Chebfun should be resampled.

        Returns:
            Chebfun: A new Chebfun resampled to the target domain.
        """
        newfuns = []
        subintervals = targetdomain.intervals
        interval = next(subintervals)  # next(..) for Python2/3 compatibility
        for fun in self:
            while interval in fun.interval:
                newfun = fun.restrict(interval)
                newfuns.append(newfun)
                try:
                    interval = next(subintervals)
                except StopIteration:
                    break
        return self.__class__(newfuns)

    # ------------
    #  properties
    # ------------
    @property
    def breakpoints(self):
        """Get the breakpoints of this Chebfun.

        Breakpoints are the points where the Chebfun transitions from one piece to another.

        Returns:
            numpy.ndarray: Array of breakpoints.
        """
        return np.array([x for x in self.breakdata.keys()])

    @property
    @self_empty(np.array([]))
    def domain(self):
        """Get the domain of this Chebfun.

        Returns:
            Domain: A Domain object corresponding to this Chebfun.
        """
        return Domain.from_chebfun(self)

    @domain.setter
    def domain(self, new_domain):
        """Set the domain of the Chebfun by restricting to the new domain.

        Args:
            new_domain (array-like): The new domain to which this Chebfun should be restricted.
        """
        self.restrict_(new_domain)

    @property
    @self_empty(Domain([]))
    def support(self):
        """Get the support interval of this Chebfun.

        The support is the interval between the first and last breakpoints.

        Returns:
            numpy.ndarray: Array containing the first and last breakpoints.
        """
        return self.domain.support

    @property
    @self_empty(0.0)
    def hscale(self):
        """Get the horizontal scale of this Chebfun.

        The horizontal scale is the maximum absolute value of the support interval.

        Returns:
            float: The horizontal scale.
        """
        return float(np.abs(self.support).max())

    @property
    @self_empty(False)
    def iscomplex(self):
        """Check if this Chebfun has complex values.

        Returns:
            bool: True if any of the functions in this Chebfun have complex values,
                False otherwise.
        """
        return any(fun.iscomplex for fun in self)

    @property
    @self_empty(False)
    def isconst(self):
        """Check if this Chebfun represents a constant function.

        A Chebfun is constant if all of its pieces are constant with the same value.

        Returns:
            bool: True if this Chebfun represents a constant function, False otherwise.

        Note:
            TODO: find an abstract way of referencing funs[0].coeffs[0]
        """
        c = self.funs[0].coeffs[0]
        return all(fun.isconst and fun.coeffs[0] == c for fun in self)

    @property
    def isempty(self):
        """Check if this Chebfun is empty.

        An empty Chebfun contains no functions.

        Returns:
            bool: True if this Chebfun is empty, False otherwise.
        """
        return self.funs.size == 0

    @property
    @self_empty(0.0)
    def vscale(self):
        """Get the vertical scale of this Chebfun.

        The vertical scale is the maximum of the vertical scales of all pieces.

        Returns:
            float: The vertical scale.
        """
        return np.max([fun.vscale for fun in self])

    @property
    @self_empty()
    def x(self):
        """Get the identity function on the support of this Chebfun.

        This property returns a new Chebfun representing the identity function f(x) = x
        defined on the same support as this Chebfun.

        Returns:
            Chebfun: A Chebfun representing the identity function on the support of this Chebfun.
        """
        return self.__class__.initidentity(self.support)

    # -----------
    #  utilities
    # ----------

    def imag(self):
        """Get the imaginary part of this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing the imaginary part of this Chebfun.
                If this Chebfun is real-valued, returns a zero Chebfun.
        """
        if self.iscomplex:
            return self.__class__([fun.imag() for fun in self])
        else:
            return self.initconst(0, domain=self.domain)

    def real(self):
        """Get the real part of this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing the real part of this Chebfun.
                If this Chebfun is already real-valued, returns this Chebfun.
        """
        if self.iscomplex:
            return self.__class__([fun.real() for fun in self])
        else:
            return self

    def copy(self):
        """Create a deep copy of this Chebfun.

        Returns:
            Chebfun: A new Chebfun that is a deep copy of this Chebfun.
        """
        return self.__class__([fun.copy() for fun in self])

    @self_empty()
    def _restrict(self, subinterval):
        """Restrict a Chebfun to a subinterval, without simplifying.

        This is an internal method that restricts the Chebfun to a subinterval
        without performing simplification.

        Args:
            subinterval (array-like): The subinterval to which this Chebfun should be restricted.

        Returns:
            Chebfun: A new Chebfun restricted to the specified subinterval, without simplification.
        """
        newdom = self.domain.restrict(Domain(subinterval))
        return self._break(newdom)

    def restrict(self, subinterval):
        """Restrict a Chebfun to a subinterval.

        This method creates a new Chebfun that is restricted to the specified subinterval
        and simplifies the result.

        Args:
            subinterval (array-like): The subinterval to which this Chebfun should be restricted.

        Returns:
            Chebfun: A new Chebfun restricted to the specified subinterval.
        """
        return self._restrict(subinterval).simplify()

    @self_empty()
    def restrict_(self, subinterval):
        """Restrict a Chebfun to a subinterval, modifying the object in place.

        This method modifies the current Chebfun by restricting it to the specified
        subinterval and simplifying the result.

        Args:
            subinterval (array-like): The subinterval to which this Chebfun should be restricted.

        Returns:
            Chebfun: The modified Chebfun (self).
        """
        restricted = self._restrict(subinterval).simplify()
        self.funs = restricted.funs
        self.breakdata = compute_breakdata(self.funs)
        return self

    @cache
    @self_empty(np.array([]))
    def roots(self, merge=None):
        """Compute the roots of a Chebfun.

        This method finds the values x for which f(x) = 0, by computing the roots
        of each piece of the Chebfun and combining them.

        Args:
            merge (bool, optional): Whether to merge roots at breakpoints. If None,
                uses the value from preferences. Defaults to None.

        Returns:
            numpy.ndarray: Array of roots sorted in ascending order.
        """
        merge = merge if merge is not None else prefs.mergeroots
        allrts = []
        prvrts = np.array([])
        htol = 1e2 * self.hscale * prefs.eps
        for fun in self:
            rts = fun.roots()
            # ignore first root if equal to the last root of previous fun
            # TODO: there could be multiple roots at breakpoints
            if prvrts.size > 0 and rts.size > 0:
                if merge and abs(prvrts[-1] - rts[0]) <= htol:
                    rts = rts[1:]
            allrts.append(rts)
            prvrts = rts
        return np.concatenate([x for x in allrts])

    @self_empty()
    def simplify(self):
        """Simplify each fun in the chebfun."""
        return self.__class__([fun.simplify() for fun in self])

    def translate(self, c):
        """Translate a chebfun by c, i.e., return f(x-c)."""
        return self.__class__([x.translate(c) for x in self])

    # ----------
    #  calculus
    # ----------
    def cumsum(self):
        """Compute the indefinite integral (antiderivative) of the Chebfun.

        This method computes the indefinite integral of the Chebfun, with the
        constant of integration chosen so that the indefinite integral evaluates
        to 0 at the left endpoint of the domain. For piecewise functions, constants
        are added to ensure continuity across the pieces.

        Returns:
            Chebfun: A new Chebfun representing the indefinite integral of this Chebfun.
        """
        newfuns = []
        prevfun = None
        for fun in self:
            integral = fun.cumsum()
            if prevfun:
                # enforce continuity by adding the function value
                # at the right endpoint of the previous fun
                _, fb = prevfun.endvalues
                integral = integral + fb
            newfuns.append(integral)
            prevfun = integral
        return self.__class__(newfuns)

    def diff(self, n=1):
        """Compute the derivative of the Chebfun.

        This method calculates the nth derivative of the Chebfun with respect to x.
        It creates a new Chebfun where each piece is the derivative of the
        corresponding piece in the original Chebfun.

        Args:
            n: Order of differentiation (default: 1). Must be non-negative integer.

        Returns:
            Chebfun: A new Chebfun representing the nth derivative of this Chebfun.

        Examples:
            f = chebfun(lambda x: x**3)
            f.diff()    # first derivative: 3*x**2
            f.diff(2)   # second derivative: 6*x
            f.diff(3)   # third derivative: 6
        """
        if n == 0:
            return self
        if n < 0:
            raise ValueError("Derivative order must be non-negative")

        result = self
        for _ in range(n):
            dfuns = np.array([fun.diff() for fun in result])
            result = self.__class__(dfuns)
        return result

    def sum(self):
        """Compute the definite integral of the Chebfun over its domain.

        This method calculates the definite integral of the Chebfun over its
        entire domain of definition by summing the definite integrals of each
        piece.

        Returns:
            float or complex: The definite integral of the Chebfun over its domain.
        """
        return np.sum([fun.sum() for fun in self])

    def dot(self, f):
        """Compute the dot product of this Chebfun with another function.

        This method calculates the inner product (dot product) of this Chebfun
        with another function f by multiplying them pointwise and then integrating
        the result over the domain.

        Args:
            f (Chebfun or scalar): The function or scalar to compute the dot product with.
                If not a Chebfun, it will be converted to one.

        Returns:
            float or complex: The dot product of this Chebfun with f.
        """
        return (self * f).sum()

    def norm(self, p=2):
        """Compute the Lp norm of the Chebfun over its domain.

        This method calculates the Lp norm of the Chebfun. The L2 norm is the
        default and is computed as sqrt(integral(|f|^2)). For p=inf, returns
        the maximum absolute value by checking critical points (extrema).

        Args:
            p (int or float): The norm type. Supported values are 1, 2, positive
                integers/floats, or np.inf. Defaults to 2 (L2 norm).

        Returns:
            float: The Lp norm of the Chebfun.

        Examples:
            >>> from chebpy import chebfun
            >>> import numpy as np
            >>> f = chebfun(lambda x: x**2, [-1, 1])
            >>> np.allclose(f.norm(), 0.6324555320336759)  # L2 norm
            True
            >>> np.allclose(f.norm(np.inf), 1.0)  # Maximum absolute value
            True
        """
        if p == 2:
            # L2 norm: sqrt(integral(|f|^2))
            return np.sqrt(self.dot(self))
        elif p == np.inf:
            # L-infinity norm: max|f(x)|
            # Find extrema by checking derivative roots (critical points)
            # Also check endpoints
            df = self.diff()
            critical_pts = df.roots()
            # Add endpoints
            endpoints = np.array([self.domain[0], self.domain[-1]])
            # Combine all test points
            test_pts = np.concatenate([critical_pts, endpoints])
            # Evaluate and find max
            vals = np.abs(self(test_pts))
            return np.max(vals)
        elif p == 1:
            # L1 norm: integral(|f|)
            return self.absolute().sum()
        elif p > 0:
            # General Lp norm: (integral(|f|^p))^(1/p)
            f_abs = self.absolute()
            f_pow_p = f_abs**p
            integral = f_pow_p.sum()
            return integral ** (1.0 / p)
        else:
            raise ValueError(f"norm(p={p}): p must be positive or np.inf")

    # ----------
    #  utilities
    # ----------
    @self_empty()
    def absolute(self):
        """Absolute value of a Chebfun."""
        newdom = self.domain.merge(self.roots())
        funs = [x.absolute() for x in self._break(newdom)]
        return self.__class__(funs)

    abs = absolute

    @self_empty()
    @cast_arg_to_chebfun
    def maximum(self, other):
        """Pointwise maximum of self and another chebfun."""
        return self._maximum_minimum(other, operator.ge)

    @self_empty()
    @cast_arg_to_chebfun
    def minimum(self, other):
        """Pointwise mimimum of self and another chebfun."""
        return self._maximum_minimum(other, operator.lt)

    def _maximum_minimum(self, other, comparator):
        """Method for computing the pointwise maximum/minimum of two Chebfuns.

        This internal method implements the algorithm for computing the pointwise
        maximum or minimum of two Chebfun objects, based on the provided comparator.
        It is used by the maximum() and minimum() methods.

        Args:
            other (Chebfun): Another Chebfun to compare with this one.
            comparator (callable): A function that compares two values and returns
                a boolean. For maximum, this is operator.ge (>=), and for minimum,
                this is operator.lt (<).

        Returns:
            Chebfun: A new Chebfun representing the pointwise maximum or minimum.
        """
        # Handle empty Chebfuns
        if self.isempty or other.isempty:
            return self.__class__.initempty()

        # Find the intersection of domains
        try:
            # Try to use union if supports match
            newdom = self.domain.union(other.domain)
        except SupportMismatch:
            # If supports don't match, find the intersection
            a_min, a_max = self.support
            b_min, b_max = other.support

            # Calculate intersection
            c_min = max(a_min, b_min)
            c_max = min(a_max, b_max)

            # If there's no intersection, return empty
            if c_min >= c_max:
                return self.__class__.initempty()

            # Restrict both functions to the intersection
            self_restricted = self.restrict([c_min, c_max])
            other_restricted = other.restrict([c_min, c_max])

            # Recursively call with the restricted functions
            return self_restricted._maximum_minimum(other_restricted, comparator)

        # Continue with the original algorithm
        roots = (self - other).roots()
        newdom = newdom.merge(roots)
        switch = newdom.support.merge(roots)

        # Handle the case where switch is empty
        if switch.size == 0:  # pragma: no cover
            return self.__class__.initempty()

        keys = 0.5 * ((-1) ** np.arange(switch.size - 1) + 1)
        if switch.size > 0 and comparator(other(switch[0]), self(switch[0])):
            keys = 1 - keys
        funs = np.array([])
        for interval, use_self in zip(switch.intervals, keys):
            subdom = newdom.restrict(interval)
            if use_self:
                subfun = self.restrict(subdom)
            else:
                subfun = other.restrict(subdom)
            funs = np.append(funs, subfun.funs)
        return self.__class__(funs)

    # ----------
    #  plotting
    # ----------
    def plot(self, ax=None, **kwds):
        """Plot the Chebfun over its domain.

        This method plots the Chebfun over its domain using matplotlib.
        For complex-valued Chebfuns, it plots the real part against the imaginary part.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None,
                a new axes will be created. Defaults to None.
            **kwds: Additional keyword arguments to pass to matplotlib's plot function.

        Returns:
            matplotlib.axes.Axes: The axes on which the plot was created.
        """
        return plotfun(self, self.support, ax=ax, **kwds)

    def plotcoeffs(self, ax=None, **kwds):
        """Plot the coefficients of the Chebfun on a semilogy scale.

        This method plots the absolute values of the coefficients for each piece
        of the Chebfun on a semilogy scale, which is useful for visualizing the
        decay of coefficients in the Chebyshev series.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None,
                a new axes will be created. Defaults to None.
            **kwds: Additional keyword arguments to pass to matplotlib's semilogy function.

        Returns:
            matplotlib.axes.Axes: The axes on which the plot was created.
        """
        ax = ax or plt.gca()
        for fun in self:
            fun.plotcoeffs(ax=ax, **kwds)
        return ax

    # --------------------
    #  splitting methods
    # --------------------
    @classmethod
    def _initfun_splitting(cls, f, domain):
        """Initialize a Chebfun using automatic domain splitting.

        This implements the algorithm from Pachon, Platte, Trefethen,
        "Piecewise smooth chebfuns" (IMA J. Numer. Anal., 2010).

        Args:
            f: Function to approximate
            domain: Domain object

        Returns:
            Chebfun with potentially multiple pieces
        """
        # Constants from MATLAB Chebfun
        split_max_length = 6000  # Maximum total points (MATLAB default: prevent pathological subdivision)
        split_length = 160  # Target points per piece for splitting
        max_pow2_split = int(np.ceil(np.log2(split_length - 1)))  # = 8

        # Initialize list of intervals to process and funs created
        # Each entry is (left, right, left_value, right_value)
        # left_value and right_value store function values at endpoints for continuity
        funs = []
        sad_intervals = []  # List of (a, b, fun, width) - intervals that are "unhappy"

        # Horizontal scale
        hscale = domain.support[1] - domain.support[0]
        vscale = 0.0

        # Minimum interval threshold - use aggressive value to prevent pathological subdivision
        # MATLAB uses 4e-14*hscale in getFun, but we need larger threshold to stop earlier
        min_interval = 1e-12 * hscale

        # Initial pass: try to construct on each interval in domain
        for interval in domain.intervals:
            a, b = float(interval[0]), float(interval[1])
            fun, is_happy, vscale = cls._get_fun_splitting(f, a, b, hscale, vscale, max_pow2_split)

            if is_happy:
                funs.append(fun)
            else:
                # Not resolved - add to sad intervals with width
                sad_intervals.append((a, b, fun, b - a))

        # Splitting loop - process sad intervals until all resolved or limit reached
        total_points = sum(fun.size for fun in funs)
        max_iterations = 100  # Safety limit (MATLAB typically needs ~10-20 for most functions)

        iteration = 0
        while sad_intervals and total_points < split_max_length and iteration < max_iterations:
            iteration += 1

            # Choose the LARGEST sad interval by WIDTH (MATLAB behavior)
            sad_intervals.sort(key=lambda x: -x[3])  # Sort by width, largest first
            a, b, _, _ = sad_intervals.pop(0)

            # CRITICAL FIX: Check if interval is too small to split BEFORE edge detection
            # This prevents pathological subdivision near singularities (MATLAB: line 268)
            # Use <= to catch zero-width intervals and add eps for numerical safety
            if (b - a) <= max(min_interval, 10 * np.finfo(float).eps * hscale):
                # Interval too small - accept current approximation as constant at midpoint
                fun, is_happy, vscale = cls._get_fun_splitting(f, a, b, hscale, vscale, max_pow2_split)
                funs.append(fun)
                total_points += fun.size
                continue

            # Detect edge in this interval
            edge = cls._detect_edge(f, a, b, hscale, vscale)

            if edge is None or edge <= a or edge >= b:
                # No edge found or edge at boundary - just bisect
                edge = (a + b) / 2.0
            else:
                # Snap edge to nearby "nice" values if very close
                # This helps when the singularity is at an exact point like 0
                edge = cls._snap_edge(edge, a, b, hscale)

            # Check if split would create intervals below minimum size
            # Use <= to catch zero-width intervals and add eps for numerical safety
            min_split_width = max(min_interval, 10 * np.finfo(float).eps * hscale)
            if (edge - a) <= min_split_width or (b - edge) <= min_split_width:
                # Split would create too-small interval - accept current approximation
                fun, is_happy, vscale = cls._get_fun_splitting(f, a, b, hscale, vscale, max_pow2_split)
                funs.append(fun)
                total_points += fun.size
                continue

            # Detect if this is a jump discontinuity by checking one-sided limits
            # This helps handle functions like sign(x) where sign(0)=0 but limits are -1/+1
            is_jump, left_limit, right_limit = cls._detect_jump_limits(f, edge, hscale)

            # Detect if this is a pole singularity (function blows up to infinity)
            # Only check after we've done some splitting to avoid false positives
            # (during initial construction, vscale might not be well-established)
            if len(funs) > 0:  # Only check if we've already created some funs
                is_pole = cls._detect_pole_singularity(f, edge, hscale, vscale)

                if is_pole:
                    # Pole singularity detected - stop splitting this interval
                    # Just accept the best approximation we can get
                    fun, is_happy, vscale = cls._get_fun_splitting(f, a, b, hscale, vscale, max_pow2_split)
                    funs.append(fun)
                    total_points += fun.size
                    continue

            # Try to construct on left and right subintervals
            for sub_a, sub_b in [(a, edge), (edge, b)]:
                # If we detected a jump discontinuity, use one-sided limit at the edge
                if is_jump:
                    if sub_b == edge:
                        # Left interval [a, edge] - use left limit at edge
                        f_mod = cls._make_limit_function(f, edge, left_limit, "right")
                    else:
                        # Right interval [edge, b] - use right limit at edge
                        f_mod = cls._make_limit_function(f, edge, right_limit, "left")
                else:
                    f_mod = f

                fun, is_happy, vscale = cls._get_fun_splitting(f_mod, sub_a, sub_b, hscale, vscale, max_pow2_split)
                if is_happy:
                    funs.append(fun)
                    total_points += fun.size
                else:
                    # Still not resolved - add to sad intervals
                    sad_intervals.append((sub_a, sub_b, fun, sub_b - sub_a))

        # If we hit iteration limit, add remaining sad intervals as unhappy funs
        if iteration >= max_iterations or total_points >= split_max_length:
            for sad_a, sad_b, sad_fun, _ in sad_intervals:
                funs.append(sad_fun)
                total_points += sad_fun.size

            warnings.warn(
                f"Splitting reached limit ({iteration} iterations, {total_points} points). "
                "Function may not be fully resolved."
            )

        # Sort funs by left endpoint and create Chebfun
        funs.sort(key=lambda fun: fun.support[0])
        return cls(funs)

    @classmethod
    def _get_fun_splitting(cls, f, a, b, hscale, vscale, maxpow2):
        """Construct a fun for splitting mode with limited degree.

        Args:
            f: Function to approximate
            a: Left interval endpoint.
            b: Right interval endpoint.
            hscale: Horizontal scale
            vscale: Current vertical scale
            maxpow2: Maximum power of 2 for number of points (8 for splitLength=160)

        Returns:
            tuple: (fun, is_happy, updated_vscale)
        """
        # Check for tiny interval - treat as constant
        interval_width = b - a
        if interval_width < 4 * np.finfo(float).eps * hscale:
            # Very small interval - use constant value at midpoint
            mid = (a + b) / 2.0
            try:
                c = float(f(mid))
            except (ValueError, TypeError, ZeroDivisionError, OverflowError, FloatingPointError):
                c = 0.0
            if not np.isfinite(c):
                c = 0.0
            interval = Interval(a, b)
            fun = Bndfun.initconst(c, interval)
            return fun, True, max(vscale, abs(c))

        # Create interval and try adaptive construction with limited maxpow2
        interval = Interval(a, b)

        # Save and temporarily modify prefs
        original_maxpow2 = prefs.maxpow2
        try:
            prefs.maxpow2 = maxpow2  # Limit to 2^8 = 256 points for splitting

            # Attempt construction
            with np.errstate(all="ignore"):
                fun = Bndfun.initfun_adaptive(f, interval)
                # Check if construction converged (happy = size < max possible)
                max_size = 2**maxpow2 + 1
                is_happy = fun.size < max_size

            # Update vscale
            new_vscale = max(vscale, fun.vscale)

        finally:
            prefs.maxpow2 = original_maxpow2

        return fun, is_happy, new_vscale

    @classmethod
    def _detect_edge(cls, f, a, b, hscale, vscale):
        """Detect edge (discontinuity or singularity) in [a, b].

        Implements the Pachon-Platte-Trefethen algorithm using finite differences.

        Args:
            f: Function to test
            a: Left interval endpoint.
            b: Right interval endpoint.
            hscale: Horizontal scale of full domain
            vscale: Vertical scale

        Returns:
            Location of edge, or None if no edge detected
        """
        num_test_ders = 4  # Test derivatives 1-4
        grid_size_1 = 50  # Initial grid
        grid_size_refine = 15  # Refinement grid

        # Compute norm_inf of first num_test_ders derivatives
        new_a, new_b, max_der = cls._find_max_der(f, a, b, num_test_ders, grid_size_1)

        if max_der is None:
            return (a + b) / 2.0

        # Track endpoints for the highest derivative
        ends = [new_a[num_test_ders - 1], new_b[num_test_ders - 1]]

        # Main refinement loop
        max_iterations = 100
        iteration = 0

        # Main loop: continues while derivatives are finite and interval is large enough
        # MATLAB: while (maxDer(numTestDers) ~= inf) && ~isnan(maxDer) && (diff(ends) > eps*hscale)
        while (
            max_der[num_test_ders - 1] != np.inf
            and not np.isnan(max_der[num_test_ders - 1])
            and (ends[1] - ends[0]) > np.finfo(float).eps * hscale
            and iteration < max_iterations
        ):
            iteration += 1

            # Keep track of previous max derivatives
            max_der_prev = max_der.copy()

            # Compute maximum derivatives on refined interval
            new_a, new_b, max_der = cls._find_max_der(f, ends[0], ends[1], num_test_ders, grid_size_refine)

            if max_der is None:
                return (ends[0] + ends[1]) / 2.0

            # Check which derivatives are growing (MATLAB criterion line 144-145)
            # Growth factor must be > (5.5 - d) where d is derivative order
            # AND derivative must be > 10*vscale/hscale^d
            vscale_eff = max(vscale, np.finfo(float).eps)
            growing = []
            for d in range(1, num_test_ders + 1):
                growth_threshold = 5.5 - d
                magnitude_threshold = 10 * vscale_eff / (hscale**d)
                if max_der[d - 1] > growth_threshold * max_der_prev[d - 1] and max_der[d - 1] > magnitude_threshold:
                    growing.append(d)

            if not growing:
                # Derivatives not growing - no edge detected
                return None

            # Use lowest growing derivative (MATLAB: find(..., 1, 'first'))
            num_test_ders = growing[0]

            if num_test_ders == 1 and (ends[1] - ends[0]) < 1e-3 * hscale:
                # Blow up in first derivative with small interval - use findJump bisection
                edge = cls._find_jump(f, ends[0], ends[1], vscale, hscale)
                if edge is not None:
                    return edge
                # If findJump returns None, fall through to return midpoint

            # Update endpoints for next iteration (MATLAB line 157)
            ends = [new_a[num_test_ders - 1], new_b[num_test_ders - 1]]

        # Return midpoint of final interval (MATLAB line 173)
        return (ends[0] + ends[1]) / 2.0

    @classmethod
    def _snap_edge(cls, edge, a, b, hscale):
        """Snap an edge to nearby "nice" values if very close.

        This helps when singularities occur at exact mathematical points like 0,
        pi/10, etc. Without snapping, numerical edge detection may find values
        like 6.8e-8 when the true edge is at 0, leading to many tiny intervals.

        Args:
            edge: Detected edge location
            a: Left interval endpoint.
            b: Right interval endpoint.
            hscale: Horizontal scale for tolerance

        Returns:
            Snapped edge value
        """
        # Tolerance for snapping (relative to hscale)
        # Use a more generous tolerance for higher-order singularities
        # which are harder to locate precisely
        snap_tol = 1e-3 * hscale  # Increased from 1e-6 to 1e-3

        # List of "nice" values to consider
        # 0 is always nice
        nice_values = [0.0]

        # Also consider integer values in the interval
        floor_a = int(np.floor(a))
        ceil_b = int(np.ceil(b))
        for i in range(floor_a, ceil_b + 1):
            if i != 0:
                nice_values.append(float(i))

        # Consider multiples of pi/10 (common in trigonometric functions)
        for k in range(-100, 101):
            val = k * np.pi / 10
            if a < val < b:
                nice_values.append(val)

        # Check if edge is close to any nice value
        for nice in nice_values:
            if a < nice < b and abs(edge - nice) < snap_tol:
                return nice

        return edge

    @classmethod
    def _find_max_der(cls, f, a, b, num_ders, grid_size):
        """Find maximum of derivatives 1 through num_ders using finite differences.

        Args:
            f: Function
            a: Left interval endpoint.
            b: Right interval endpoint.
            num_ders: Number of derivatives to estimate
            grid_size: Number of grid points

        Returns:
            tuple: (new_a, new_b, max_der) arrays of length num_ders
                   new_a[d], new_b[d] bracket the max location of derivative d+1
                   max_der[d] is the max norm of derivative d+1
        """
        max_der = np.zeros(num_ders)
        new_a = np.full(num_ders, a)
        new_b = np.full(num_ders, b)

        # Generate grid
        dx = (b - a) / (grid_size - 1)
        x = np.linspace(a, b, grid_size)

        # Evaluate function
        with np.errstate(all="ignore"):
            try:
                y = np.atleast_1d(f(x))
                if y.shape[0] != grid_size:
                    y = np.array([f(xi) for xi in x])
            except (ValueError, TypeError, ZeroDivisionError, OverflowError, FloatingPointError):
                return None, None, None

        if not np.all(np.isfinite(y)):
            # Function has non-finite values - there's definitely an edge
            # Find where non-finite values occur
            bad_idx = np.where(~np.isfinite(y))[0]
            if len(bad_idx) > 0:
                idx = bad_idx[0]
                if idx > 0:
                    new_a[:] = x[idx - 1]
                if idx < len(x) - 1:
                    new_b[:] = x[idx + 1]
                max_der[:] = np.inf
                return new_a, new_b, max_der
            return None, None, None

        # Compute undivided differences (MATLAB's approach)
        dy = y.copy()
        x_mid = x.copy()
        for d in range(num_ders):
            dy = np.diff(dy)
            x_mid = (x_mid[:-1] + x_mid[1:]) / 2.0

            if len(dy) == 0:
                break

            # Find maximum absolute value
            abs_dy = np.abs(dy)
            idx = np.argmax(abs_dy)
            max_der[d] = abs_dy[idx]

            # Update brackets
            if idx > 0:
                new_a[d] = x_mid[idx - 1]
            else:
                new_a[d] = a
            if idx < len(x_mid) - 1:
                new_b[d] = x_mid[idx + 1]
            else:
                new_b[d] = b

        # Convert to actual derivative estimates (divide by dx^d)
        if dx**num_ders <= np.finfo(float).eps:
            max_der[:] = np.inf
        else:
            for d in range(num_ders):
                max_der[d] = max_der[d] / (dx ** (d + 1))

        return new_a, new_b, max_der

    @classmethod
    def _find_jump(cls, f, a, b, vscale, hscale):
        """Locate a jump discontinuity using bisection.

        Implements MATLAB's findJump algorithm from detectEdge.m.

        Args:
            f: Function
            a: Left interval endpoint.
            b: Right interval endpoint.
            vscale: Vertical scale
            hscale: Horizontal scale

        Returns:
            Location of jump, or None if no jump detected
        """
        # Evaluate at endpoints
        with np.errstate(all="ignore"):
            try:
                ya = float(f(a))
                yb = float(f(b))
            except (ValueError, TypeError, ZeroDivisionError, OverflowError, FloatingPointError):
                return (a + b) / 2.0

        if not np.isfinite(ya) or not np.isfinite(yb):
            return (a + b) / 2.0

        # Estimate derivative
        max_der = abs(ya - yb) / (b - a) if (b - a) > 0 else np.inf

        # If derivative is very small, probably a false edge
        vscale_eff = max(vscale, np.finfo(float).eps)
        if max_der < 1e-5 * vscale_eff / hscale:
            return None

        # Bisection loop
        e1 = (a + b) / 2.0
        e0 = e1 + 1  # Force first iteration
        cont = 0  # Counter for stalled iterations

        while (cont < 2 or max_der == np.inf) and e0 != e1:
            c = (a + b) / 2.0

            with np.errstate(all="ignore"):
                try:
                    yc = float(f(c))
                except (ValueError, TypeError, ZeroDivisionError, OverflowError, FloatingPointError):
                    return c

            if not np.isfinite(yc):
                return c

            # Undivided differences (not divided by interval width)
            dyl = abs(yc - ya)
            dyr = abs(yb - yc)

            max_der_prev = max_der

            if dyl > dyr:
                # Jump is in [a, c]
                b = c
                yb = yc
                max_der = dyl / (b - a) if (b - a) > 0 else np.inf
            else:
                # Jump is in [c, b]
                a = c
                ya = yc
                max_der = dyr / (b - a) if (b - a) > 0 else np.inf

            e0 = e1
            e1 = (a + b) / 2.0

            # Check if derivative stopped growing
            if max_der < max_der_prev * 1.5:
                cont += 1

        # Final refinement: check floating point precision (MATLAB lines 247-256)
        # When we've converged to machine precision, determine exact edge location
        if abs(e0 - e1) <= 2 * np.finfo(float).eps * abs(e0):
            with np.errstate(all="ignore"):
                try:
                    # Look at the floating point at the right
                    yright = float(f(b + np.finfo(float).eps * abs(b)))
                    # If there is a small jump at b, that is the edge
                    if abs(yright - yb) > 100 * np.finfo(float).eps * vscale_eff:
                        return b
                    else:
                        return a
                except (ValueError, TypeError, ZeroDivisionError, OverflowError, FloatingPointError):
                    pass

        return e1

    @classmethod
    def _detect_jump_limits(cls, f, edge, hscale):
        """Detect if there's a jump discontinuity at edge and compute one-sided limits.

        A jump discontinuity is detected if the left and right limits differ significantly,
        indicating that the function value at the edge point may differ from the limiting
        behavior on either side.

        Args:
            f: Function to test
            edge: Point where potential jump discontinuity is located
            hscale: Horizontal scale for determining "small" offset

        Returns:
            tuple: (is_jump, left_limit, right_limit)
                   is_jump: True if a jump discontinuity was detected
                   left_limit: Estimated left-hand limit at edge
                   right_limit: Estimated right-hand limit at edge
        """
        # Use a small offset to estimate one-sided limits
        # We use multiple points and extrapolate for robustness
        eps = np.finfo(float).eps
        small_offset = max(1e-10 * hscale, 1000 * eps * abs(edge) if edge != 0 else 1000 * eps)

        with np.errstate(all="ignore"):
            try:
                # Sample points approaching from the left
                x_left = np.array([edge - 4 * small_offset, edge - 2 * small_offset, edge - small_offset])
                y_left = np.array([float(f(x)) for x in x_left])

                # Sample points approaching from the right
                x_right = np.array([edge + small_offset, edge + 2 * small_offset, edge + 4 * small_offset])
                y_right = np.array([float(f(x)) for x in x_right])

                # Value at the edge itself
                y_edge = float(f(edge))

                # Check if all values are finite
                if not (np.all(np.isfinite(y_left)) and np.all(np.isfinite(y_right)) and np.isfinite(y_edge)):
                    return False, None, None

                # Estimate one-sided limits by extrapolation (linear for simplicity)
                # Left limit: use the closest point as the estimate
                left_limit = y_left[-1]  # Value at edge - small_offset

                # Right limit: use the closest point as the estimate
                right_limit = y_right[0]  # Value at edge + small_offset

                # Detect jump: limits differ significantly from each other OR
                # the function value at edge differs from both limits
                vscale = max(abs(left_limit), abs(right_limit), abs(y_edge), eps)
                jump_threshold = 0.01 * vscale  # 1% of vscale

                # Check for jump between left and right limits
                left_right_diff = abs(left_limit - right_limit)

                # Check if edge value differs from limits (like sign(0) = 0 vs limits of -1/+1)
                edge_left_diff = abs(y_edge - left_limit)
                edge_right_diff = abs(y_edge - right_limit)

                is_jump = (
                    left_right_diff > jump_threshold
                    or edge_left_diff > jump_threshold
                    or edge_right_diff > jump_threshold
                )

                if is_jump:
                    return True, left_limit, right_limit
                else:
                    return False, None, None

            except (ValueError, TypeError, ZeroDivisionError, OverflowError, FloatingPointError):
                return False, None, None

    @classmethod
    def _make_limit_function(cls, f, edge, limit_value, replace_side):
        """Create a modified function that replaces the endpoint value with a limit value.

        This is used to construct chebfuns on intervals where the function has a jump
        discontinuity at the edge. Instead of using f(edge), which may be discontinuous
        with the rest of the interval, we use the one-sided limit.

        Args:
            f: Original function
            edge: Point where the limit is taken
            limit_value: The one-sided limit value to use at the edge
            replace_side: 'left' to replace f(edge) when approaching from the left endpoint,
                         'right' to replace f(edge) when approaching from the right endpoint

        Returns:
            Modified function that returns limit_value at points very close to edge
        """
        eps = np.finfo(float).eps
        tol = 100 * eps * max(abs(edge), 1.0)

        def f_modified(x):
            x = np.atleast_1d(x)
            result = np.empty_like(x, dtype=float)

            # Find points that are essentially at the edge
            at_edge = np.abs(x - edge) < tol

            # Evaluate normally at non-edge points
            if np.any(~at_edge):
                with np.errstate(all="ignore"):
                    result[~at_edge] = f(x[~at_edge])

            # Replace edge points with limit value
            if np.any(at_edge):
                result[at_edge] = limit_value

            return result if len(result) > 1 else result[0]

        return f_modified

    @classmethod
    def _detect_pole_singularity(cls, f, edge, hscale, vscale):
        """Detect if there's a pole singularity (blowup to infinity) at edge.

        A pole singularity is characterized by the function values growing very large
        as we approach the edge point, indicating the function is not representable
        by polynomials and we should stop splitting.

        Args:
            f: Function to test
            edge: Point where singularity might be
            hscale: Horizontal scale
            vscale: Current vertical scale estimate

        Returns:
            bool: True if pole singularity detected, False otherwise
        """
        eps = np.finfo(float).eps

        # Sample points approaching the edge from both sides
        # Use smaller offsets to detect poles
        small_offset = max(1e-8 * hscale, 1000 * eps * abs(edge) if edge != 0 else 1000 * eps)

        try:
            # Sample from left and right
            x_left = edge - small_offset
            x_right = edge + small_offset

            with np.errstate(all="ignore"):
                y_left = float(f(x_left))
                y_right = float(f(x_right))

            # Check for inf/nan
            if not (np.isfinite(y_left) and np.isfinite(y_right)):
                return True

            # Check if values are extremely large compared to vscale
            # Pole threshold: function values > 1e6 * vscale suggest a pole
            # Lower threshold to catch poles earlier before excessive subdivision
            max_val = max(abs(y_left), abs(y_right))
            pole_threshold = 1e6 * max(vscale, eps)

            if max_val > pole_threshold:
                return True

            return False

        except (ValueError, OverflowError, ZeroDivisionError):
            # Function evaluation failed - likely a pole
            return True


# ---------
#  ufuncs
# ---------
def add_ufunc(op):
    """Add a NumPy universal function method to the Chebfun class.

    This function creates a method that applies a NumPy universal function (ufunc)
    to each piece of a Chebfun and returns a new Chebfun representing the result.

    Args:
        op (callable): The NumPy universal function to apply.

    Note:
        The created method will have the same name as the NumPy function
        and will take no arguments other than self.
    """

    @self_empty()
    def method(self):
        """Apply a NumPy universal function to this Chebfun.

        This method applies a NumPy universal function (ufunc) to each piece
        of this Chebfun and returns a new Chebfun representing the result.

        Args:
            self (Chebfun): The Chebfun object to which the function is applied.

        Returns:
            Chebfun: A new Chebfun representing op(f(x)).
        """
        return self.__class__([op(fun) for fun in self])

    name = op.__name__
    method.__name__ = name
    method.__doc__ = method.__doc__
    setattr(Chebfun, name, method)


ufuncs = (
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
