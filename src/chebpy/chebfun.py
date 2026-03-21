"""Implementation of the Chebfun class for piecewise function approximation.

This module provides the Chebfun class, which is the main user-facing class in the
ChebPy package. It represents functions using piecewise polynomial approximations
on arbitrary intervals, allowing for operations such as integration, differentiation,
root-finding, and more.

The Chebfun class is inspired by the MATLAB package of the same name and provides
similar functionality for working with functions rather than numbers.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterator
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .algorithms import _conv_legendre, cheb2leg, leg2cheb
from .bndfun import Bndfun
from .chebtech import Chebtech
from .decorators import cache, cast_arg_to_chebfun, float_argument, self_empty
from .exceptions import BadFunLengthArgument, SupportMismatch
from .plotting import plotfun
from .settings import _preferences as prefs
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

    def __init__(self, funs: Any) -> None:
        """Initialize a Chebfun object.

        Args:
            funs (list): List of function objects to be included in the Chebfun.
                These will be checked and sorted using check_funs.
        """
        self.funs = check_funs(funs)
        self.breakdata = compute_breakdata(self.funs)
        self.transposed = False

    @classmethod
    def initempty(cls) -> Chebfun:
        """Initialize an empty Chebfun.

        Returns:
            Chebfun: An empty Chebfun object with no functions.

        Examples:
            >>> f = Chebfun.initempty()
            >>> f.isempty
            True
        """
        return cls([])

    @classmethod
    def initidentity(cls, domain: Any = None) -> Chebfun:
        """Initialize a Chebfun representing the identity function f(x) = x.

        Args:
            domain (array-like, optional): Domain on which to define the identity function.
                If None, uses the default domain from preferences.

        Returns:
            Chebfun: A Chebfun object representing the identity function on the specified domain.

        Examples:
            >>> import numpy as np
            >>> x = Chebfun.initidentity([-1, 1])
            >>> float(x(0.5))
            0.5
            >>> np.allclose(x([0, 0.5, 1]), [0, 0.5, 1])
            True
        """
        return cls(generate_funs(domain, Bndfun.initidentity))

    @classmethod
    def initconst(cls, c: Any, domain: Any = None) -> Chebfun:
        """Initialize a Chebfun representing a constant function f(x) = c.

        Args:
            c (float or complex): The constant value.
            domain (array-like, optional): Domain on which to define the constant function.
                If None, uses the default domain from preferences.

        Returns:
            Chebfun: A Chebfun object representing the constant function on the specified domain.

        Examples:
            >>> import numpy as np
            >>> f = Chebfun.initconst(3.14, [-1, 1])
            >>> float(f(0))
            3.14
            >>> float(f(0.5))
            3.14
            >>> np.allclose(f([0, 0.5, 1]), [3.14, 3.14, 3.14])
            True
        """
        return cls(generate_funs(domain, Bndfun.initconst, {"c": c}))

    @classmethod
    def initfun_adaptive(cls, f: Callable[..., Any], domain: Any = None) -> Chebfun:
        """Initialize a Chebfun by adaptively sampling a function.

        This method determines the appropriate number of points needed to represent
        the function to the specified tolerance using an adaptive algorithm.

        Args:
            f (callable): The function to be approximated.
            domain (array-like, optional): Domain on which to define the function.
                If None, uses the default domain from preferences.

        Returns:
            Chebfun: A Chebfun object representing the function on the specified domain.

        Examples:
            >>> import numpy as np
            >>> f = Chebfun.initfun_adaptive(lambda x: np.sin(x), [-np.pi, np.pi])
            >>> bool(abs(f(0)) < 1e-10)
            True
            >>> bool(abs(f(np.pi/2) - 1) < 1e-10)
            True
        """
        return cls(generate_funs(domain, Bndfun.initfun_adaptive, {"f": f}))

    @classmethod
    def initfun_fixedlen(cls, f: Callable[..., Any], n: Any, domain: Any = None) -> Chebfun:
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
            for interval, length in zip(domain.intervals, nn, strict=False):
                funs.append(Bndfun.initfun_fixedlen(f, interval, length))
        return cls(funs)

    @classmethod
    def initfun(cls, f: Callable[..., Any], domain: Any = None, n: Any = None) -> Chebfun:
        """Initialize a Chebfun from a function.

        This is a general-purpose constructor that delegates to either initfun_adaptive
        or initfun_fixedlen based on whether n is provided.

        Args:
            f (callable): The function to be approximated.
            domain (array-like, optional): Domain on which to define the function.
                If None, uses the default domain from preferences.
            n (int or array-like, optional): Number of points to use. If None, determines
                the number adaptively. If provided, uses a fixed number of points.

        Returns:
            Chebfun: A Chebfun object representing the function on the specified domain.
        """
        if n is None:
            return cls.initfun_adaptive(f, domain)
        else:
            return cls.initfun_fixedlen(f, n, domain)

    # --------------------
    #  operator overloads
    # --------------------
    def __add__(self, f: Any) -> Any:
        """Add a Chebfun with another Chebfun or a scalar.

        Args:
            f (Chebfun or scalar): The object to add to this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing the sum.
        """
        return self._apply_binop(f, operator.add)

    @self_empty(np.array([]))
    @float_argument
    def __call__(self, x: Any) -> Any:
        """Evaluate the Chebfun at points x.

        This method evaluates the Chebfun at the specified points. It handles interior
        points, breakpoints, and points outside the domain appropriately.

        Args:
            x (float or array-like): Points at which to evaluate the Chebfun.

        Returns:
            float or numpy.ndarray: The value(s) of the Chebfun at the specified point(s).
                Returns a scalar if x is a scalar, otherwise an array of the same size as x.
        """
        # initialise output
        dtype = complex if self.iscomplex else float
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
        return out

    def __iter__(self) -> Iterator[Any]:
        """Return an iterator over the functions in this Chebfun.

        Returns:
            iterator: An iterator over the functions (funs) in this Chebfun.
        """
        return self.funs.__iter__()

    def __len__(self) -> int:
        """Return the total number of coefficients across all funs.

        Returns:
            int: The sum of sizes of all constituent funs.
        """
        return sum(f.size for f in self.funs)

    def __eq__(self, other: object) -> bool:
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
        return bool(np.all(np.abs(self(xx) - other(xx)) <= tol))

    def __mul__(self, f: Any) -> Any:
        """Multiply a Chebfun with another Chebfun or a scalar.

        Args:
            f (Chebfun or scalar): The object to multiply with this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing the product.
        """
        return self._apply_binop(f, operator.mul)

    def __neg__(self) -> Chebfun:
        """Return the negative of this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing -f(x).
        """
        return self.__class__(-self.funs)

    def __pos__(self) -> Chebfun:
        """Return the positive of this Chebfun (which is the Chebfun itself).

        Returns:
            Chebfun: This Chebfun object (unchanged).
        """
        return self

    def __abs__(self) -> Chebfun:
        """Return the absolute value of this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing |f(x)|.
        """
        abs_funs = []
        for fun in self.funs:
            abs_funs.append(fun.absolute())
        return self.__class__(abs_funs)

    def __pow__(self, f: Any) -> Any:
        """Raise this Chebfun to a power.

        Args:
            f (Chebfun or scalar): The exponent to which this Chebfun is raised.

        Returns:
            Chebfun: A new Chebfun representing self^f.
        """
        return self._apply_binop(f, operator.pow)

    def __rtruediv__(self, c: Any) -> Chebfun:
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

        def constfun(cheb: Any, const: Any) -> Any:
            return 0.0 * cheb + const

        def make_divfun(fun: Any) -> Callable[..., Any]:
            return lambda x: constfun(x, c) / fun(x)

        newfuns = [fun.initfun_adaptive(make_divfun(fun), fun.interval) for fun in self]
        return self.__class__(newfuns)

    @self_empty("Chebfun<empty>")
    def __repr__(self) -> str:
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

    def __rsub__(self, f: Any) -> Any:
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
    def __rpow__(self, f: Any) -> Any:
        """Raise another object to the power of this Chebfun.

        This method is called when another object is raised to the power of this Chebfun,
        i.e., f ** self.

        Args:
            f (Chebfun or scalar): The base to be raised to the power of this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing f ** self.
        """
        return f**self

    def __truediv__(self, f: Any) -> Any:
        """Divide this Chebfun by another object.

        Args:
            f (Chebfun or scalar): The divisor.

        Returns:
            Chebfun: A new Chebfun representing self / f.
        """
        return self._apply_binop(f, operator.truediv)

    __rmul__ = __mul__
    __div__ = __truediv__
    __rdiv__ = __rtruediv__
    __radd__ = __add__

    def __str__(self) -> str:
        """Return a human-readable string representation of the Chebfun.

        This method returns the same detailed representation as ``__repr__``,
        so that ``print(f)`` shows the full summary table.  This is consistent
        with the behaviour of numpy and pandas objects.

        Returns:
            str: A detailed string representation of the Chebfun.
        """
        return repr(self)

    def __sub__(self, f: Any) -> Any:
        """Subtract another object from this Chebfun.

        Args:
            f (Chebfun or scalar): The object to subtract from this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing self - f.
        """
        return self._apply_binop(f, operator.sub)

    # ------------------
    #  internal helpers
    # ------------------
    @self_empty()
    def _apply_binop(self, f: Any, op: Callable[..., Any]) -> Any:
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
        if np.isscalar(f):
            chbfn1 = self
            chbfn2 = f * np.ones(self.funs.size)
            simplify = False
        else:
            newdom = self.domain.union(f.domain)
            chbfn1 = self._break(newdom)
            chbfn2 = f._break(newdom)
            simplify = True
        newfuns = []
        for fun1, fun2 in zip(chbfn1, chbfn2, strict=False):
            newfun = op(fun1, fun2)
            if simplify:
                newfun = newfun.simplify()
            newfuns.append(newfun)
        return self.__class__(newfuns)

    def _break(self, targetdomain: Domain) -> Chebfun:
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
        subintervals = iter(targetdomain.intervals)
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
    def breakpoints(self) -> np.ndarray:
        """Get the breakpoints of this Chebfun.

        Breakpoints are the points where the Chebfun transitions from one piece to another.

        Returns:
            numpy.ndarray: Array of breakpoints.
        """
        return np.array(list(self.breakdata.keys()))

    @property
    @self_empty(Domain([]))
    def domain(self) -> Domain:
        """Get the domain of this Chebfun.

        Returns:
            Domain: A Domain object corresponding to this Chebfun.
        """
        return Domain.from_chebfun(self)

    @domain.setter
    def domain(self, new_domain: Any) -> None:
        """Set the domain of the Chebfun by restricting to the new domain.

        Args:
            new_domain (array-like): The new domain to which this Chebfun should be restricted.
        """
        self.restrict_(new_domain)

    @property
    @self_empty(Domain([]))
    def support(self) -> Any:
        """Get the support interval of this Chebfun.

        The support is the interval between the first and last breakpoints.

        Returns:
            numpy.ndarray: Array containing the first and last breakpoints.
        """
        return self.domain.support

    @property
    @self_empty(0.0)
    def hscale(self) -> float:
        """Get the horizontal scale of this Chebfun.

        The horizontal scale is the maximum absolute value of the support interval.

        Returns:
            float: The horizontal scale.
        """
        return float(np.abs(self.support).max())

    @property
    @self_empty(False)
    def iscomplex(self) -> bool:
        """Check if this Chebfun has complex values.

        Returns:
            bool: True if any of the functions in this Chebfun have complex values,
                False otherwise.
        """
        return any(fun.iscomplex for fun in self)

    @property
    @self_empty(False)
    def isconst(self) -> bool:
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
    def isempty(self) -> bool:
        """Check if this Chebfun is empty.

        An empty Chebfun contains no functions.

        Returns:
            bool: True if this Chebfun is empty, False otherwise.
        """
        return self.funs.size == 0

    @property
    @self_empty(0.0)
    def vscale(self) -> Any:
        """Get the vertical scale of this Chebfun.

        The vertical scale is the maximum of the vertical scales of all pieces.

        Returns:
            float: The vertical scale.
        """
        return np.max([fun.vscale for fun in self])

    @property
    @self_empty()
    def x(self) -> Chebfun:
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

    def imag(self) -> Chebfun:
        """Get the imaginary part of this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing the imaginary part of this Chebfun.
                If this Chebfun is real-valued, returns a zero Chebfun.
        """
        if self.iscomplex:
            return self.__class__([fun.imag() for fun in self])
        else:
            return self.initconst(0, domain=self.domain)

    def real(self) -> Chebfun:
        """Get the real part of this Chebfun.

        Returns:
            Chebfun: A new Chebfun representing the real part of this Chebfun.
                If this Chebfun is already real-valued, returns this Chebfun.
        """
        if self.iscomplex:
            return self.__class__([fun.real() for fun in self])
        else:
            return self

    def copy(self) -> Chebfun:
        """Create a deep copy of this Chebfun.

        Returns:
            Chebfun: A new Chebfun that is a deep copy of this Chebfun.
        """
        return self.__class__([fun.copy() for fun in self])

    @self_empty()
    def _restrict(self, subinterval: Any) -> Chebfun:
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

    def restrict(self, subinterval: Any) -> Any:
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
    def restrict_(self, subinterval: Any) -> Chebfun:
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
    def roots(self, merge: Any = None) -> np.ndarray:
        """Compute the roots of a Chebfun.

        This method finds the values x for which f(x) = 0, by computing the roots
        of each piece of the Chebfun and combining them.

        Args:
            merge (bool, optional): Whether to merge roots at breakpoints. If None,
                uses the value from preferences. Defaults to None.

        Returns:
            numpy.ndarray: Array of roots sorted in ascending order.

        Examples:
            >>> import numpy as np
            >>> f = Chebfun.initfun_adaptive(lambda x: x**2 - 1, [-2, 2])
            >>> roots = f.roots()
            >>> len(roots)
            2
            >>> np.allclose(sorted(roots), [-1, 1])
            True
        """
        merge = merge if merge is not None else prefs.mergeroots
        allrts = []
        prvrts = np.array([])
        htol = 1e2 * self.hscale * prefs.eps
        for fun in self:
            rts = fun.roots()
            # ignore first root if equal to the last root of previous fun
            # TODO: there could be multiple roots at breakpoints
            if prvrts.size > 0 and rts.size > 0 and merge and abs(prvrts[-1] - rts[0]) <= htol:
                rts = rts[1:]
            allrts.append(rts)
            prvrts = rts
        return np.concatenate(list(allrts))

    @self_empty()
    def simplify(self) -> Chebfun:
        """Simplify each fun in the chebfun."""
        return self.__class__([fun.simplify() for fun in self])

    def translate(self, c: Any) -> Chebfun:
        """Translate a chebfun by c, i.e., return f(x-c)."""
        return self.__class__([x.translate(c) for x in self])

    # ----------
    #  calculus
    # ----------
    def cumsum(self) -> Chebfun:
        """Compute the indefinite integral (antiderivative) of the Chebfun.

        This method computes the indefinite integral of the Chebfun, with the
        constant of integration chosen so that the indefinite integral evaluates
        to 0 at the left endpoint of the domain. For piecewise functions, constants
        are added to ensure continuity across the pieces.

        Returns:
            Chebfun: A new Chebfun representing the indefinite integral of this Chebfun.

        Examples:
            >>> import numpy as np
            >>> f = Chebfun.initconst(1.0, [-1, 1])
            >>> F = f.cumsum()
            >>> bool(abs(F(-1)) < 1e-10)
            True
            >>> bool(abs(F(1) - 2.0) < 1e-10)
            True
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

    def diff(self, n: int = 1) -> Chebfun:
        """Compute the derivative of the Chebfun.

        This method calculates the nth derivative of the Chebfun with respect to x.
        It creates a new Chebfun where each piece is the derivative of the
        corresponding piece in the original Chebfun.

        Args:
            n: Order of differentiation (default: 1). Must be non-negative integer.

        Returns:
            Chebfun: A new Chebfun representing the nth derivative of this Chebfun.

        Examples:
            >>> from chebpy import chebfun
            >>> f = chebfun(lambda x: x**3)
            >>> df1 = f.diff()    # first derivative: 3*x**2
            >>> df2 = f.diff(2)   # second derivative: 6*x
            >>> df3 = f.diff(3)   # third derivative: 6
            >>> bool(abs(df1(0.5) - 0.75) < 1e-10)
            True
            >>> bool(abs(df2(0.5) - 3.0) < 1e-10)
            True
            >>> bool(abs(df3(0.5) - 6.0) < 1e-10)
            True
        """
        if not isinstance(n, int):
            raise TypeError(n)
        if n == 0:
            return self
        if n < 0:
            raise ValueError(n)

        result = self
        for _ in range(n):
            dfuns = np.array([fun.diff() for fun in result])
            result = self.__class__(dfuns)
        return result

    def conv(self, g: Chebfun) -> Chebfun:
        """Compute the convolution of this Chebfun with g.

        Computes h(x) = (f ★ g)(x) = ∫ f(t) g(x-t) dt, where domain(f) is
        [a, b] and domain(g) is [c, d].  The result is a piecewise Chebfun on
        [a + c, b + d] whose breakpoints are the pairwise sums of the
        breakpoints of f and g.

        Both f and g may be piecewise (contain an arbitrary number of funs).

        When both inputs are single-piece with equal-width domains, the fast
        Hale-Townsend Legendre convolution algorithm is used.  Otherwise, each
        output sub-interval is constructed adaptively using Gauss-Legendre
        quadrature.

        The algorithm is based on:
            N. Hale and A. Townsend, "An algorithm for the convolution of
            Legendre series", SIAM J. Sci. Comput., 36(3), A1207-A1220, 2014.

        Args:
            g (Chebfun): A Chebfun (single-piece or piecewise).

        Returns:
            Chebfun: A piecewise Chebfun on [a + c, b + d] representing
                (f ★ g).

        Examples:
            >>> import numpy as np
            >>> from chebpy import chebfun
            >>> f = chebfun(lambda x: np.ones_like(x), [-1, 1])
            >>> h = f.conv(f)
            >>> bool(abs(h(0.0) - 2.0) < 1e-10)
            True
            >>> bool(abs(h(-1.0) - 1.0) < 1e-10)
            True
            >>> bool(abs(h(1.0) - 1.0) < 1e-10)
            True
        """
        if self.isempty or g.isempty:
            return self.__class__.initempty()

        # Fast path: both single-piece with equal-width domains
        if self.funs.size == 1 and g.funs.size == 1:
            f_fun, g_fun = self.funs[0], g.funs[0]
            f_w = float(f_fun.support[1]) - float(f_fun.support[0])
            g_w = float(g_fun.support[1]) - float(g_fun.support[0])
            if np.isclose(f_w, g_w):
                return self._conv_equal_width_pair(f_fun, g_fun)

        # General piecewise convolution
        return self._conv_piecewise(g)

    def _conv_equal_width_pair(self, f_fun: Any, g_fun: Any) -> Chebfun:
        """Convolve two single Bndfuns of equal width using the fast algorithm.

        Uses the Hale-Townsend Legendre convolution.  The two funs may be on
        different intervals as long as they have the same width.
        """
        a = float(f_fun.support[0])
        b = float(f_fun.support[1])
        c = float(g_fun.support[0])
        d = float(g_fun.support[1])

        h = (b - a) / 2.0  # half-width (same for both funs)

        leg_f = cheb2leg(f_fun.coeffs)
        leg_g = cheb2leg(g_fun.coeffs)

        gamma_left, gamma_right = _conv_legendre(leg_f, leg_g)

        gamma_left = h * gamma_left
        gamma_right = h * gamma_right

        cheb_left = leg2cheb(gamma_left)
        cheb_right = leg2cheb(gamma_right)

        mid = (a + b + c + d) / 2.0
        left_interval = Interval(a + c, mid)
        right_interval = Interval(mid, b + d)

        left_fun = Bndfun(Chebtech(cheb_left), left_interval)
        right_fun = Bndfun(Chebtech(cheb_right), right_interval)

        return self.__class__([left_fun, right_fun])

    def _conv_piecewise(self, g: Chebfun) -> Chebfun:
        """General piecewise convolution via Gauss-Legendre quadrature.

        The breakpoints of the result are the sorted, unique pairwise sums of
        the breakpoints of self and g.  On each sub-interval the convolution
        integral is smooth, so we construct it adaptively.
        """
        f_breaks = self.breakpoints
        g_breaks = g.breakpoints
        f_a, f_b = float(f_breaks[0]), float(f_breaks[-1])
        g_c, g_d = float(g_breaks[0]), float(g_breaks[-1])

        # Output breakpoints: all pairwise sums, uniquified and coalesced
        out_breaks = np.unique(np.add.outer(f_breaks, g_breaks).ravel())
        hscl = max(abs(out_breaks[0]), abs(out_breaks[-1]), 1.0)
        tol = 10.0 * np.finfo(float).eps * hscl
        mask = np.concatenate(([True], np.diff(out_breaks) > tol))
        out_breaks = out_breaks[mask]

        # Quadrature order: sufficient for exact integration of polynomial
        # integrand on each smooth sub-interval
        max_deg = max(fun.size for fun in self.funs) + max(fun.size for fun in g.funs)
        n_quad = max(int(np.ceil((max_deg + 1) / 2)), 16)
        quad_nodes, quad_weights = np.polynomial.legendre.leggauss(n_quad)

        # Pre-convert breakpoints to plain float lists for the inner loop
        f_bps = [float(bp) for bp in f_breaks]
        g_bps = [float(bp) for bp in g_breaks]

        def conv_eval(x: np.ndarray) -> np.ndarray:
            """Evaluate (self ★ g)(x) via Gauss-Legendre quadrature."""
            x = np.atleast_1d(np.asarray(x, dtype=float))
            result = np.zeros(x.shape)
            for idx in range(x.size):
                xi = x[idx]
                t_lo = max(f_a, xi - g_d)
                t_hi = min(f_b, xi - g_c)
                if t_hi <= t_lo:
                    continue
                # Break integration at breakpoints of f and shifted breakpoints
                # of g so the integrand is polynomial on each sub-interval.
                inner = [t_lo, t_hi]
                for bp in f_bps:
                    if t_lo < bp < t_hi:
                        inner.append(bp)
                for bp in g_bps:
                    shifted = xi - bp
                    if t_lo < shifted < t_hi:
                        inner.append(shifted)
                inner = sorted(set(inner))

                total = 0.0
                for j in range(len(inner) - 1):
                    a_int, b_int = inner[j], inner[j + 1]
                    hw = (b_int - a_int) / 2.0
                    mid = (a_int + b_int) / 2.0
                    nodes = hw * quad_nodes + mid
                    wts = hw * quad_weights
                    total += np.dot(wts, self(nodes) * g(xi - nodes))
                result[idx] = total
            return result

        # Build a Bndfun on each output sub-interval
        funs_list = []
        for i in range(len(out_breaks) - 1):
            interval = Interval(out_breaks[i], out_breaks[i + 1])
            fun = Bndfun.initfun_adaptive(conv_eval, interval)
            funs_list.append(fun)

        return self.__class__(funs_list)

    def sum(self) -> Any:
        """Compute the definite integral of the Chebfun over its domain.

        This method calculates the definite integral of the Chebfun over its
        entire domain of definition by summing the definite integrals of each
        piece.

        Returns:
            float or complex: The definite integral of the Chebfun over its domain.

        Examples:
            >>> import numpy as np
            >>> f = Chebfun.initfun_adaptive(lambda x: x**2, [-1, 1])
            >>> bool(abs(f.sum() - 2.0/3.0) < 1e-10)
            True
            >>> g = Chebfun.initconst(1.0, [-1, 1])
            >>> bool(abs(g.sum() - 2.0) < 1e-10)
            True
        """
        return np.sum([fun.sum() for fun in self])

    def dot(self, f: Any) -> Any:
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

    def norm(self, p: Any = 2) -> Any:
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
            raise ValueError(p)

    # ----------
    #  utilities
    # ----------
    @self_empty()
    def absolute(self) -> Chebfun:
        """Absolute value of a Chebfun."""
        newdom = self.domain.merge(self.roots())
        funs = [x.absolute() for x in self._break(newdom)]
        return self.__class__(funs)

    abs = absolute

    @self_empty()
    def sign(self) -> Chebfun:
        """Sign function of a Chebfun.

        Computes the piecewise sign of a Chebfun by finding its roots
        and splitting the domain at those points, then creating constant
        pieces with the appropriate sign values.

        Returns:
            Chebfun: A new Chebfun representing sign(f(x)).
        """
        roots = self.roots()
        newdom = self.domain.merge(roots)
        funs = []
        for fun in self._break(newdom):
            mid = fun.support[0] + 0.5 * (fun.support[-1] - fun.support[0])
            s = float(np.sign(float(self(mid))))
            funs.append(Bndfun.initconst(s, fun.interval))
        result = self.__class__(funs)
        # Set breakdata: at roots sign is 0, elsewhere use sign of function
        htol = max(1e2 * self.hscale * prefs.eps, prefs.eps)
        for bp in result.breakpoints:
            if roots.size > 0 and np.any(np.abs(bp - roots) <= htol):
                result.breakdata[bp] = 0.0
            else:
                result.breakdata[bp] = float(np.sign(float(self(bp))))
        return result

    @self_empty()
    def ceil(self) -> Chebfun:
        """Ceiling function of a Chebfun.

        Computes the piecewise ceiling of a Chebfun by finding where
        the function crosses integer values and splitting the domain
        at those points, then creating constant pieces with the
        appropriate ceiling values.

        Returns:
            Chebfun: A new Chebfun representing ceil(f(x)).
        """
        crossings = self._integer_crossings()
        newdom = self.domain.merge(crossings)
        funs = []
        for fun in self._break(newdom):
            mid = fun.support[0] + 0.5 * (fun.support[-1] - fun.support[0])
            c = float(np.ceil(float(self(mid))))
            funs.append(Bndfun.initconst(c, fun.interval))
        result = self.__class__(funs)
        for bp in result.breakpoints:
            result.breakdata[bp] = float(np.ceil(float(self(bp))))
        return result

    @self_empty()
    def floor(self) -> Chebfun:
        """Floor function of a Chebfun.

        Computes the piecewise floor of a Chebfun by finding where
        the function crosses integer values and splitting the domain
        at those points, then creating constant pieces with the
        appropriate floor values.

        Returns:
            Chebfun: A new Chebfun representing floor(f(x)).
        """
        crossings = self._integer_crossings()
        newdom = self.domain.merge(crossings)
        funs = []
        for fun in self._break(newdom):
            mid = fun.support[0] + 0.5 * (fun.support[-1] - fun.support[0])
            c = float(np.floor(float(self(mid))))
            funs.append(Bndfun.initconst(c, fun.interval))
        result = self.__class__(funs)
        for bp in result.breakpoints:
            result.breakdata[bp] = float(np.floor(float(self(bp))))
        return result

    def _integer_crossings(self) -> np.ndarray:
        """Find where this Chebfun crosses integer values.

        This helper method identifies all points in the domain where the
        Chebfun value equals an integer, by finding roots of (self - n)
        for each integer n in the range of the function.

        Returns:
            numpy.ndarray: Array of x-values where the function crosses integers.
        """
        all_values = np.concatenate([fun.values() for fun in self])
        lo = int(np.floor(np.min(all_values)))
        hi = int(np.ceil(np.max(all_values)))
        crossings = []
        for n in range(lo, hi + 1):
            shifted = self - n
            crossings.extend(shifted.roots().tolist())
        return np.array(crossings)

    @self_empty()
    @cast_arg_to_chebfun
    def maximum(self, other: Any) -> Any:
        """Pointwise maximum of self and another chebfun."""
        return self._maximum_minimum(other, operator.ge)

    @self_empty()
    @cast_arg_to_chebfun
    def minimum(self, other: Any) -> Any:
        """Pointwise mimimum of self and another chebfun."""
        return self._maximum_minimum(other, operator.lt)

    def _maximum_minimum(self, other: Chebfun, comparator: Callable[..., bool]) -> Any:
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
        for interval, use_self in zip(switch.intervals, keys, strict=False):
            subdom = newdom.restrict(interval)
            subfun = self.restrict(subdom) if use_self else other.restrict(subdom)
            funs = np.append(funs, subfun.funs)
        return self.__class__(funs)

    # ----------
    #  plotting
    # ----------
    def plot(self, ax: Axes | None = None, **kwds: Any) -> Any:
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

    def plotcoeffs(self, ax: Axes | None = None, **kwds: Any) -> Axes:
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


# ---------
#  ufuncs
# ---------
def add_ufunc(op: Callable[..., Any]) -> None:
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
    def method(self: Chebfun) -> Chebfun:
        """Apply a NumPy universal function to this Chebfun.

        This method applies a NumPy universal function (ufunc) to each piece
        of this Chebfun and returns a new Chebfun representing the result.

        Args:
            self (Chebfun): The Chebfun object to which the function is applied.

        Returns:
            Chebfun: A new Chebfun representing op(f(x)).
        """
        return self.__class__([op(fun) for fun in self])

    name = op.__name__  # type: ignore[attr-defined]
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
