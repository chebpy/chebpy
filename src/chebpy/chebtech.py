"""Implementation of Chebyshev polynomial technology for function approximation.

This module provides the Chebtech class, which is an abstract base class for
representing functions using Chebyshev polynomial expansions. It serves as the
foundation for the Chebtech class, which uses Chebyshev points of the second kind.

The Chebtech classes implement core functionality for working with Chebyshev
expansions, including:
- Function evaluation using Clenshaw's algorithm or barycentric interpolation
- Algebraic operations (addition, multiplication, etc.)
- Calculus operations (differentiation, integration, etc.)
- Rootfinding
- Plotting

These classes are primarily used internally by higher-level classes like Bndfun
and Chebfun, rather than being used directly by end users.
"""

from abc import ABC

import matplotlib.pyplot as plt
import numpy as np

from .algorithms import (
    adaptive,
    bary,
    barywts2,
    chebpts2,
    clenshaw,
    coeffmult,
    coeffs2vals2,
    newtonroots,
    rootsunit,
    standard_chop,
    vals2coeffs2,
)
from .decorators import self_empty
from .plotting import plotfun, plotfuncoeffs
from .settings import _preferences as prefs
from .smoothfun import Smoothfun
from .utilities import Interval, coerce_list


class Chebtech(Smoothfun, ABC):
    """Abstract base class serving as the template for Chebtech1 and Chebtech subclasses.

    Chebtech objects always work with first-kind coefficients, so much
    of the core operational functionality is defined at this level.

    The user will rarely work with these classes directly so we make
    several assumptions regarding input data types.
    """

    @classmethod
    def initconst(cls, c, *, interval=None):
        """Initialise a Chebtech from a constant c."""
        if not np.isscalar(c):
            raise ValueError(c)
        if isinstance(c, int):
            c = float(c)
        return cls(np.array([c]), interval=interval)

    @classmethod
    def initempty(cls, *, interval=None):
        """Initialise an empty Chebtech."""
        return cls(np.array([]), interval=interval)

    @classmethod
    def initidentity(cls, *, interval=None):
        """Chebtech representation of f(x) = x on [-1,1]."""
        return cls(np.array([0, 1]), interval=interval)

    @classmethod
    def initfun(cls, fun, n=None, *, interval=None):
        """Convenience constructor to automatically select the adaptive or fixedlen constructor.

        This constructor automatically selects between the adaptive or fixed-length
        constructor based on the input arguments passed.
        """
        if n is None:
            return cls.initfun_adaptive(fun, interval=interval)
        else:
            return cls.initfun_fixedlen(fun, n, interval=interval)

    @classmethod
    def initfun_fixedlen(cls, fun, n, *, interval=None):
        """Initialise a Chebtech from the callable fun using n degrees of freedom.

        This constructor creates a Chebtech representation of the function using
        a fixed number of degrees of freedom specified by n.
        """
        points = cls._chebpts(n)
        values = fun(points)
        coeffs = vals2coeffs2(values)
        return cls(coeffs, interval=interval)

    @classmethod
    def initfun_adaptive(cls, fun, *, interval=None, min_samples=None, maxpow2=None):
        """Initialise a Chebtech from the callable fun utilising the adaptive constructor.

        This constructor uses an adaptive algorithm to determine the appropriate
        number of degrees of freedom needed to represent the function.

        Args:
            fun: Callable function to approximate
            interval: Domain interval (defaults to prefs.domain)
            min_samples: Minimum number of sample points (for composition operations)
            maxpow2: Maximum power of 2 for adaptive refinement (for composition operations)
        """
        interval = interval if interval is not None else prefs.domain
        interval = Interval(*interval)
        coeffs = adaptive(cls, fun, hscale=interval.hscale, min_samples=min_samples, maxpow2=maxpow2)
        return cls(coeffs, interval=interval)

    @classmethod
    def initvalues(cls, values, *, interval=None):
        """Initialise a Chebtech from an array of values at Chebyshev points."""
        return cls(cls._vals2coeffs(values), interval=interval)

    def __init__(self, coeffs, interval=None):
        """Initialize a Chebtech object.

        This method initializes a new Chebtech object with the given coefficients
        and interval. If no interval is provided, the default interval from
        preferences is used.

        Args:
            coeffs (array-like): The coefficients of the Chebyshev series.
            interval (array-like, optional): The interval on which the function
                is defined. Defaults to None, which uses the default interval
        """
        interval = interval if interval is not None else prefs.domain
        self._coeffs = np.array(coeffs)
        self._interval = Interval(*interval)

    def __call__(self, x, how="clenshaw"):
        """Evaluate the Chebtech at the given points.

        Args:
            x: Points at which to evaluate the Chebtech.
            how (str, optional): Method to use for evaluation. Either "clenshaw" or "bary".
                Defaults to "clenshaw".

        Returns:
            The values of the Chebtech at the given points.

        Raises:
            ValueError: If the specified method is not supported.
        """
        method = {
            "clenshaw": self.__call__clenshaw,
            "bary": self.__call__bary,
        }
        try:
            return method[how](x)
        except KeyError:
            raise ValueError(how)

    def __call__clenshaw(self, x):
        return clenshaw(x, self.coeffs)

    def __call__bary(self, x):
        fk = self.values()
        xk = self._chebpts(fk.size)
        vk = self._barywts(fk.size)
        return bary(x, fk, xk, vk)

    def __repr__(self):  # pragma: no cover
        """Return a string representation of the Chebtech.

        Returns:
            str: A string representation of the Chebtech.
        """
        out = f"<{self.__class__.__name__}{{{self.size}}}>"
        return out

    # ------------
    #  properties
    # ------------
    @property
    def coeffs(self):
        """Chebyshev expansion coefficients in the T_k basis."""
        return self._coeffs

    @property
    def interval(self):
        """Interval that Chebtech is mapped to."""
        return self._interval

    @property
    def size(self):
        """Return the size of the object."""
        return self.coeffs.size

    @property
    def isempty(self):
        """Return True if the Chebtech is empty."""
        return self.size == 0

    @property
    def iscomplex(self):
        """Determine whether the underlying onefun is complex or real valued."""
        return self._coeffs.dtype == complex

    @property
    def isconst(self):
        """Return True if the Chebtech represents a constant."""
        return self.size == 1

    @property
    @self_empty(0.0)
    def vscale(self):
        """Estimate the vertical scale of a Chebtech."""
        return np.abs(coerce_list(self.values())).max()

    # -----------
    #  utilities
    # -----------
    def copy(self):
        """Return a deep copy of the Chebtech."""
        return self.__class__(self.coeffs.copy(), interval=self.interval.copy())

    def imag(self):
        """Return the imaginary part of the Chebtech.

        Returns:
            Chebtech: A new Chebtech representing the imaginary part of this Chebtech.
                If this Chebtech is real-valued, returns a zero Chebtech.
        """
        if self.iscomplex:
            return self.__class__(np.imag(self.coeffs), self.interval)
        else:
            return self.initconst(0, interval=self.interval)

    def prolong(self, n):
        """Return a Chebtech of length n.

        Obtained either by truncating if n < self.size or zero-padding if n > self.size.
        In all cases a deep copy is returned.
        """
        m = self.size
        ak = self.coeffs
        cls = self.__class__
        if n - m < 0:
            out = cls(ak[:n].copy(), interval=self.interval)
        elif n - m > 0:
            out = cls(np.append(ak, np.zeros(n - m)), interval=self.interval)
        else:
            out = self.copy()
        return out

    def real(self):
        """Return the real part of the Chebtech.

        Returns:
            Chebtech: A new Chebtech representing the real part of this Chebtech.
                If this Chebtech is already real-valued, returns self.
        """
        if self.iscomplex:
            return self.__class__(np.real(self.coeffs), self.interval)
        else:
            return self

    def simplify(self):
        """Call standard_chop on the coefficients of self.

        Returns a Chebtech comprised of a copy of the truncated coefficients.
        """
        # coefficients
        oldlen = len(self.coeffs)
        longself = self.prolong(max(17, oldlen))
        cfs = longself.coeffs
        # scale (decrease) tolerance by hscale
        tol = prefs.eps * max(self.interval.hscale, 1)
        # chop
        npts = standard_chop(cfs, tol=tol)
        npts = min(oldlen, npts)
        # construct
        return self.__class__(cfs[:npts].copy(), interval=self.interval)

    def values(self):
        """Function values at Chebyshev points."""
        return coeffs2vals2(self.coeffs)

    # ---------
    #  algebra
    # ---------
    @self_empty()
    def __add__(self, f):
        """Add a scalar or another Chebtech to this Chebtech.

        Args:
            f: A scalar or another Chebtech to add to this Chebtech.

        Returns:
            Chebtech: A new Chebtech representing the sum.
        """
        cls = self.__class__
        if np.isscalar(f):
            if np.iscomplexobj(f):
                dtype = complex
            else:
                dtype = self.coeffs.dtype
            cfs = np.array(self.coeffs, dtype=dtype)
            cfs[0] += f
            return cls(cfs, interval=self.interval)
        else:
            # TODO: is a more general decorator approach better here?
            # TODO: for constant Chebtech, convert to constant and call __add__ again
            if f.isempty:
                return f.copy()
            g = self
            n, m = g.size, f.size
            if n < m:
                g = g.prolong(m)
            elif m < n:
                f = f.prolong(n)
            cfs = f.coeffs + g.coeffs

            # check for zero output
            eps = prefs.eps
            tol = 0.5 * eps * max([f.vscale, g.vscale])
            if all(abs(cfs) < tol):
                return cls.initconst(0.0, interval=self.interval)
            else:
                return cls(cfs, interval=self.interval)

    @self_empty()
    def __div__(self, f):
        """Divide this Chebtech by a scalar or another Chebtech.

        Args:
            f: A scalar or another Chebtech to divide this Chebtech by.

        Returns:
            Chebtech: A new Chebtech representing the quotient.
        """
        cls = self.__class__
        if np.isscalar(f):
            cfs = 1.0 / f * self.coeffs
            return cls(cfs, interval=self.interval)
        else:
            # TODO: review with reference to __add__
            if f.isempty:
                return f.copy()
            return cls.initfun_adaptive(lambda x: self(x) / f(x), interval=self.interval)

    __truediv__ = __div__

    @self_empty()
    def __mul__(self, g):
        """Multiply this Chebtech by a scalar or another Chebtech.

        Args:
            g: A scalar or another Chebtech to multiply this Chebtech by.

        Returns:
            Chebtech: A new Chebtech representing the product.
        """
        cls = self.__class__
        if np.isscalar(g):
            cfs = g * self.coeffs
            return cls(cfs, interval=self.interval)
        else:
            # TODO: review with reference to __add__
            if g.isempty:
                return g.copy()
            f = self
            n = f.size + g.size - 1
            f = f.prolong(n)
            g = g.prolong(n)
            cfs = coeffmult(f.coeffs, g.coeffs)
            out = cls(cfs, interval=self.interval)
            return out

    def __neg__(self):
        """Return the negative of this Chebtech.

        Returns:
            Chebtech: A new Chebtech representing the negative of this Chebtech.
        """
        coeffs = -self.coeffs
        return self.__class__(coeffs, interval=self.interval)

    def __pos__(self):
        """Return this Chebtech (unary positive).

        Returns:
            Chebtech: This Chebtech (self).
        """
        return self

    @self_empty()
    def __pow__(self, f):
        """Raise this Chebtech to a power.

        Args:
            f: The exponent, which can be a scalar or another Chebtech.

        Returns:
            Chebtech: A new Chebtech representing this Chebtech raised to the power f.
        """

        def powfun(fn, x):
            if np.isscalar(fn):
                return fn
            else:
                return fn(x)

        min_samples = len(self.coeffs)
        if not np.isscalar(f) and hasattr(f, "coeffs"):
            min_samples = max(min_samples, len(f.coeffs))

        return self.__class__.initfun_adaptive(
            lambda x: np.power(self(x), powfun(f, x)), interval=self.interval, min_samples=min_samples
        )

    def __rdiv__(self, f):
        """Divide a scalar by this Chebtech.

        This is called when f / self is executed and f is not a Chebtech.

        Args:
            f: A scalar to be divided by this Chebtech.

        Returns:
            Chebtech: A new Chebtech representing f divided by this Chebtech.
        """

        # Executed when __div__(f, self) fails, which is to say whenever f
        # is not a Chebtech. We proceed on the assumption f is a scalar.
        def constfun(x):
            return 0.0 * x + f

        return self.__class__.initfun_adaptive(lambda x: constfun(x) / self(x), interval=self.interval)

    __radd__ = __add__

    def __rsub__(self, f):
        """Subtract this Chebtech from a scalar.

        This is called when f - self is executed and f is not a Chebtech.

        Args:
            f: A scalar from which to subtract this Chebtech.

        Returns:
            Chebtech: A new Chebtech representing f minus this Chebtech.
        """
        return -(self - f)

    @self_empty()
    def __rpow__(self, f):
        """Raise a scalar to the power of this Chebtech.

        This is called when f ** self is executed and f is not a Chebtech.

        Args:
            f: A scalar to be raised to the power of this Chebtech.

        Returns:
            Chebtech: A new Chebtech representing f raised to the power of this Chebtech.
        """
        return self.__class__.initfun_adaptive(lambda x: np.power(f, self(x)), interval=self.interval)

    __rtruediv__ = __rdiv__
    __rmul__ = __mul__

    def __sub__(self, f):
        """Subtract a scalar or another Chebtech from this Chebtech.

        Args:
            f: A scalar or another Chebtech to subtract from this Chebtech.

        Returns:
            Chebtech: A new Chebtech representing the difference.
        """
        return self + (-f)

    # -------
    #  roots
    # -------
    def roots(self, sort=None):
        """Compute the roots of the Chebtech on [-1,1].

        Uses the coefficients in the associated Chebyshev series approximation.
        """
        sort = sort if sort is not None else prefs.sortroots
        rts = rootsunit(self.coeffs)
        rts = newtonroots(self, rts)
        # fix problems with newton for roots that are numerically very close
        rts = np.clip(rts, -1, 1)  # if newton roots are just outside [-1,1]
        rts = rts if not sort else np.sort(rts)
        return rts

    # ----------
    #  calculus
    # ----------
    # Note that function returns 0 for an empty Chebtech object; this is
    # consistent with numpy, which returns zero for the sum of an empty array
    @self_empty(resultif=0.0)
    def sum(self):
        """Definite integral of a Chebtech on the interval [-1,1]."""
        if self.isconst:
            out = 2.0 * self(0.0)
        else:
            ak = self.coeffs.copy()
            ak[1::2] = 0
            kk = np.arange(2, ak.size)
            ii = np.append([2, 0], 2 / (1 - kk**2))
            out = (ak * ii).sum()
        return out

    @self_empty()
    def cumsum(self):
        """Return a Chebtech object representing the indefinite integral.

        Computes the indefinite integral of a Chebtech on the interval [-1,1].
        The constant term is chosen such that F(-1) = 0.
        """
        n = self.size
        ak = np.append(self.coeffs, [0, 0])
        bk = np.zeros(n + 1, dtype=self.coeffs.dtype)
        rk = np.arange(2, n + 1)
        bk[2:] = 0.5 * (ak[1:n] - ak[3:]) / rk
        bk[1] = ak[0] - 0.5 * ak[2]
        vk = np.ones(n)
        vk[1::2] = -1
        bk[0] = (vk * bk[1:]).sum()
        out = self.__class__(bk, interval=self.interval)
        return out

    @self_empty()
    def diff(self):
        """Return a Chebtech object representing the derivative.

        Computes the derivative of a Chebtech on the interval [-1,1].
        """
        if self.isconst:
            out = self.__class__(np.array([0.0]), interval=self.interval)
        else:
            n = self.size
            ak = self.coeffs
            zk = np.zeros(n - 1, dtype=self.coeffs.dtype)
            wk = 2 * np.arange(1, n)
            vk = wk * ak[1:]
            zk[-1::-2] = vk[-1::-2].cumsum()
            zk[-2::-2] = vk[-2::-2].cumsum()
            zk[0] = 0.5 * zk[0]
            out = self.__class__(zk, interval=self.interval)
        return out

    @staticmethod
    def _chebpts(n):
        """Return n Chebyshev points of the second-kind."""
        return chebpts2(n)

    @staticmethod
    def _barywts(n):
        """Barycentric weights for Chebyshev points of 2nd kind."""
        return barywts2(n)

    @staticmethod
    def _vals2coeffs(vals):
        """Map function values at Chebyshev points of 2nd kind.

        Converts values at Chebyshev points of 2nd kind to first-kind Chebyshev polynomial coefficients.
        """
        return vals2coeffs2(vals)

    @staticmethod
    def _coeffs2vals(coeffs):
        """Map first-kind Chebyshev polynomial coefficients.

        Converts first-kind Chebyshev polynomial coefficients to function values at Chebyshev points of 2nd kind.
        """
        return coeffs2vals2(coeffs)

    # ----------
    #  plotting
    # ----------
    def plot(self, ax=None, **kwargs):
        """Plot the Chebtech on the interval [-1, 1].

        Args:
            ax (matplotlib.axes.Axes, optional): The axes on which to plot. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the plot function.

        Returns:
            matplotlib.lines.Line2D: The line object created by the plot.
        """
        return plotfun(self, (-1, 1), ax=ax, **kwargs)

    def plotcoeffs(self, ax=None, **kwargs):
        """Plot the absolute values of the Chebyshev coefficients.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes on which to plot. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the plot function.

        Returns:
            matplotlib.lines.Line2D: The line object created by the plot.
        """
        ax = ax or plt.gca()
        return plotfuncoeffs(abs(self.coeffs), ax=ax, **kwargs)
