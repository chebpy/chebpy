"""Immutable representation of Chebyshev polynomials.

This module provides a class for the immutable representation of Chebyshev
polynomials and various factory functions to construct such polynomials.
"""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.chebyshev as cheb
from matplotlib.axes import Axes

from .algorithms import coeffs2vals2, standard_chop, vals2coeffs2
from .settings import _preferences as prefs

# Type aliases
ArrayLike = list[float] | tuple[float, ...] | np.ndarray
DomainLike = tuple[float, float] | list[float] | np.ndarray
ScalarLike = int | float | complex


class ChebyshevPolynomial(cheb.Chebyshev):
    """Immutable representation of a Chebyshev polynomial.

    This class represents a Chebyshev polynomial using its coefficients in the
    Chebyshev basis. The polynomial is defined on a specific domain.

    Attributes:
        coef (np.ndarray): The coefficients of the Chebyshev polynomial.
        domain (np.ndarray): The domain on which the polynomial is defined.
        window (np.ndarray): The window on which the polynomial is mapped. Please use [-1, +1]
        symbol (str): Symbol used to represent the independent variable.
    """

    def __init__(self, coef: ArrayLike, domain: DomainLike | None = None, window=None, symbol: str = "x") -> None:
        """Initialize a ChebyshevPolynomial object.

        Args:
            coef: Chebyshev coefficients in order of increasing degree.
            domain: Domain to use. The interval [domain[0], domain[1]] is mapped
                to the interval [window[0], window[1]] by shifting and scaling.
                If None, the default domain [-1, 1] is used.
            window: Window to use. The interval [domain[0], domain[1]] is mapped
                to the interval [window[0], window[1]] by shifting and scaling.
                If None, the default window [-1, 1] is used.
            symbol: Symbol used to represent the independent variable in string
                representations of the polynomial expression. Default is 'x'.
        """
        if window is None:
            window = np.array([-1, 1])

        super().__init__(coef, domain, window=window, symbol=symbol)

    def copy(self) -> "ChebyshevPolynomial":
        """Create a copy of the ChebyshevPolynomial object.

        Returns:
            A new ChebyshevPolynomial object with the same attributes.
        """
        return ChebyshevPolynomial(coef=self.coef.copy(), domain=self.domain.copy(), symbol=self.symbol)

    def real(self) -> "ChebyshevPolynomial":
        """Return the real part of the polynomial.

        Returns:
            ChebyshevPolynomial: A new polynomial with the real part of the coefficients
                if the polynomial is complex, otherwise the original polynomial.
        """
        if self.iscomplex:
            return ChebyshevPolynomial(coef=np.real(self.coef), domain=self.domain, symbol=f"{self.symbol}")
        else:
            return self

    def imag(self) -> "ChebyshevPolynomial":
        """Return the imaginary part of the polynomial.

        Returns:
            ChebyshevPolynomial: A new polynomial with the imaginary part of the coefficients
                if the polynomial is complex, otherwise the original polynomial.
        """
        if self.iscomplex:
            return ChebyshevPolynomial(coef=np.imag(self.coef), domain=self.domain, symbol=f"{self.symbol}")
        else:
            return self

    def __call__(self, arg: ScalarLike | ArrayLike) -> ScalarLike | np.ndarray:
        """Evaluate the polynomial at points x.

        Args:
            arg: Points at which to evaluate the polynomial. Can be a scalar or array-like.

        Returns:
            If arg is a scalar, returns a scalar value.
            If arg is an array, returns an array of values.
        """
        # If the input is a scalar, directly evaluate the polynomial
        if np.isscalar(arg):
            # Map the input to the window
            mapped_arg = np.asarray(arg)
            mapped_arg = (mapped_arg - self.domain[0]) / (self.domain[1] - self.domain[0]) * (
                self.window[1] - self.window[0]
            ) + self.window[0]

            # Evaluate the polynomial using the chebval function
            return cheb.chebval(mapped_arg, self.coef)

        # For array inputs, call the parent class's __call__ method
        return super().__call__(arg)

    @property
    def iscomplex(self) -> bool:
        """Determine whether the polynomial has complex coefficients.

        Returns:
            bool: True if the polynomial has complex coefficients, False otherwise.
        """
        return np.iscomplexobj(self.coef)

    @property
    def size(self) -> int:
        """Return the size of the polynomial (number of coefficients).

        Returns:
            int: The number of coefficients in the polynomial.
        """
        return self.coef.size

    @property
    def isempty(self) -> bool:
        """Return True if the polynomial is empty (has no coefficients).

        Returns:
            bool: True if the polynomial has no coefficients, False otherwise.
        """
        return self.size == 0

    @property
    def isconst(self) -> bool:
        """Return True if the polynomial represents a constant (has only one coefficient).

        Returns:
            bool: True if the polynomial has only one coefficient, False otherwise.
        """
        return self.size == 1

    @property
    def vscale(self) -> float:
        """Estimate the vertical scale of the polynomial.

        The vertical scale is the maximum absolute value of the polynomial
        evaluated at Chebyshev points.

        Returns:
            float: The maximum absolute value of the polynomial at Chebyshev points.
        """
        return np.abs(self.values).max()

    def sum(self) -> float:
        """Return the definite integral of the polynomial over its domain [a, b].

        Computes the definite integral of the polynomial over its domain using
        numpy.polynomial.chebyshev tools with correct domain/window logic.

        Returns:
            float: The definite integral of the polynomial over its domain.
        """
        if self.isempty:
            return 0.0

        a, b = self.domain
        ch = ChebyshevPolynomial(self.coef, domain=self.domain)  # window = [-1, 1] by default
        integral = ch.integ()
        return integral(b) - integral(a)

    def plot(self, ax: Axes | None = None, n: int | None = None, **kwds: Any) -> Axes:
        """Plot the Chebyshev polynomial over its domain.

        This method plots the Chebyshev polynomial over its domain using matplotlib.
        For complex-valued polynomials, it plots the real part against the imaginary part.

        Args:
            ax: The axes on which to plot. If None, a new axes will be created.
            n: Number of points to use for plotting. If None, uses the value from preferences.
            **kwds: Additional keyword arguments to pass to matplotlib's plot function.

        Returns:
            The axes on which the plot was created.
        """
        ax = ax or plt.gca()
        n = n if n is not None else prefs.N_plot
        xx = np.linspace(self.domain[0], self.domain[1], n)
        ff = self(xx)
        if self.iscomplex:
            ax.plot(np.real(ff), np.imag(ff), **kwds)
            ax.set_xlabel(kwds.pop("ylabel", "real"))
            ax.set_ylabel(kwds.pop("xlabel", "imag"))
        else:
            ax.plot(xx, ff, **kwds)
        return ax

    def diff(self) -> "ChebyshevPolynomial":
        """Return the derivative as a new ChebyshevPolynomial.

        Computes the first derivative of the polynomial with respect to its variable.

        Returns:
            ChebyshevPolynomial: The derivative of the polynomial.
        """
        # Get the coefficients of the derivative
        deriv_coef = cheb.chebder(self.coef, m=1)
        return ChebyshevPolynomial(coef=deriv_coef, domain=self.domain, symbol=f"{self.symbol}")

    def cumsum(self) -> "ChebyshevPolynomial":
        """Return the antiderivative as a new ChebyshevPolynomial.

        Computes the first antiderivative of the polynomial with respect to its variable.
        The antiderivative is calculated with the lower bound set to the lower bound of the domain
        and the integration constant set to 0.

        Returns:
            ChebyshevPolynomial: The antiderivative of the polynomial.
        """
        # Get the coefficients of the antiderivative
        integ_coef = cheb.chebint(self.coef, m=1, lbnd=self.domain[0], k=0)
        return ChebyshevPolynomial(coef=integ_coef, domain=self.domain, symbol=f"{self.symbol}")

    @property
    def values(self) -> np.ndarray:
        """Get function values at Chebyshev points.

        Computes the values of the polynomial at Chebyshev points of the second kind
        using the coeffs2vals2 algorithm.

        Returns:
            np.ndarray: Function values at Chebyshev points.
        """
        return coeffs2vals2(self.coef)

    def prolong(self, n: int) -> "ChebyshevPolynomial":
        """Return a ChebyshevPolynomial of length n.

        Creates a new ChebyshevPolynomial with a specified number of coefficients.

        Args:
            n: The desired number of coefficients.

        Returns:
            ChebyshevPolynomial: A new polynomial with n coefficients.

        Note:
            If n < self.size, the result is a truncated copy.
            If n > self.size, the result is zero-padded.
            If n == self.size, a copy of the original polynomial is returned.
            In all cases, a deep copy is returned.
        """
        m = self.size
        ak = self.coef

        if n < m:
            new_coeffs = ak[:n].copy()
        elif n > m:
            new_coeffs = np.concatenate([ak, np.zeros(n - m, dtype=ak.dtype)])
        else:
            return self.copy()

        return ChebyshevPolynomial(new_coeffs, domain=self.domain, symbol=self.symbol)


def from_coefficients(
    coef: ArrayLike, domain: DomainLike | None = None, window: DomainLike | None = None, symbol: str = "x"
) -> ChebyshevPolynomial:
    """Create a Chebyshev polynomial from its coefficients.

    Args:
        coef: Chebyshev coefficients in order of increasing degree.
        domain: Domain to use. The interval [domain[0], domain[1]] is mapped
            to the interval [window[0], window[1]] by shifting and scaling.
            If None, the default domain [-1, 1] is used.
        window: Window, see domain for its use. If None, the default window
            [-1, 1] is used.
        symbol: Symbol used to represent the independent variable in string
            representations of the polynomial expression. Default is 'x'.

    Returns:
        A new Chebyshev polynomial with the given coefficients.

    Raises:
        ValueError: If the coefficient array is empty.
    """
    if len(coef) == 0:
        raise ValueError("Empty coefficients")

    return ChebyshevPolynomial(coef, domain, window, symbol)


def from_values(
    values: ArrayLike, domain: DomainLike | None = None, window: DomainLike | None = None, symbol: str = "x"
) -> ChebyshevPolynomial:
    """Create a Chebyshev polynomial from values at Chebyshev points.

    Constructs a Chebyshev polynomial that interpolates the given values at
    Chebyshev points of the second kind.

    Args:
        values: Values at Chebyshev points of the second kind.
        domain: Domain to use. The interval [domain[0], domain[1]] is mapped
            to the interval [window[0], window[1]] by shifting and scaling.
            If None, the default domain [-1, 1] is used.
        window: Window, see domain for its use. If None, the default window
            [-1, 1] is used.
        symbol: Symbol used to represent the independent variable in string
            representations of the polynomial expression. Default is 'x'.

    Returns:
        A new Chebyshev polynomial that interpolates the given values.

    Raises:
        ValueError: If the values array is empty.
    """
    if len(values) == 0:
        raise ValueError("Empty values")

    coef = vals2coeffs2(values)
    return ChebyshevPolynomial(coef, domain, window, symbol)


def from_roots(
    roots: ArrayLike, domain: DomainLike | None = None, window: DomainLike | None = None, symbol: str = "x"
) -> ChebyshevPolynomial:
    """Create a Chebyshev polynomial from its roots.

    Constructs a Chebyshev polynomial that has the specified roots.

    Args:
        roots: Sequence of root values.
        domain: Domain to use. The interval [domain[0], domain[1]] is mapped
            to the interval [window[0], window[1]] by shifting and scaling.
            If None, the default domain [-1, 1] is used.
        window: Window, see domain for its use. If None, the default window
            [-1, 1] is used.
        symbol: Symbol used to represent the independent variable in string
            representations of the polynomial expression. Default is 'x'.

    Returns:
        A new Chebyshev polynomial with the specified roots.

    Raises:
        ValueError: If the roots array is empty.
    """
    if len(roots) == 0:
        raise ValueError("Empty roots")

    coef = cheb.chebfromroots(roots)
    return ChebyshevPolynomial(coef, domain, window, symbol)


def from_constant(
    c: ScalarLike, domain: DomainLike | None = None, window: DomainLike | None = None, symbol: str = "x"
) -> ChebyshevPolynomial:
    """Create a Chebyshev polynomial representing a constant value.

    Constructs a Chebyshev polynomial of degree 0 that represents a constant value.

    Args:
        c: The constant value (must be a scalar).
        domain: Domain to use. The interval [domain[0], domain[1]] is mapped
            to the interval [window[0], window[1]] by shifting and scaling.
            If None, the default domain [-1, 1] is used.
        window: Window, see domain for its use. If None, the default window
            [-1, 1] is used.
        symbol: Symbol used to represent the independent variable in string
            representations of the polynomial expression. Default is 'x'.

    Returns:
        A new Chebyshev polynomial representing the constant value.

    Raises:
        ValueError: If the input is not a scalar value.
    """
    if not np.isscalar(c):
        raise ValueError("Input must be a scalar value")

    # Convert integer to float to match behavior in other parts of the codebase
    if isinstance(c, int):
        c = float(c)

    return ChebyshevPolynomial([c], domain, window, symbol)


def from_function(
    fun: callable,
    domain: DomainLike | None = None,
    window: DomainLike | None = None,
    symbol: str = "x",
    n: int | None = None,
) -> ChebyshevPolynomial:
    """Create a Chebyshev polynomial from a callable function.

    Constructs a Chebyshev polynomial that approximates the given function.
    If n is provided, uses a fixed number of degrees of freedom.
    If n is None, uses an adaptive algorithm to determine the appropriate
    number of degrees of freedom.

    Args:
        fun: Callable function to approximate.
        domain: Domain to use. The interval [domain[0], domain[1]] is mapped
            to the interval [window[0], window[1]] by shifting and scaling.
            If None, the default domain [-1, 1] is used.
        window: Window, see domain for its use. If None, the default window
            [-1, 1] is used.
        symbol: Symbol used to represent the independent variable in string
            representations of the polynomial expression. Default is 'x'.
        n: Number of degrees of freedom to use. If None, uses an adaptive algorithm.

    Returns:
        A new Chebyshev polynomial that approximates the given function.
    """
    from .algorithms import chebpts2, vals2coeffs2

    domain_arr = np.array([-1, 1]) if domain is None else np.array(domain)

    # Create a wrapper function that maps points from [-1, 1] to the custom domain
    def mapped_fun(x):
        # Map x from [-1, 1] to the custom domain
        a, b = domain_arr
        mapped_x = 0.5 * (b - a) * (x + 1) + a
        return fun(mapped_x)

    if n is None:
        # Use adaptive algorithm
        hscale = (domain_arr[1] - domain_arr[0]) / 2
        coeffs = __adaptive(ChebyshevPolynomial, mapped_fun, hscale=hscale)
    else:
        # Use fixed number of degrees of freedom
        points = chebpts2(n)
        values = mapped_fun(points)
        coeffs = vals2coeffs2(values)

    return ChebyshevPolynomial(coeffs, domain, window, symbol)


def __adaptive(cls: type, fun: callable, hscale: float = 1, maxpow2: int = None) -> np.ndarray:
    """Adaptively determine the number of points needed to represent a function.

    This function implements an adaptive algorithm to determine the appropriate
    number of points needed to represent a function to a specified tolerance.
    It cycles over powers of two, evaluating the function at Chebyshev points
    and checking if the resulting coefficients can be truncated.

    Args:
        cls: The class that provides the _chebpts and _vals2coeffs methods.
        fun (callable): The function to be approximated.
        hscale (float, optional): Scale factor for the tolerance. Defaults to 1.
        maxpow2 (int, optional): Maximum power of 2 to try. If None, uses the
            value from preferences.

    Returns:
        numpy.ndarray: Coefficients of the Chebyshev series representing the function.

    Warns:
        UserWarning: If the constructor does not converge within the maximum
            number of iterations.
    """
    minpow2 = 4  # 17 points
    maxpow2 = maxpow2 if maxpow2 is not None else prefs.maxpow2
    for k in range(minpow2, max(minpow2, maxpow2) + 1):
        n = 2**k + 1
        points = cheb.chebpts2(n)
        values = fun(points)
        coeffs = vals2coeffs2(values)
        eps = prefs.eps
        tol = eps * max(hscale, 1)  # scale (decrease) tolerance by hscale
        chplen = standard_chop(coeffs, tol=tol)
        if chplen < coeffs.size:
            coeffs = coeffs[:chplen]
            break
        if k == maxpow2:
            warnings.warn(f"The {cls.__name__} constructor did not converge: using {n} points")
            break
    return coeffs
