"""Immutable representation of Chebyshev polynomials.

This module provides a dataclass for the immutable representation of Chebyshev
polynomials and various build functions to construct such polynomials.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.chebyshev as cheb
#
from .algorithms import vals2coeffs2
from .settings import _preferences as prefs


class ChebyshevPolynomial(cheb.Chebyshev):
    """Immutable representation of a Chebyshev polynomial.

    This class represents a Chebyshev polynomial using its coefficients in the
    Chebyshev basis. The polynomial is defined on a specific interval.

    Attributes:
        coeffs: The coefficients of the Chebyshev polynomial.
        interval: The interval on which the polynomial is defined.
    """

    def __init__(self, coef, domain=None, window=None, symbol="x"):
        """Initialize a ChebyshevPolynomial object."""
        super().__init__(coef, domain, window, symbol)

    def __call__(self, arg):
        """Evaluate the polynomial at points x.

        Args:
            arg: Points at which to evaluate the polynomial.

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
    def iscomplex(self):
        """Determine whether the polynomial has complex coefficients."""
        return np.iscomplexobj(self.coef)

    def plot(self, ax=None, n=None, **kwds):
        """Plot the Chebyshev polynomial over its domain.

        This method plots the Chebyshev polynomial over its domain using matplotlib.
        For complex-valued polynomials, it plots the real part against the imaginary part.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None,
                a new axes will be created. Defaults to None.
            n (int, optional): Number of points to use for plotting. If None, uses
                the value from preferences. Defaults to None.
            **kwds: Additional keyword arguments to pass to matplotlib's plot function.

        Returns:
            matplotlib.axes.Axes: The axes on which the plot was created.
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

    def diff(self):
        """Return the derivative as a new ChebyshevPolynomial.

        Returns:
            ChebyshevPolynomial: The derivative of the polynomial.
        """
        # Get the coefficients of the derivative
        deriv_coef = cheb.chebder(self.coef, m=1)
        return ChebyshevPolynomial(coef=deriv_coef, domain=self.domain, window=self.window, symbol=f"{self.symbol}")

    def cumsum(self):
        """Return the antiderivative as a new ChebyshevPolynomial.

        The antiderivative is calculated with the lower bound set to the lower bound of the domain
        and the integration constant set to 0.

        Returns:
            ChebyshevPolynomial: The antiderivative of the polynomial.
        """
        # Get the coefficients of the antiderivative
        integ_coef = cheb.chebint(self.coef, m=1, lbnd=self.domain[0], k=0)
        return ChebyshevPolynomial(coef=integ_coef, domain=self.domain, window=self.window, symbol=f"{self.symbol}")


def from_coefficients(coef, domain=None, window=None, symbol="x"):
    """Create a Chebyshev polynomial from its coefficients.

    Returns:
        ChebyshevPolynomial: A new Chebyshev polynomial with the given coefficients.
    """
    if len(coef) == 0:
        raise ValueError("Empty coefficients")

    return ChebyshevPolynomial(coef, domain, window, symbol)


def from_values(values, domain=None, window=None, symbol="x"):
    """Create a Chebyshev polynomial from values at Chebyshev points.

    Args:
        values: Values at Chebyshev points.
        domain: The interval on which to define the polynomial.
            If None, the standard interval [-1, 1] is used.
        window: The window for the polynomial.
        symbol: The symbol to use for the polynomial.

    Returns:
        ChebyshevPolynomial: A new Chebyshev polynomial with the given values.
    """
    if len(values) == 0:
        raise ValueError("Empty values")

    coef = vals2coeffs2(values)
    return ChebyshevPolynomial(coef, domain, window, symbol)


def from_roots(roots, domain=None, window=None, symbol="x"):
    """Create a Chebyshev polynomial from its roots."""
    if len(roots) == 0:
        raise ValueError("Empty roots")

    coef = cheb.chebfromroots(roots)
    return ChebyshevPolynomial(coef, domain, window, symbol)


def from_constant(c, domain=None, window=None, symbol="x"):
    """Create a Chebyshev polynomial representing a constant value.

    Args:
        c: The constant value.
        domain: The interval on which to define the polynomial.
            If None, the standard interval [-1, 1] is used.
        window: The window for the polynomial.
        symbol: The symbol to use for the polynomial.

    Returns:
        ChebyshevPolynomial: A new Chebyshev polynomial representing the constant value.
    """
    if not np.isscalar(c):
        raise ValueError("Input must be a scalar value")

    # Convert integer to float to match behavior in other parts of the codebase
    if isinstance(c, int):
        c = float(c)

    return ChebyshevPolynomial([c], domain, window, symbol)
