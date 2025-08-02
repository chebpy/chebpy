"""Immutable representation of Chebyshev polynomials.

This module provides a dataclass for the immutable representation of Chebyshev
polynomials and various build functions to construct such polynomials.
"""

import numpy as np
import numpy.polynomial.chebyshev as cheb

from .algorithms import (
    vals2coeffs2,
)

class ChebyshevPolynomial(cheb.Chebyshev):
    """Immutable representation of a Chebyshev polynomial.

    This class represents a Chebyshev polynomial using its coefficients in the
    Chebyshev basis. The polynomial is defined on a specific interval.

    Attributes:
        coeffs: The coefficients of the Chebyshev polynomial.
        interval: The interval on which the polynomial is defined.
    """
    def __init__(self, coef, domain=None, window=None, symbol='x'):
        """Initialize a ChebyshevPolynomial object."""
        super().__init__(coef, domain, window, symbol)


    def __call__(self, x):
        """Evaluate the polynomial at the given points.

        Args:
            x: Points at which to evaluate the polynomial.

        Returns:
            The values of the polynomial at the given points.
        """
        return cheb.chebval(x, self.coef)

    def __eq__(self, other):
        """Check if two ChebyshevPolynomial objects are equal."""
        if not isinstance(other, ChebyshevPolynomial):
            return False

        if not np.array_equal(self.domain, other.domain):
            return False

        if not np.array_equal(self.window, other.window):
            return False

        return np.allclose(self.coef, other.coef)

    def __repr__(self):
        """Return a string representation of the ChebyshevPolynomial object."""
        return f"ChebyshevPolynomial(coeffs={self.coef}, domain={self.domain}, window={self.window})"

    def __str__(self):
        """Return a string representation of the ChebyshevPolynomial object."""
        return f"Chebyshev polynomial of degree {self.degree} on {self.domain}"

    def __hash__(self):
        """Return a hash value for the ChebyshevPolynomial object."""
        return hash((self.domain, self.window, self.coef.tobytes()))


def from_coefficients(coef, domain=None, window=None, symbol='x'):
    """Create a Chebyshev polynomial from its coefficients.

    Returns:
        ChebyshevPolynomial: A new Chebyshev polynomial with the given coefficients.
    """
    if len(coef) == 0:
        raise ValueError("Empty coefficients")

    return ChebyshevPolynomial(coef, domain, window, symbol)



def from_values(values, domain=None, window=None, symbol='x'):
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


def from_roots(roots, domain=None, window=None, symbol='x'):
    """Create a Chebyshev polynomial from its roots."""
    if len(roots) == 0:
        raise ValueError("Empty roots")

    coef = cheb.chebfromroots(roots)
    return ChebyshevPolynomial(coef, domain, window, symbol)
