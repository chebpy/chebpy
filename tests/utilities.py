"""Utility functions for testing chebfun."""
import numpy as np

from chebpy.core.settings import DefaultPreferences

# aliases
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
eps = DefaultPreferences.eps

def scaled_tol(n: int) -> float:
    """Calculate a scaled tolerance based on the size of the input.

    This function returns a tolerance that increases with the size of the input,
    which is useful for tests where the expected error grows with problem size.

    Args:
        n (int): Size parameter, typically the length of an array.

    Returns:
        float: Scaled tolerance value.
    """
    tol = 5e1 * eps if n < 20 else np.log(n) ** 2.5 * eps
    return tol


def joukowsky(z):
    """Apply the Joukowsky transformation to z.

    The Joukowsky transformation maps the unit circle to an ellipse and is used
    in complex analysis and fluid dynamics. It is defined as f(z) = 0.5 * (z + 1/z).

    Args:
        z (complex or numpy.ndarray): Complex number or array of complex numbers.

    Returns:
        complex or numpy.ndarray: Result of the Joukowsky transformation.
    """
    return 0.5 * (z + 1 / z)
