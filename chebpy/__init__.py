"""ChebPy: A Python implementation of Chebfun.

ChebPy is a Python package for numerical computing with functions. It is inspired
by the MATLAB package Chebfun and provides similar functionality for working with
functions rather than numbers. ChebPy represents functions using Chebyshev series
and allows for operations such as integration, differentiation, root-finding, and more.

Attributes:
    __version__: The current version of the ChebPy package.
"""

__version__ = "0.4.3.3"

from .api import chebfun, pwc
from .core.settings import ChebPreferences as UserPreferences

__all__ = ["chebfun", "pwc", "UserPreferences"]
