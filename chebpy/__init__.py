"""ChebPy: A Python implementation of Chebfun.

ChebPy is a Python package for computing with functions using Chebyshev series
approximations. It provides tools for function approximation, differentiation,
integration, rootfinding, and more.

The package is inspired by the MATLAB Chebfun package and aims to provide similar
functionality in Python.

Main components:
- chebfun: Function for creating Chebfun objects
- pwc: Function for creating piecewise-constant Chebfun objects
- UserPreferences: Class for configuring package preferences
"""

import importlib.metadata

from .api import chebfun, pwc
from .core.settings import ChebPreferences as UserPreferences

__all__ = ["chebfun", "pwc", "UserPreferences"]
__version__ = importlib.metadata.version("chebpy")
