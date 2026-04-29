"""ChebPy: A Python implementation of Chebfun.

ChebPy is a Python package for computing with functions using Chebyshev series
approximations. It provides tools for function approximation, differentiation,
integration, rootfinding, and more, inspired by the MATLAB Chebfun package.

This package contains the core classes and functions that implement ChebPy's functionality,
including fundamental data structures for representing functions, algorithms for numerical
operations, and utilities for working with these objects.

Main components and modules:
- chebfun: Main class for representing functions and creating Chebfun objects
- pwc: Function for creating piecewise-constant Chebfun objects
- UserPreferences: Class for configuring package preferences
- bndfun: Functions on bounded intervals
- chebtech: Chebyshev technology for approximating functions
- algorithms: Numerical algorithms used throughout the package
- utilities: Helper functions and classes
"""

import importlib.metadata

from .api import chebfun, chebpts, pwc, trigfun
from .compactfun import CompactFun
from .gpr import gpr
from .quasimatrix import Quasimatrix, polyfit
from .settings import ChebPreferences as UserPreferences
from .singfun import Singfun
from .trigtech import Trigtech

__all__ = [
    "CompactFun",
    "Quasimatrix",
    "Singfun",
    "Trigtech",
    "UserPreferences",
    "chebfun",
    "chebpts",
    "gpr",
    "polyfit",
    "pwc",
    "trigfun",
]
__version__ = importlib.metadata.version("chebfun")
