"""ChebPy: A Python implementation of Chebfun.

ChebPy is a Python package for computing with functions using Chebyshev series
approximations. It provides tools for function approximation, differentiation,
integration, rootfinding, and more, inspired by the MATLAB Chebfun package.

This package contains the core classes and functions that implement ChebPy's functionality,
including fundamental data structures for representing functions, algorithms for numerical
operations, and utilities for working with these objects.

Main components and modules:
- chebfun: Main class for representing functions and creating Chebfun objects
- chebop: Framework for defining and solving differential operators and boundary-value problems
- pwc: Function for creating piecewise-constant Chebfun objects
- UserPreferences: Class for configuring package preferences
- bndfun: Functions on bounded intervals
- chebtech: Chebyshev technology for approximating functions
- trigtech: Fourier technology for approximating functions
- algorithms: Numerical algorithms used throughout the package
- utilities: Helper functions and classes
"""

import importlib.metadata

from .api import chebfun, chebop, pwc
from .settings import ChebPreferences as UserPreferences

__all__ = ["chebfun", "chebop", "pwc", "UserPreferences"]
__version__ = importlib.metadata.version("chebpy")
