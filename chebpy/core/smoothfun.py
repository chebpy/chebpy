"""Abstract base class for smooth functions on the interval [-1, 1].

This module defines the Smoothfun class, which is an abstract base class for
representing smooth functions on the standard interval [-1, 1]. It inherits from
Onefun and serves as a base class for specific implementations of smooth function
representations.
"""

from abc import ABC

from .onefun import Onefun


class Smoothfun(Onefun, ABC):
    """Abstract base class for smooth functions on the interval [-1, 1].

    This class extends the Onefun abstract base class to specifically represent
    smooth functions on the standard interval [-1, 1]. Smoothness properties
    enable certain operations and optimizations that are not possible with
    general functions.

    Concrete subclasses must implement all the abstract methods defined in the
    Onefun base class.
    """
    pass
