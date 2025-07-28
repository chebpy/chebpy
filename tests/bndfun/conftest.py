"""Configuration and fixtures for pytest in the bndfun tests.

This module contains fixtures and helper functions specific to testing
the bndfun class. It provides fixtures for common test objects and
helper functions for testing algebraic operations.
"""

import numpy as np
import pytest

from chebpy.core.bndfun import Bndfun
from chebpy.core.settings import DefaultPreferences

# aliases
eps = DefaultPreferences.eps
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp

@pytest.fixture
def emptyfun():
    """Create an empty Bndfun function for testing."""
    return Bndfun.initempty()
