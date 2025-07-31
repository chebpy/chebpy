"""Generic test functions for root-finding operations.

This module contains test functions for root-finding operations that can be used
with any type of function object (Bndfun, Chebfun, or Chebtech2). These tests
focus on operations with empty function objects.
"""
import numpy as np
from ..utilities import cos, pi, sin

def test_empty(emptyfun):
    """Test the roots method on an empty Bndfun."""
    assert emptyfun.roots().size == 0

# Define test functions and their expected roots
rootstestfuns = [
    (lambda x: 3 * x + 2.0, np.array([-2 / 3])),
    (lambda x: x**2, np.array([0.0, 0.0])),
    (lambda x: x**2 + 0.2 * x - 0.08, np.array([-0.4, 0.2])),
    (lambda x: sin(x), np.array([0])),
    (lambda x: cos(2 * pi * x), np.array([-0.75, -0.25, 0.25, 0.75])),
    (lambda x: sin(100 * pi * x), np.linspace(-1, 1, 201)),
    (lambda x: sin(5 * pi / 2 * x), np.array([-0.8, -0.4, 0, 0.4, 0.8])),
]
