"""Unit-tests for Chebtech2 roots functionality.

This module contains tests for finding the roots of Chebtech2 objects,
including empty, constant, and various polynomial and trigonometric functions.
"""

import numpy as np
import pytest

from chebpy.core.chebtech import Chebtech2

from ..utilities import cos, eps, pi, sin


def test_empty(emptyfun):
    """Test that empty Chebtech2 objects have no roots.

    This test verifies that the roots() method of an empty Chebtech2 object
    returns an empty array.
    """
    assert emptyfun.roots().size == 0


def test_const():
    """Test that constant Chebtech2 objects have no roots.

    This test verifies that the roots() method of constant Chebtech2 objects
    (both zero and non-zero) returns an empty array.
    """
    ff = Chebtech2.initconst(0.0)
    gg = Chebtech2.initconst(2.0)
    assert ff.roots().size == 0
    assert gg.roots().size == 0


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

# Ensure reproducibility
np.random.seed(0)


@pytest.mark.parametrize("f, roots", rootstestfuns)
def test_roots(f, roots, tol=eps):
    """Test that the roots of a function are correctly identified.

    This test verifies that the roots() method of a Chebtech2 object
    correctly identifies the roots of various functions within the
    specified tolerance.

    Args:
        f: Function to find roots of
        roots: Expected roots
        tol: Tolerance for comparison
    """
    ff = Chebtech2.initfun_adaptive(f)
    rts = ff.roots()
    assert np.max(rts - roots) <= tol
