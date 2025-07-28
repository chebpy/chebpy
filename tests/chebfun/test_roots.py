"""Unit-tests for Chebfun roots functionality.

This module contains tests for finding the roots of Chebfun objects,
including empty, constant, and various polynomial and trigonometric functions.
"""

import pytest
import numpy as np

from chebpy.core.chebfun import Chebfun
from chebpy.core.utilities import Interval

from .conftest import eps, sin, cos, pi


@pytest.fixture
def roots_fixtures():
    """Create Chebfun objects for testing roots functionality.

    This fixture creates several Chebfun objects with different characteristics
    for testing the roots method.

    Returns:
        dict: Dictionary containing:
            f0: Empty Chebfun
            f1: Chebfun representing sin(x)
            f2: Chebfun representing cos(x)
            f3: Chebfun representing x^2 - 0.25
            f4: Chebfun representing x^3 - x
    """
    f0 = Chebfun.initempty()
    f1 = Chebfun.initfun_adaptive(sin, [-1, 1])
    f2 = Chebfun.initfun_adaptive(cos, [0, 2])
    f3 = Chebfun.initfun_adaptive(lambda x: x**2 - 0.25, [-1, 1])
    f4 = Chebfun.initfun_adaptive(lambda x: x**3 - x, [-1, 1])

    return {"f0": f0, "f1": f1, "f2": f2, "f3": f3, "f4": f4}


def test_roots_empty(roots_fixtures):
    """Test roots method on an empty Chebfun.

    This test verifies that the roots method on an empty Chebfun
    returns an empty array.

    Args:
        roots_fixtures: Fixture providing test Chebfun objects.
    """
    f0 = roots_fixtures["f0"]
    assert f0.roots().size == 0


def test_roots_sin(roots_fixtures):
    """Test roots method on sin(x).

    This test verifies that the roots method correctly identifies
    the roots of sin(x) in the interval [-1, 1].

    Args:
        roots_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = roots_fixtures["f1"]
    roots = f1.roots()
    assert roots.size == 1
    assert abs(roots[0]) < 1e-10  # sin(0) = 0


def test_roots_cos(roots_fixtures):
    """Test roots method on cos(x).

    This test verifies that the roots method correctly identifies
    the roots of cos(x) in the interval [-1, 1].

    Args:
        roots_fixtures: Fixture providing test Chebfun objects.
    """
    f2 = roots_fixtures["f2"]
    roots = f2.roots()
    assert roots.size == 1
    assert abs(roots[0] - pi/2) < 1e-10  # cos(pi/2) = 0


def test_roots_quadratic(roots_fixtures):
    """Test roots method on a quadratic function.

    This test verifies that the roots method correctly identifies
    the roots of x^2 - 0.25 in the interval [-1, 1].

    Args:
        roots_fixtures: Fixture providing test Chebfun objects.
    """
    f3 = roots_fixtures["f3"]
    roots = f3.roots()
    assert roots.size == 2
    assert abs(roots[0] + 0.5) < 1e-10  # x = -0.5
    assert abs(roots[1] - 0.5) < 1e-10  # x = 0.5


def test_roots_cubic(roots_fixtures):
    """Test roots method on a cubic function.

    This test verifies that the roots method correctly identifies
    the roots of x^3 - x in the interval [-1, 1].

    Args:
        roots_fixtures: Fixture providing test Chebfun objects.
    """
    f4 = roots_fixtures["f4"]
    roots = f4.roots()
    assert roots.size == 3
    assert abs(roots[0] + 1) < 1e-10    # x = -1
    assert abs(roots[1]) < 1e-10        # x = 0
    assert abs(roots[2] - 1) < 1e-10    # x = 1


def test_roots_const():
    """Test roots method on constant Chebfun objects.

    This test verifies that the roots method correctly handles
    constant Chebfun objects, both zero and non-zero.
    """
    # Non-zero constant should have no roots
    f_nonzero = Chebfun.initconst(1.0, [-1, 1])
    assert f_nonzero.roots().size == 0

    # Zero constant is a special case - technically every point is a root
    # but the implementation should return an empty array
    f_zero = Chebfun.initconst(0.0, [-1, 1])
    assert f_zero.roots().size == 0


def test_roots_multiple_intervals():
    """Test roots method on a Chebfun with multiple intervals.

    This test verifies that the roots method correctly identifies
    roots across multiple intervals.
    """
    # Create a Chebfun with sin(2*pi*x) on a domain with multiple breakpoints
    # This function has roots at x = 0, 0.5, 1, 1.5, etc.
    f = Chebfun.initfun_adaptive(lambda x: sin(2*pi*x), [-1, 0, 1])

    # Find the roots
    roots = f.roots()

    # Expected roots at x = -1, -0.5, 0, 0.5, 1
    # The roots at the breakpoints (0) should only be counted once
    expected_roots = np.array([-1, -0.5, 0, 0.5, 1])

    # Check that we have the correct number of roots
    assert roots.size == expected_roots.size

    # Check that each expected root is found (within tolerance)
    sorted_roots = np.sort(roots)
    assert np.allclose(sorted_roots, expected_roots, atol=1e-10)


def test_roots_high_frequency():
    """Test roots method on a high-frequency function.

    This test verifies that the roots method can accurately find
    all roots of a high-frequency function.
    """
    # Create a Chebfun for sin(10*pi*x) which has 20 roots in [-1, 1]
    f = Chebfun.initfun_adaptive(lambda x: sin(10*pi*x), [-1, 1])

    # Find the roots
    roots = f.roots()

    # Should have 21 roots at x = k/10 for k = -10, -9, ..., 9, 10
    expected_roots = np.linspace(-1, 1, 21)

    assert roots.size == expected_roots.size
    assert np.allclose(np.sort(roots), expected_roots, atol=1e-10)
