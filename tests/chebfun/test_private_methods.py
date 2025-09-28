"""Unit-tests for Chebfun private methods.

This module contains tests for the private methods of Chebfun objects,
particularly the _break method which is used to modify the domain of a Chebfun.
"""

import numpy as np
import pytest

from chebpy.chebfun import Chebfun
from chebpy.utilities import Domain

from .conftest import eps


@pytest.fixture
def private_methods_fixtures():
    """Create Chebfun objects for testing private methods.

    This fixture creates several Chebfun objects with different domains
    for testing the private methods of Chebfun.

    Returns:
        dict: Dictionary containing:
            f1: Chebfun representing sin(x-0.1) on domain [-2, 0, 3]
            f2: Chebfun representing sin(x-0.1) on domain with 5 breakpoints
    """

    def f(x):
        return np.sin(x - 0.1)

    f1 = Chebfun.initfun_adaptive(f, [-2, 0, 3])
    f2 = Chebfun.initfun_adaptive(f, np.linspace(-2, 3, 5))

    return {"f1": f1, "f2": f2}


def test_break_1(private_methods_fixtures):
    """Test the _break method with a new domain that has interior breakpoints.

    This test verifies that the _break method correctly modifies the domain
    of a Chebfun to include new breakpoints, while preserving the function values.

    Args:
        private_methods_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = private_methods_fixtures["f1"]
    altdom = Domain([-2, -1, 1, 2, 3])
    newdom = f1.domain.union(altdom)
    f1_new = f1._break(newdom)

    # Check that the domain is correctly updated
    assert f1_new.domain == newdom
    assert f1_new.domain != altdom
    assert f1_new.domain != f1.domain

    # Check that the function values are preserved
    xx = np.linspace(-2, 3, 1000)
    error = np.max(np.abs(f1(xx) - f1_new(xx)))
    assert error <= 3 * eps


def test_break_2(private_methods_fixtures):
    """Test the _break method with a new domain that has the same endpoints.

    This test verifies that the _break method correctly modifies the domain
    of a Chebfun to include new breakpoints, even when the endpoints remain the same.

    Args:
        private_methods_fixtures: Fixture providing test Chebfun objects.
    """
    f1 = private_methods_fixtures["f1"]
    altdom = Domain([-2, 3])
    newdom = f1.domain.union(altdom)
    f1_new = f1._break(newdom)

    # Check that the domain is correctly updated
    assert f1_new.domain == newdom
    assert f1_new.domain != altdom

    # Check that the function values are preserved
    xx = np.linspace(-2, 3, 1000)
    error = np.max(np.abs(f1(xx) - f1_new(xx)))
    assert error <= 3 * eps


def test_break_3(private_methods_fixtures):
    """Test the _break method with a new domain that has many breakpoints.

    This test verifies that the _break method correctly modifies the domain
    of a Chebfun to include many new breakpoints, while preserving the function values.

    Args:
        private_methods_fixtures: Fixture providing test Chebfun objects.
    """
    f2 = private_methods_fixtures["f2"]
    altdom = Domain(np.linspace(-2, 3, 1000))
    newdom = f2.domain.union(altdom)
    f2_new = f2._break(newdom)

    # Check that the domain is correctly updated
    assert f2_new.domain == newdom
    assert f2_new.domain != altdom
    assert f2_new.domain != f2.domain

    # Check that the function values are preserved
    xx = np.linspace(-2, 3, 1000)
    error = np.max(np.abs(f2(xx) - f2_new(xx)))
    assert error <= 3 * eps


def test_break_identity():
    """Test the _break method with the same domain.

    This test verifies that the _break method returns a copy of the original
    Chebfun when the new domain is the same as the original domain.
    """
    # Create a Chebfun with a simple domain
    f = Chebfun.initfun_adaptive(np.sin, [-1, 1])

    # Break with the same domain
    f_new = f._break(f.domain)

    # Check that the domains are the same
    assert f_new.domain == f.domain

    # Check that the function values are preserved
    xx = np.linspace(-1, 1, 1000)
    error = np.max(np.abs(f(xx) - f_new(xx)))
    assert error <= eps

    # Check that f_new is a copy, not the same object
    assert f is not f_new


def test_break_with_tolerance():
    """Test the _break method with breakpoints that differ by small amounts.

    This test verifies that the _break method correctly handles domains with
    breakpoints that differ by small amounts (within tolerance).
    """
    # Create a Chebfun with a simple domain
    f = Chebfun.initfun_adaptive(np.sin, [-1, 0, 1])

    # Create a domain with breakpoints that differ slightly from the original
    tol = 0.8 * eps
    altdom = Domain([-1 - tol, 0 + tol, 1 - tol])

    # Break with the new domain
    newdom = f.domain.union(altdom)

    # Check that the domains are considered equal (within tolerance)
    assert newdom == f.domain

    # Since the domains are equal within tolerance, we expect _break to return
    # either a copy of the original function or an empty Chebfun
    f_new = f._break(newdom)
    assert f_new.isempty


def test_break_multipiece():
    """Test the _break method with a multi-piece Chebfun.

    This test verifies that the _break method correctly handles Chebfun objects
    with multiple pieces, preserving the function values across all pieces.
    """
    # Create a multi-piece Chebfun
    f = Chebfun.initfun_adaptive(np.sin, [-2, -1, 0, 1, 2])

    # Create a new domain with additional breakpoints
    newdom = Domain([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])

    # Break with the new domain
    f_new = f._break(newdom)

    # Check that the domain is correctly updated
    assert f_new.domain == newdom

    # Check that the function values are preserved
    xx = np.linspace(-2, 2, 1000)
    error = np.max(np.abs(f(xx) - f_new(xx)))
    assert error <= 3 * eps
