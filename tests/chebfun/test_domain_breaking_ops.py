"""Unit-tests for Chebfun domain-breaking operations.

This module contains tests for operations that can break the domain of a Chebfun,
such as maximum and minimum functions.
"""

import numpy as np
import pytest

from chebpy import chebfun
from chebpy.chebfun import Chebfun

from .conftest import eps


@pytest.fixture
def domain_breaking_fixtures() -> dict:
    """Create Chebfun objects for testing domain-breaking operations.

    This fixture creates several Chebfun objects with different characteristics
    for testing operations that can break the domain.

    Returns:
        dict: Dictionary containing:
            x: Chebfun representing the identity function on a multi-piece domain
            y: Chebfun representing a constant function on the same domain as x
    """
    x = chebfun("x", np.linspace(-2, 3, 11))
    y = chebfun(2, x.domain)

    return {"x": x, "y": y}


def test_maximum_multipiece(domain_breaking_fixtures: dict) -> None:
    """Test the maximum method with a multi-piece Chebfun and a constant.

    This test verifies that the maximum method correctly computes the
    maximum of a multi-piece Chebfun and a constant value.

    Args:
        domain_breaking_fixtures: Fixture providing test Chebfun objects.
    """
    x = domain_breaking_fixtures["x"]
    y = domain_breaking_fixtures["y"]

    # Compute x^2 and its maximum with 1.5
    g = (x**y).maximum(1.5)
    t = np.linspace(-2, 3, 2001)

    # Define the expected result function
    def f(x):
        return np.maximum(x**2, 1.5)

    # Check that the result matches the expected function
    assert np.max(np.abs(f(t) - g(t))) <= 1e1 * eps


def test_minimum_multipiece(domain_breaking_fixtures: dict) -> None:
    """Test the minimum method with a multi-piece Chebfun and a constant.

    This test verifies that the minimum method correctly computes the
    minimum of a multi-piece Chebfun and a constant value.

    Args:
        domain_breaking_fixtures: Fixture providing test Chebfun objects.
    """
    x = domain_breaking_fixtures["x"]
    y = domain_breaking_fixtures["y"]

    # Compute x^2 and its minimum with 1.5
    g = (x**y).minimum(1.5)
    t = np.linspace(-2, 3, 2001)

    # Define the expected result function
    def f(x):
        return np.minimum(x**2, 1.5)

    # Check that the result matches the expected function
    assert np.max(np.abs(f(t) - g(t))) <= 1e1 * eps


@pytest.mark.parametrize(
    "domain,tol",
    [
        ([-1, 1], eps),
        ([-1, 0, 1], eps),
        ([-2, 0, 3], eps),
    ],
)
def test_maximum_identity_constant(domain: list, tol: float) -> None:
    """Test the maximum method with the identity function and a constant.

    This test verifies that the maximum method correctly computes the
    maximum of the identity function and a constant value.

    Args:
        domain: Domain for the Chebfun objects
        tol: Tolerance for the comparison
    """
    # Create the identity function and a constant
    x = chebfun("x", domain)
    c = 0

    # Compute the maximum
    g = x.maximum(c)
    xx = np.linspace(domain[0], domain[-1], 1001)

    # Define the expected result function
    def f(x):
        return np.maximum(x, c)

    # Check that the result matches the expected function
    assert np.max(np.abs(f(xx) - g(xx))) <= tol


@pytest.mark.parametrize(
    "domain,tol",
    [
        ([-1, 1], eps),
        ([-1, 0, 1], eps),
        ([-2, 0, 3], eps),
    ],
)
def test_minimum_identity_constant(domain: list, tol: float) -> None:
    """Test the minimum method with the identity function and a constant.

    This test verifies that the minimum method correctly computes the
    minimum of the identity function and a constant value.

    Args:
        domain: Domain for the Chebfun objects
        tol: Tolerance for the comparison
    """
    # Create the identity function and a constant
    x = chebfun("x", domain)
    c = 0

    # Compute the minimum
    g = x.minimum(c)
    xx = np.linspace(domain[0], domain[-1], 1001)

    # Define the expected result function
    def f(x):
        return np.minimum(x, c)

    # Check that the result matches the expected function
    assert np.max(np.abs(f(xx) - g(xx))) <= tol


@pytest.mark.parametrize(
    "domain,tol",
    [
        ([-1, 1], eps),
        ([-1, 0, 1], eps),
        ([-2, 0, 3], eps),
    ],
)
def test_maximum_sin_cos(domain: list, tol: float) -> None:
    """Test the maximum method with sine and cosine functions.

    This test verifies that the maximum method correctly computes the
    maximum of sine and cosine functions.

    Args:
        domain: Domain for the Chebfun objects
        tol: Tolerance for the comparison
    """
    # Create sine and cosine functions
    f1 = chebfun(np.sin, domain)
    f2 = chebfun(np.cos, domain)

    # Compute the maximum
    g = f1.maximum(f2)
    xx = np.linspace(domain[0], domain[-1], 1001)

    # Define the expected result function
    def f(x):
        return np.maximum(np.sin(x), np.cos(x))

    # Check that the result matches the expected function
    vscl = max([f1.vscale, f2.vscale])
    hscl = max([f1.hscale, f2.hscale])
    lscl = max([fun.size for fun in np.append(f1.funs, f2.funs)])
    assert np.max(np.abs(f(xx) - g(xx))) <= vscl * hscl * lscl * tol


@pytest.mark.parametrize(
    "domain,tol",
    [
        ([-1, 1], eps),
        ([-1, 0, 1], eps),
        ([-2, 0, 3], eps),
    ],
)
def test_minimum_sin_cos(domain: list, tol: float) -> None:
    """Test the minimum method with sine and cosine functions.

    This test verifies that the minimum method correctly computes the
    minimum of sine and cosine functions.

    Args:
        domain: Domain for the Chebfun objects
        tol: Tolerance for the comparison
    """
    # Create sine and cosine functions
    f1 = chebfun(np.sin, domain)
    f2 = chebfun(np.cos, domain)

    # Compute the minimum
    g = f1.minimum(f2)
    xx = np.linspace(domain[0], domain[-1], 1001)

    # Define the expected result function
    def f(x):
        return np.minimum(np.sin(x), np.cos(x))

    # Check that the result matches the expected function
    vscl = max([f1.vscale, f2.vscale])
    hscl = max([f1.hscale, f2.hscale])
    lscl = max([fun.size for fun in np.append(f1.funs, f2.funs)])
    assert np.max(np.abs(f(xx) - g(xx))) <= vscl * hscl * lscl * tol


def test_maximum_empty() -> None:
    """Test the maximum method with an empty Chebfun.

    This test verifies that the maximum method correctly handles
    empty Chebfun objects.
    """
    # Create an empty Chebfun and a regular Chebfun
    f_empty = Chebfun.initempty()
    f = chebfun(np.sin, [-1, 1])

    # Check that maximum with an empty Chebfun returns an empty Chebfun
    assert f_empty.maximum(f).isempty
    assert f.maximum(f_empty).isempty
    assert f_empty.maximum(f_empty).isempty


def test_minimum_empty() -> None:
    """Test the minimum method with an empty Chebfun.

    This test verifies that the minimum method correctly handles
    empty Chebfun objects.
    """
    # Create an empty Chebfun and a regular Chebfun
    f_empty = Chebfun.initempty()
    f = chebfun(np.sin, [-1, 1])

    # Check that minimum with an empty Chebfun returns an empty Chebfun
    assert f_empty.minimum(f).isempty
    assert f.minimum(f_empty).isempty
    assert f_empty.minimum(f_empty).isempty
