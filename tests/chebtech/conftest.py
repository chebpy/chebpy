"""Configuration and fixtures for pytest in the chebtech tests.

This module contains fixtures and helper functions specific to testing
the Chebtech2 class. It provides fixtures for common test objects and
helper functions for testing algebraic operations.
"""

import numpy as np
import pytest

from chebpy.core.chebtech import Chebtech2
from chebpy.core.settings import DefaultPreferences

# aliases
eps = DefaultPreferences.eps
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp


@pytest.fixture
def emptyfun() -> Chebtech2:
    """Create an empty Chebtech2 function for testing.

    This fixture creates an empty Chebtech2 object that can be used
    to test the behavior of algebraic operations on empty functions.

    Returns:
        Chebtech2: An empty Chebtech2 object
    """
    return Chebtech2.initempty()


@pytest.fixture
def random_points() -> np.ndarray:
    """Create an array of random points in [-1, 1] for testing.

    This fixture creates an array of 1000 random points in the interval [-1, 1]
    that can be used for evaluating and testing Chebtech2 objects.

    Returns:
        numpy.ndarray: Array of 1000 random points in [-1, 1]
    """
    np.random.seed(0)  # Ensure reproducibility
    return -1 + 2 * np.random.rand(1000)


def binary_op_tester(f: callable, g: callable, binop: callable, nf: int, ng: int) -> tuple:
    """Test binary operations between two Chebtech objects.

    This function creates Chebtech2 objects from the given functions and tests
    that the binary operation between them produces the expected result within
    a tolerance that scales with the size and scale of the functions.

    Args:
        f (callable): First function
        g (callable): Second function
        binop (callable): Binary operator function (e.g., operator.add)
        nf (int): Number of points for the first function
        ng (int): Number of points for the second function

    Returns:
        tuple: The Chebtech2 objects and the result of the binary operation
    """
    ff = Chebtech2.initfun_fixedlen(f, nf)
    gg = Chebtech2.initfun_fixedlen(g, ng)
    xx = np.linspace(-1, 1, 1000)

    def fg_expected(x):
        return binop(f(x), g(x))

    fg = binop(ff, gg)

    vscl = max([ff.vscale, gg.vscale, fg.vscale])
    lscl = max([ff.size, gg.size, fg.size])
    tol = 5e1 * eps * lscl * vscl

    return ff, gg, fg, fg_expected, xx, tol


def unary_op_tester(unaryop: callable, f: callable, nf: int) -> tuple:
    """Test unary operations on a Chebtech object.

    This function creates a Chebtech2 object from the given function and tests
    that the unary operation on it produces the expected result within a tolerance
    that scales with the size and scale of the function.

    Args:
        unaryop (callable): Unary operator function (e.g., operator.neg)
        f (callable): Function to test
        nf (int): Number of points for the function

    Returns:
        tuple: The Chebtech2 object, the result of the unary operation, and other test data
    """
    ff = Chebtech2.initfun_fixedlen(f, nf)
    xx = np.linspace(-1, 1, 1000)

    def gg(x):
        return unaryop(f(x))

    gg_result = unaryop(ff)

    vscl = max([ff.vscale, gg_result.vscale])
    lscl = max([ff.size, gg_result.size])
    tol = 5e1 * eps * lscl * vscl

    return ff, gg_result, gg, xx, tol


def scaled_tol(n: int) -> float:
    """Calculate a scaled tolerance based on the size of the input.

    This function returns a tolerance that increases with the size of the input,
    which is useful for tests where the expected error grows with problem size.

    Args:
        n (int): Size parameter, typically the length of an array.

    Returns:
        float: Scaled tolerance value.
    """
    tol = 5e1 * eps if n < 20 else np.log(n) ** 2.5 * eps
    return tol
